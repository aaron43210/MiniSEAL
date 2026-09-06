import os
import math
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tokenizers import Tokenizer
from src.model.config import ModelConfig
from src.model.llm import DecoderOnlyTransformer
from src.data.dpo_dataset import DPODataset

def setup():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size

def cleanup():
    dist.destroy_process_group()

def get_batch_logps(logits, labels, ignore_index=-100):
    """
    Computes log probabilities of the given labels under the given logits.
    logits: [B, T, V]
    labels: [B, T]
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_index)
    ce_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    ce_loss = ce_loss.view(shift_labels.shape)
    loss_mask = shift_labels != ignore_index
    logps = (-ce_loss * loss_mask).sum(dim=-1)
    
    return logps

def dpo_loss(
    policy_chosen_logps,
    policy_rejected_logps,
    ref_chosen_logps,
    ref_rejected_logps,
    beta=0.1
):
    """
    Computes the DPO loss for a batch.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits).mean()
    
    return loss, (pi_logratios > ref_logratios).float().mean()

def load_config(config_path="configs/dpo_dgx_a100.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_lr(it, train_cfg):
    warmup_steps = train_cfg["warmup_steps"]
    max_steps = train_cfg["max_steps"]
    lr = train_cfg["learning_rate"]
    min_lr = train_cfg["min_lr"]
    if it < warmup_steps:
        return lr * it / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)

def main():
    rank, local_rank, world_size = setup()
    is_master = rank == 0

    train_cfg = load_config()
    out_dir = train_cfg["out_dir"]
    if is_master:
        os.makedirs(out_dir, exist_ok=True)
        print("=== STARTING DPO ===")

    torch.manual_seed(train_cfg["seed"] + rank)
    
    tokenizer = Tokenizer.from_file(train_cfg["tokenizer_path"])

    if is_master:
        print(f"Loading base SFT model weights from {train_cfg['pretrained_model_path']}")
    
    checkpoint = torch.load(train_cfg["pretrained_model_path"], map_location="cpu")
    model_config = ModelConfig(**checkpoint["model_config"])
    
    # Policy Model (Active)
    model = DecoderOnlyTransformer(model_config)
    model.load_state_dict(checkpoint["model"])
    model.to(local_rank)
    
    # Reference Model (Frozen)
    ref_model = DecoderOnlyTransformer(model_config)
    ref_model.load_state_dict(checkpoint["model"])
    ref_model.to(local_rank)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
        
    model = DDP(model, device_ids=[local_rank])

    dataset = DPODataset(
        data_path=train_cfg["dpo_data_path"],
        tokenizer=tokenizer,
        max_len=model_config.block_size
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg["micro_batch_size"],
        sampler=sampler,
        num_workers=train_cfg["num_workers"],
        pin_memory=True
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        betas=(train_cfg["beta1"], train_cfg["beta2"]),
        weight_decay=train_cfg["weight_decay"]
    )

    grad_accum_steps = train_cfg["grad_accum_steps"]
    beta = train_cfg.get("beta", 0.1)

    precision = train_cfg.get("precision", "fp16")
    ptdtype = {'fp32': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[precision]
    use_amp = (precision in ["fp16", "bf16"])
    scaler = torch.amp.GradScaler('cuda', enabled=(precision == "fp16" and use_amp))

    model.train()
    step = 0
    data_iter = iter(dataloader)
    
    total_loss = 0.0
    total_acc = 0.0

    while step < train_cfg["max_steps"]:
        lr = get_lr(step, train_cfg)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.zero_grad(set_to_none=True)
        
        batch_loss = 0.0
        batch_acc = 0.0
        
        for micro_step in range(grad_accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                sampler.set_epoch(step)
                data_iter = iter(dataloader)
                batch = next(data_iter)

            chosen_ids = batch["chosen_input_ids"].to(local_rank)
            chosen_labels = batch["chosen_labels"].to(local_rank)
            rejected_ids = batch["rejected_input_ids"].to(local_rank)
            rejected_labels = batch["rejected_labels"].to(local_rank)

            with torch.amp.autocast('cuda', dtype=ptdtype, enabled=use_amp):
                pi_chosen_logits, _ = model(chosen_ids)
                pi_rejected_logits, _ = model(rejected_ids)
                
                with torch.no_grad():
                    ref_chosen_logits, _ = ref_model(chosen_ids)
                    ref_rejected_logits, _ = ref_model(rejected_ids)

                pi_chosen_logps = get_batch_logps(pi_chosen_logits, chosen_labels)
                pi_rejected_logps = get_batch_logps(pi_rejected_logits, rejected_labels)
                
                ref_chosen_logps = get_batch_logps(ref_chosen_logits, chosen_labels)
                ref_rejected_logps = get_batch_logps(ref_rejected_logits, rejected_labels)
                
                loss, acc = dpo_loss(
                    pi_chosen_logps, pi_rejected_logps,
                    ref_chosen_logps, ref_rejected_logps,
                    beta=beta
                )
                
                loss = loss / grad_accum_steps
                
            scaler.scale(loss).backward()
            
            batch_loss += loss.item() * grad_accum_steps
            batch_acc += acc.item() / grad_accum_steps

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
        scaler.step(optimizer)
        scaler.update()

        total_loss += batch_loss
        total_acc += batch_acc
        step += 1

        if step % train_cfg["log_interval"] == 0 and is_master:
            avg_loss = total_loss / train_cfg["log_interval"]
            avg_acc = total_acc / train_cfg["log_interval"]
            print(f"Step {step:4d} | loss: {avg_loss:.4f} | chosen_acc: {avg_acc:.2f} | lr: {lr:.2e}")
            total_loss = 0.0
            total_acc = 0.0

    if is_master:
        print("Saving final DPO model...")
        checkpoint = {
            "model": model.module.state_dict(),
            "model_config": model_config.__dict__,
            "optimizer": optimizer.state_dict(),
            "step": step
        }
        torch.save(checkpoint, os.path.join(out_dir, "latest_dpo.pt"))
        print("Done!")

    cleanup()

if __name__ == "__main__":
    main()
