"""
Phase 3: Pretraining script.

Run single GPU:
    python -m src.train.pretrain --config configs/pretrain_200m.yaml

Run multi-GPU:
    torchrun --nproc_per_node=8 -m src.train.pretrain --config configs/pretrain_200m.yaml

Input:
    tokenized train.bin / val.bin
    tokenizer config
    model config

Output:
    checkpoints:
        latest.pt
        ckpt_step_x.pt

    logs:
        train_log.jsonl
"""

import argparse
import json
import math
import os
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

# Add project root to python path so 'src' can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.common.utils import ensure_dir, is_rank_zero, load_json, set_seed
from src.data.sft_dataset import SFTDataset
from src.model.config import ModelConfig
from src.model.llm import DecoderOnlyTransformer
from tokenizers import Tokenizer


def setup_distributed():
    """
    Output:
        distributed: bool
        rank: global rank
        local_rank: GPU index on current node
        world_size: number of processes
        device: torch.device
    """
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ

    if distributed:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    return distributed, rank, local_rank, world_size, device


def cleanup_distributed(distributed: bool):
    if distributed:
        dist.destroy_process_group()


def get_lr(step: int, cfg: dict) -> float:
    """
    Cosine learning rate schedule with warmup.

    Input:
        step: current training step.
        cfg: training config.

    Output:
        learning rate.
    """
    max_lr = cfg["learning_rate"]
    min_lr = cfg["min_lr"]
    warmup_steps = cfg["warmup_steps"]
    max_steps = cfg["max_steps"]

    if step < warmup_steps:
        return max_lr * step / max(1, warmup_steps)

    if step > max_steps:
        return min_lr

    decay_ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    return min_lr + coeff * (max_lr - min_lr)


def configure_optimizer(model: nn.Module, train_cfg: dict):
    """
    AdamW optimizer with separate weight decay groups.

    Weight decay is applied to matrix weights, not norms/biases.
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    optim_groups = [
        {
            "params": decay_params,
            "weight_decay": train_cfg["weight_decay"],
        },
        {
            "params": no_decay_params,
            "weight_decay": 0.0,
        },
    ]

    fused_available = torch.cuda.is_available()

    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=train_cfg["learning_rate"],
        betas=(train_cfg["beta1"], train_cfg["beta2"]),
        fused=fused_available,
    )

    return optimizer


def save_checkpoint(
    model,
    optimizer,
    step: int,
    cfg: dict,
    out_dir: str,
):
    """
    Input:
        model: DDP or normal model.
        optimizer: AdamW optimizer.
        step: current step.
        cfg: full config.
        out_dir: checkpoint directory.

    Output:
        Saves checkpoint.
    """
    if not is_rank_zero():
        return

    raw_model = model.module if hasattr(model, "module") else model
    
    checkpoint = {
        "model": raw_model.state_dict(),
        "model_config": raw_model.config.__dict__,
        "step": step,
    }
    # Save SFT model
    save_path = os.path.join(out_dir, "model_sft.pt")
    torch.save(checkpoint, save_path)
    print(f"Saved SFT checkpoint: {save_path}")

    ensure_dir(out_dir)

    latest_path = Path(out_dir) / "latest.pt"
    step_path = Path(out_dir) / f"ckpt_step_{step}.pt"

    torch.save(checkpoint, latest_path)
    torch.save(checkpoint, step_path)

    print(f"Saved checkpoint: {latest_path}")


@torch.no_grad()
def estimate_loss(
    model,
    val_loader,
    device,
    eval_iters: int,
    precision: str,
    distributed: bool,
):
    """
    Input:
        model: language model.
        val_loader: validation DataLoader.
        eval_iters: number of validation batches.

    Output:
        mean validation loss.
    """
    model.eval()
    losses = []

    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    use_amp = device.type == "cuda"

    iterator = iter(val_loader)

    for _ in range(eval_iters):
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(val_loader)
            x, y = next(iterator)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.amp.autocast(
            device_type=device.type if use_amp else "cpu",
            dtype=amp_dtype if use_amp else torch.float32,
            enabled=use_amp,
        ):
            logits, _ = model(x)
            # Shift for autoregressive SFT loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = y[..., 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1), 
                ignore_index=-100
            )

        losses.append(loss.detach())

    loss_tensor = torch.stack(losses).mean()

    if distributed:
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)

    model.train()

    return loss_tensor.item()


@torch.no_grad()
def sample_text(
    model,
    tokenizer_path: str,
    prompt: str,
    device,
    max_new_tokens: int = 120,
):
    """
    Input:
        model: trained model.
        tokenizer_path: tokenizer.json path.
        prompt: text prompt.

    Output:
        generated sample text.
    """
    from tokenizers import Tokenizer

    raw_model = model.module if hasattr(model, "module") else model
    raw_model.eval()

    tokenizer = Tokenizer.from_file(tokenizer_path)
    ids = tokenizer.encode(prompt).ids

    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    out = raw_model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=0.8,
        top_k=50,
    )

    text = tokenizer.decode(out[0].tolist())
    raw_model.train()
    return text


def append_log(out_dir: str, record: dict):
    """
    Input:
        record: dictionary of metrics.

    Output:
        Appends JSON line to train_log.jsonl.
    """
    if not is_rank_zero():
        return

    ensure_dir(out_dir)
    log_path = Path(out_dir) / "train_log.jsonl"

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    distributed, rank, local_rank, world_size, device = setup_distributed()

    set_seed(cfg["seed"] + rank)

    train_cfg = cfg
    
    out_dir = train_cfg["out_dir"]
    ensure_dir(out_dir)

    tokenizer = Tokenizer.from_file(train_cfg["tokenizer_path"])
    
    # We must load the base model checkpoint for SFT
    checkpoint_path = train_cfg.get("pretrained_model_path", "checkpoints/dgx_pretrain/latest.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Base model checkpoint not found at {checkpoint_path}")
        
    if is_rank_zero():
        print(f"Loading base model weights from {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_config_dict = checkpoint["model_config"]
    model_config = ModelConfig(**model_config_dict)
    block_size = model_config.block_size
    
    model = DecoderOnlyTransformer(model_config)
    model.load_state_dict(checkpoint["model"])
    
    with open(train_cfg["sft_data_path"], "r") as f:
        sft_data = [json.loads(line) for line in f]
        
    # Simple split
    split_idx = int(len(sft_data) * 0.9)
    train_data = sft_data[:split_idx]
    val_data = sft_data[split_idx:]

    train_dataset = SFTDataset(train_data, tokenizer, block_size)
    val_dataset = SFTDataset(val_data, tokenizer, block_size)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if distributed else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["micro_batch_size"],
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=train_cfg["num_workers"],
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["micro_batch_size"],
        sampler=val_sampler,
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        pin_memory=True,
    )

    if is_rank_zero():
        print("Model config:", model_config)
        print(f"Number of parameters: {model.get_num_params():,}")

    start_step = 0

    resume_from = train_cfg.get("resume_from")

    if resume_from:
        ckpt = torch.load(resume_from, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        start_step = ckpt["step"] + 1

        if is_rank_zero():
            print(f"Resumed from {resume_from} at step {start_step}")

    model.to(device)

    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
        )

    optimizer = configure_optimizer(model, train_cfg)

    if resume_from:
        optimizer.load_state_dict(ckpt["optimizer"])

    precision = train_cfg["precision"]
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    use_amp = device.type == "cuda"

    scaler = torch.cuda.amp.GradScaler(enabled=(precision == "fp16" and use_amp))

    model.train()

    train_iter = iter(train_loader)

    progress = tqdm(
        range(start_step, train_cfg["max_steps"]),
        disable=not is_rank_zero(),
        desc="Pretraining",
    )

    for step in progress:
        if train_sampler is not None:
            train_sampler.set_epoch(step)

        lr = get_lr(step, train_cfg)

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0

        for micro_step in range(train_cfg["grad_accum_steps"]):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            sync_context = (
                model.no_sync()
                if distributed and micro_step < train_cfg["grad_accum_steps"] - 1
                else nullcontext()
            )

            with sync_context:
                with torch.amp.autocast(
                    device_type=device.type if use_amp else "cpu",
                    dtype=amp_dtype if use_amp else torch.float32,
                    enabled=use_amp,
                ):
                    logits, _ = model(x)
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = y[..., 1:].contiguous()
                    loss = torch.nn.functional.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)), 
                        shift_labels.view(-1), 
                        ignore_index=-100
                    )
                    loss = loss / train_cfg["grad_accum_steps"]

                if use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                    
                total_loss += loss.detach().item()

        if train_cfg["grad_clip"] > 0:
            if use_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                train_cfg["grad_clip"],
            )

        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        if step % train_cfg["log_interval"] == 0:
            train_loss = total_loss
            ppl = math.exp(min(train_loss, 20))

            if is_rank_zero():
                progress.set_postfix(
                    {
                        "loss": f"{train_loss:.4f}",
                        "ppl": f"{ppl:.2f}",
                        "lr": f"{lr:.2e}",
                    }
                )

            append_log(
                out_dir,
                {
                    "step": step,
                    "train_loss": train_loss,
                    "train_ppl": ppl,
                    "lr": lr,
                },
            )

        if step % train_cfg["eval_interval"] == 0 and step > 0:
            val_loss = estimate_loss(
                model=model,
                val_loader=val_loader,
                device=device,
                eval_iters=train_cfg["eval_iters"],
                precision=precision,
                distributed=distributed,
            )

            val_ppl = math.exp(min(val_loss, 20))

            if is_rank_zero():
                print(f"\nStep {step}: val_loss={val_loss:.4f}, val_ppl={val_ppl:.2f}")

                try:
                    raw_model = model.module if hasattr(model, "module") else model
                    raw_model.eval()
                    sample_text_prompt = "User: Tell me a story about a brave dog.\nAssistant: "
                    context = torch.tensor(tokenizer.encode(sample_text_prompt).ids, dtype=torch.long, device=device).unsqueeze(0)
                    generated = raw_model.generate(context, max_new_tokens=100, temperature=0.8, top_k=50)
                    gen_text = tokenizer.decode(generated[0].tolist())
                    print(f"\nSample:\n {gen_text}\n")
                    raw_model.train()
                except Exception as e:
                    print(f"Sampling skipped due to error: {e}")

            append_log(
                out_dir,
                {
                    "step": step,
                    "val_loss": val_loss,
                    "val_ppl": val_ppl,
                },
            )

        if step % train_cfg["save_interval"] == 0 and step > 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                step=step,
                cfg=cfg,
                out_dir=out_dir,
            )

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        step=train_cfg["max_steps"],
        cfg=cfg,
        out_dir=out_dir,
    )

    cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
