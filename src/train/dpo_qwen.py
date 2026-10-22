# src/train/dpo_qwen.py
#
# DPO (Direct Preference Optimization) on HF Qwen + PEFT LoRA -- the Qwen-track
# counterpart to src/train/dpo.py (which is the custom from-scratch model).
#
# Policy  = Qwen + a trainable LoRA adapter.
# Reference = the SAME model with the adapter DISABLED (peft `disable_adapter()`),
#             so we need only ONE model in memory for the DPO loss.
# Loss (per preference pair): -log sigmoid( beta * [ (logp_pol(chosen)-logp_ref(chosen))
#                                                  - (logp_pol(reject)-logp_ref(reject)) ] )
#
# Run:  python3 -m src.train.dpo_qwen --config configs/dpo_qwen.yaml

import argparse
import os
import sys
import json
import time

import yaml
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# Add project root to python path (mirrors src/train/sft.py lines 11-13)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.common.utils import ensure_dir, set_seed

LORA_TARGET_MODULES = ["q_proj", "v_proj"]


def _load_pairs(data_path, max_pairs=None):
    """Read preference rows {prompt, chosen, rejected} from a JSONL file."""
    pairs = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            p, c, r = row.get("prompt"), row.get("chosen"), row.get("rejected")
            if p and c and r:
                pairs.append((p, c, r))
    if max_pairs:
        pairs = pairs[:max_pairs]
    return pairs


def _encode_pair(tokenizer, prompt, answer, device, max_length):
    """(prompt, answer) -> (input_ids, labels) with the prompt tokens masked (-100).

    Returns None if the full sequence exceeds max_length (truncation would corrupt
    the preference signal, so we skip over-long pairs instead).
    """
    user = {"role": "user", "content": prompt}
    assistant = {"role": "assistant", "content": answer}
    full = tokenizer.apply_chat_template([user, assistant], add_generation_prompt=False)
    pre = tokenizer.apply_chat_template([user], add_generation_prompt=True)
    if len(pre) >= len(full) or len(full) > max_length:
        return None
    labels = [-100] * len(pre) + full[len(pre):]
    x = torch.tensor([full], dtype=torch.long, device=device)
    y = torch.tensor([labels], dtype=torch.long, device=device)
    return x, y


def _seq_logp(model, x, y):
    """Summed log-prob of the non-masked (answer) tokens: scalar tensor per sequence."""
    logits = model(input_ids=x).logits[:, :-1, :]
    labels = y[:, 1:]
    mask = labels != -100
    logp = torch.log_softmax(logits, dim=-1)
    safe = labels.masked_fill(~mask, 0)
    tok_logp = logp.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    return (tok_logp * mask).sum(dim=-1)


def main():
    parser = argparse.ArgumentParser(description="DPO preference alignment on HF Qwen + LoRA.")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))
    out_dir = cfg["out_dir"]
    ensure_dir(out_dir)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    base_model_path = cfg.get("pretrained_model_path", "Qwen/Qwen2.5-1.5B-Instruct")
    data_path = cfg.get("dpo_data_path", "data/alignment/preferences.jsonl")
    beta = float(cfg.get("beta", 0.1))
    epochs = int(cfg.get("epochs", 1))
    lr = float(cfg.get("learning_rate", 5e-6))
    max_length = int(cfg.get("max_length", 1024))
    max_pairs = cfg.get("max_pairs", None)
    sft_adapter_path = cfg.get("sft_adapter_path", None)

    print(f"[DPO] Loading base model: {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model = model.to(device)

    # Optional: start DPO from a prior SFT adapter (merge it into the base first,
    # so the reference = the SFT model and the new LoRA learns the preference delta).
    if sft_adapter_path and os.path.isdir(sft_adapter_path):
        from peft import PeftModel
        print(f"[DPO] Merging SFT adapter from {sft_adapter_path}")
        model = PeftModel.from_pretrained(model, sft_adapter_path)
        model = model.merge_and_unload()

    lora_config = LoraConfig(
        r=int(cfg.get("lora_r", 16)),
        lora_alpha=int(cfg.get("lora_alpha", 32)),
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    pairs = _load_pairs(data_path, max_pairs)
    samples, skipped = [], 0
    for p, c, r in pairs:
        ec = _encode_pair(tokenizer, p, c, device, max_length)
        er = _encode_pair(tokenizer, p, r, device, max_length)
        if ec is None or er is None:
            skipped += 1
            continue
        samples.append((ec, er))
    print(f"[DPO] {len(samples)} usable preference pair(s) (skipped {skipped} over-long/empty) from {data_path}")
    if not samples:
        print("[DPO] No usable pairs. Exiting.")
        return

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    model.train()
    for epoch in range(epochs):
        running_loss, running_acc = 0.0, 0.0
        for (xc, yc), (xr, yr) in samples:
            optimizer.zero_grad()
            # Policy (adapter ON) log-probs, with gradient.
            pol_ch = _seq_logp(model, xc, yc)
            pol_rej = _seq_logp(model, xr, yr)
            # Reference (adapter OFF) log-probs, frozen -- same weights, no adapter.
            with torch.no_grad(), model.disable_adapter():
                ref_ch = _seq_logp(model, xc, yc)
                ref_rej = _seq_logp(model, xr, yr)
            logits = beta * ((pol_ch - ref_ch) - (pol_rej - ref_rej))
            loss = -F.logsigmoid(logits).mean()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_acc += (logits > 0).float().mean().item()  # chosen preferred
        n = len(samples)
        print(f"[DPO] Epoch {epoch + 1}/{epochs} | loss {running_loss / n:.4f} | "
              f"pref-acc {running_acc / n:.3f}")

    version_id = int(time.time())
    adapter_path = os.path.join(out_dir, f"dpo_adapter_v{version_id}")
    ensure_dir(adapter_path)
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    with open(os.path.join(adapter_path, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "source": "dpo_qwen", "base_model": base_model_path,
            "sft_adapter": sft_adapter_path, "beta": beta, "epochs": epochs, "lr": lr,
            "num_pairs": len(samples), "skipped": skipped, "version_id": version_id,
        }, f, indent=2)
    print(f"[DPO] Saved DPO LoRA adapter to {adapter_path}")


if __name__ == "__main__":
    main()
