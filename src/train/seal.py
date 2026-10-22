# src/train/seal.py
#
# SEAL (Self-Adapting LLM) trainer -- HuggingFace Qwen + PEFT LoRA edition.
#
# Mirrors the reward-gated self-edit logic in src/adaptation/adaptation_service.py
# but as a standalone CLI on a real HF model (no FastAPI, no from-scratch model),
# with a HELD-OUT evaluation so the reward measures generalization, not memory:
#   1. split each passage's Q&A into an ADAPT set and a held-out EVAL set,
#   2. measure BASELINE token-F1 on the held-out set (before any training),
#   3. inject a LoRA adapter and train it ONLY on the adapt-set self-edits,
#   4. measure the ADAPTED token-F1 on the held-out set,
#   5. ACCEPT the adapter only if the HELD-OUT reward improved by >= accept_margin.
# A safety gate screens every self-edit first and drops HIGH-risk pairs.
#
# Run:  python3 -m src.train.seal --config configs/seal_qwen.yaml

import argparse
import os
import sys
import json
import random
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# Add project root to python path (mirrors src/train/sft.py lines 11-13)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.common.utils import ensure_dir, set_seed
from src.common.metrics import token_f1
from src.hitl.validators import RiskValidator

LORA_TARGET_MODULES = ["q_proj", "v_proj"]
ADAPT_PROMPT = "Passage Context: {passage}\nQuestion: {q}"


def _load_grouped(data_path, max_passages=None):
    """Load squad-style rows as per-passage groups: [{passage, qa:[{q,a}]}].

    Grouping is required because the held-out split happens WITHIN each passage:
    we adapt on some of a passage's Q&A and score on different, unseen Q&A about
    the SAME passage -- that is what makes the reward a generalization signal
    rather than a memorization one.
    """
    groups = []
    with open(data_path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if max_passages:
        rows = rows[:max_passages]
    for row in rows:
        passage = row.get("passage", "")
        qa = [{"q": p.get("q"), "a": p.get("a")}
              for p in row.get("qa_pairs", []) if p.get("q") and p.get("a")]
        if qa:
            groups.append({"passage": passage, "qa": qa})
    return groups


def _split_adapt_heldout(groups, holdout_frac, min_qa, seed):
    """Per-passage split into (adapt_edits, heldout_edits, skipped).

    For each passage we shuffle its Q&A with a FIXED seed (reproducible) and peel
    off `holdout_frac` of them (at least 1) as the held-out eval set; the rest are
    the self-edits we adapt on. A passage with fewer than `min_qa` pairs cannot be
    split (we need >=1 to train AND >=1 to test), so it is skipped and counted.
    """
    rng = random.Random(seed)
    adapt, heldout, skipped = [], [], 0
    for g in groups:
        qa = list(g["qa"])
        if len(qa) < min_qa:
            skipped += 1
            continue
        rng.shuffle(qa)
        n_hold = max(1, round(len(qa) * holdout_frac))
        n_hold = min(n_hold, len(qa) - 1)  # always leave >=1 pair to adapt on
        hold, keep = qa[:n_hold], qa[n_hold:]
        adapt.extend({"passage": g["passage"], "q": p["q"], "a": p["a"]} for p in keep)
        heldout.extend({"passage": g["passage"], "q": p["q"], "a": p["a"]} for p in hold)
    return adapt, heldout, skipped


def _apply_safety_gate(edits):
    """Drop HIGH-risk self-edits before spending any compute (SEAL safety gate)."""
    validator = RiskValidator(
        blocked_keywords=["ignore previous instructions", "system prompt"],
        sensitive_topics=["hack", "illegal", "exploit"],
    )
    safe = []
    for e in edits:
        level, flags = validator.analyze(e["q"], e["a"])
        if level == "HIGH":
            print(f"[SEAL] Safety gate DROPPED a HIGH-risk self-edit: {flags} | Q={e['q'][:60]!r}")
            continue
        safe.append(e)
    return safe


@torch.no_grad()
def _greedy_answer(model, tokenizer, passage, question, device, max_new_tokens):
    """Deterministic (greedy) answer, so baseline vs adapted rewards are comparable."""
    messages = [{"role": "user", "content": ADAPT_PROMPT.format(passage=passage, q=question)}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(device)
    out = model.generate(
        input_ids=inputs,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    )
    gen_ids = out[0, inputs.shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def _evaluate(model, tokenizer, edits, device, max_new_tokens):
    """Mean token-F1 of greedy answers over the self-edit QA set."""
    if not edits:
        return 0.0
    model.eval()
    total = 0.0
    for e in edits:
        pred = _greedy_answer(model, tokenizer, e["passage"], e["q"], device, max_new_tokens)
        total += token_f1(pred, e["a"])
    return total / len(edits)


def _build_sample(tokenizer, passage, question, answer, device):
    """Tokenize one self-edit; mask the prompt so loss falls only on the answer."""
    user = {"role": "user", "content": ADAPT_PROMPT.format(passage=passage, q=question)}
    assistant = {"role": "assistant", "content": answer}
    full_ids = tokenizer.apply_chat_template([user, assistant], add_generation_prompt=False)
    prompt_ids = tokenizer.apply_chat_template([user], add_generation_prompt=True)
    if len(prompt_ids) >= len(full_ids):
        return None
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    x = torch.tensor([full_ids], dtype=torch.long, device=device)
    y = torch.tensor([labels], dtype=torch.long, device=device)
    return x, y


def main():
    parser = argparse.ArgumentParser(description="SEAL self-adapting LoRA trainer (HF Qwen, held-out eval).")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))
    out_dir = cfg["out_dir"]
    ensure_dir(out_dir)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    base_model_path = cfg.get("pretrained_model_path", "Qwen/Qwen2.5-1.5B-Instruct")
    data_path = cfg.get("self_adapt_data_path", "data/self_adapt/squad_mini.jsonl")
    epochs = int(cfg.get("epochs", 5))
    lr = float(cfg.get("learning_rate", 1e-4))
    accept_margin = float(cfg.get("accept_margin", 0.0))
    enforce_safety = bool(cfg.get("enforce_safety", True))
    max_new_tokens = int(cfg.get("max_new_tokens", 32))
    max_passages = cfg.get("max_passages", None)
    holdout_frac = float(cfg.get("holdout_frac", 0.34))
    min_qa = int(cfg.get("min_qa_per_passage", 2))
    split_seed = int(cfg.get("split_seed", 42))
    results_path = cfg.get("results_path", "runs/self_adapt/results.json")

    # --- Load + split per passage: adapt on some Q&A, score on held-out Q&A ---
    groups = _load_grouped(data_path, max_passages)
    adapt_edits, heldout_edits, skipped = _split_adapt_heldout(groups, holdout_frac, min_qa, split_seed)
    print(f"[SEAL] {len(groups)} passage(s) | adapt={len(adapt_edits)} "
          f"held-out={len(heldout_edits)} | skipped {skipped} passage(s) with < {min_qa} QA")

    if enforce_safety:
        adapt_edits = _apply_safety_gate(adapt_edits)
        heldout_edits = _apply_safety_gate(heldout_edits)
    if not adapt_edits or not heldout_edits:
        print("[SEAL] Not enough data after split/safety-gate "
              "(need >=1 adapt and >=1 held-out pair). Exiting.")
        return

    print(f"[SEAL] Loading base model: {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model = model.to(device)

    # 1) BASELINE rewards BEFORE adaptation: on held-out (the real claim) and on
    #    the adapt set (sanity -- it should rise MORE than held-out = memorization).
    base_heldout = _evaluate(model, tokenizer, heldout_edits, device, max_new_tokens)
    base_train = _evaluate(model, tokenizer, adapt_edits, device, max_new_tokens)
    print(f"[SEAL] Base reward (token-F1) | held-out: {base_heldout:.4f} | adapt-set: {base_train:.4f}")

    # 2) Inject LoRA and train it ONLY on the adapt-set self-edits.
    lora_config = LoraConfig(
        r=int(cfg.get("lora_r", 16)),
        lora_alpha=int(cfg.get("lora_alpha", 32)),
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=float(cfg.get("lora_dropout", 0.0)),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    samples = []
    for e in adapt_edits:
        s = _build_sample(tokenizer, e["passage"], e["q"], e["a"], device)
        if s is not None:
            samples.append(s)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    model.train()
    print(f"[SEAL] Adapting LoRA weights on {len(samples)} adapt-set self-edits...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        for x, y in samples:
            optimizer.zero_grad()
            outputs = model(input_ids=x, labels=y)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"[SEAL] Adaptation Epoch {epoch + 1}/{epochs} | "
              f"Loss: {epoch_loss / max(len(samples), 1):.4f}")

    # 3) ADAPTED rewards, then 4) accept/reject gate -- gated on HELD-OUT.
    adapted_heldout = _evaluate(model, tokenizer, heldout_edits, device, max_new_tokens)
    adapted_train = _evaluate(model, tokenizer, adapt_edits, device, max_new_tokens)
    improvement_heldout = adapted_heldout - base_heldout
    improvement_train = adapted_train - base_train
    accepted = improvement_heldout >= accept_margin
    print(f"[SEAL] Held-out reward: {base_heldout:.4f} -> {adapted_heldout:.4f} "
          f"(improvement {improvement_heldout:+.4f})  [HEADLINE]")
    print(f"[SEAL] Adapt-set reward: {base_train:.4f} -> {adapted_train:.4f} "
          f"(improvement {improvement_train:+.4f})  [sanity/memorization]")
    print(f"[SEAL] accept_margin={accept_margin} -> accepted={accepted}")

    # Record every run (accepted or not) so the report always has the numbers.
    results = {
        "base_model": base_model_path,
        "data_path": data_path,
        "n_passages": len(groups),
        "n_adapt": len(adapt_edits),
        "n_heldout": len(heldout_edits),
        "skipped_passages": skipped,
        "holdout_frac": holdout_frac,
        "split_seed": split_seed,
        "epochs": epochs,
        "lr": lr,
        "base_heldout": round(base_heldout, 4),
        "adapted_heldout": round(adapted_heldout, 4),
        "improvement_heldout": round(improvement_heldout, 4),
        "base_train": round(base_train, 4),
        "adapted_train": round(adapted_train, 4),
        "improvement_train": round(improvement_train, 4),
        "accept_margin": accept_margin,
        "accepted": accepted,
    }
    ensure_dir(os.path.dirname(results_path) or ".")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[SEAL] Wrote results to {results_path}")

    if accepted:
        # Persist ONLY the low-rank adapter deltas, plus the full run provenance.
        adapter_path = os.path.join(out_dir, "seal_adapter")
        ensure_dir(adapter_path)
        model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)
        with open(os.path.join(adapter_path, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({"source": "seal_self_adapt", **results}, f, indent=2)
        print(f"[SEAL] Self-edit accepted. Adapter saved to {adapter_path}")
    else:
        print("[SEAL] Self-edit rejected (held-out improvement < accept_margin). Adapter discarded.")


if __name__ == "__main__":
    main()
