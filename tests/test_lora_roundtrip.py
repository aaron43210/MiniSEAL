"""
Torch-dependent tests that the SEAL adapter loop actually closes:
a trained LoRA adapter can be saved and re-loaded onto a fresh base model,
reproducing the adapted model's outputs bit-for-bit.

Run on a torch box:
    python3 -m tests.test_lora_roundtrip

Covers:
- save_lora_adapter -> load_lora_adapter reproduces identical logits.
- A fresh base model (no adapter) differs from the adapted one (sanity: the
  adapter is actually doing something), and loading recovers the adapted output.
- load_lora_state_dict raises on a config mismatch (wrong rank).
- merge_lora_weights produces outputs equal to the unmerged LoRA forward.
- Legacy bare-state-dict adapters still load via the back-compat path.
"""

import os
import sys
import tempfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.model.config import ModelConfig
from src.model.llm import DecoderOnlyTransformer
from src.lora.lora import (
    inject_lora,
    save_lora_adapter,
    load_lora_adapter,
    load_lora_state_dict,
    get_lora_state_dict,
    merge_lora_weights,
)

TARGETS = ["q_proj", "v_proj"]
RANK, ALPHA = 8, 16


def _fresh_model(seed=0):
    torch.manual_seed(seed)
    cfg = ModelConfig(vocab_size=32, n_layer=2, n_head=2, n_embd=16, block_size=32)
    m = DecoderOnlyTransformer(cfg)
    m.eval()
    return m


def _logits(model, ids):
    with torch.no_grad():
        out = model(ids)
        return out[0] if isinstance(out, tuple) else out


def test_adapter_roundtrip_reproduces_outputs():
    ids = torch.randint(0, 32, (1, 6))

    # 1. Base model + injected LoRA, then perturb the adapter so it's non-trivial
    #    (B starts at zero -> no-op; fill it to simulate "trained" deltas).
    model = _fresh_model(seed=0)
    base_logits = _logits(model, ids).clone()

    inject_lora(model, target_keywords=TARGETS, rank=RANK, alpha=ALPHA, dropout=0.0)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.normal_(0, 0.1)  # pretend training moved B away from zero
    adapted_logits = _logits(model, ids).clone()

    # Adapter must actually change the output, else the test proves nothing.
    assert not torch.allclose(base_logits, adapted_logits, atol=1e-5)

    # 2. Save adapter, build a FRESH base, load adapter back.
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "adapter.pt")
        save_lora_adapter(model, path, rank=RANK, alpha=ALPHA,
                          target_keywords=TARGETS, dropout=0.0,
                          meta={"source": "test"})

        fresh = _fresh_model(seed=0)  # same init as the original base
        fresh = load_lora_adapter(fresh, path, device="cpu")
        reloaded_logits = _logits(fresh, ids)

    # 3. Reloaded == adapted (the loop closes).
    assert torch.allclose(adapted_logits, reloaded_logits, atol=1e-5), \
        (adapted_logits - reloaded_logits).abs().max()
    print("test_adapter_roundtrip_reproduces_outputs OK")


def test_load_rejects_config_mismatch():
    model = _fresh_model(seed=1)
    inject_lora(model, target_keywords=TARGETS, rank=RANK, alpha=ALPHA, dropout=0.0)
    sd = get_lora_state_dict(model)

    # Inject with a DIFFERENT rank -> shapes/keys won't line up -> must raise.
    other = _fresh_model(seed=1)
    inject_lora(other, target_keywords=TARGETS, rank=RANK * 2, alpha=ALPHA, dropout=0.0)
    try:
        load_lora_state_dict(other, sd, strict=True)
    except (RuntimeError, KeyError):
        print("test_load_rejects_config_mismatch OK")
        return
    raise AssertionError("expected a mismatch error on wrong rank")


def test_merge_matches_unmerged():
    ids = torch.randint(0, 32, (1, 5))
    model = _fresh_model(seed=2)
    inject_lora(model, target_keywords=TARGETS, rank=RANK, alpha=ALPHA, dropout=0.0)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.normal_(0, 0.1)
    unmerged = _logits(model, ids).clone()

    merge_lora_weights(model)
    merged = _logits(model, ids)
    assert torch.allclose(unmerged, merged, atol=1e-5), (unmerged - merged).abs().max()
    print("test_merge_matches_unmerged OK")


def test_legacy_bare_state_dict_loads():
    ids = torch.randint(0, 32, (1, 4))
    model = _fresh_model(seed=3)
    inject_lora(model, target_keywords=TARGETS, rank=16, alpha=32, dropout=0.0)  # legacy defaults
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.normal_(0, 0.1)
    adapted = _logits(model, ids).clone()

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "legacy.pt")
        torch.save(get_lora_state_dict(model), path)  # bare dict, no wrapper/config

        fresh = _fresh_model(seed=3)
        fresh = load_lora_adapter(fresh, path, device="cpu")  # falls back to defaults
        reloaded = _logits(fresh, ids)
    assert torch.allclose(adapted, reloaded, atol=1e-5)
    print("test_legacy_bare_state_dict_loads OK")


def main():
    tests = [
        test_adapter_roundtrip_reproduces_outputs,
        test_load_rejects_config_mismatch,
        test_merge_matches_unmerged,
        test_legacy_bare_state_dict_loads,
    ]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} LoRA round-trip tests passed.")


if __name__ == "__main__":
    main()
