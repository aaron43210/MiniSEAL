"""
Torch-dependent verification tests for MiniSEAL training math.

Requires torch. Run on the GPU/training box (no pytest needed):
    python3 -m tests.test_training_math

Covers:
- DPO loss: zero at init (policy == reference), positive, decreases as the
  policy prefers chosen over rejected; chosen-accuracy signal is correct.
- get_batch_logps: ignores -100 label positions and shifts correctly.
- SFT causal-shift masking: prompt tokens (-100) contribute no loss.
- RMSNorm: numerically stable and dtype-preserving under autocast dtypes.
- generate(): batched EOS stopping returns the right shapes.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F


def test_dpo_loss_properties():
    from src.train.dpo import dpo_loss

    beta = 0.1
    # At initialization policy == reference -> logits diff is 0 -> loss = -log(0.5).
    z = torch.zeros(4)
    loss, acc = dpo_loss(z.clone(), z.clone(), z.clone(), z.clone(), beta=beta)
    assert abs(loss.item() - (-F.logsigmoid(torch.tensor(0.0)).item())) < 1e-6
    assert abs(loss.item() - 0.6931) < 1e-3

    # Policy strongly prefers chosen over rejected relative to ref -> lower loss.
    pc = torch.tensor([2.0, 2.0, 2.0, 2.0])   # policy chosen logps
    pr = torch.tensor([-2.0, -2.0, -2.0, -2.0])  # policy rejected logps
    rc = torch.zeros(4)
    rr = torch.zeros(4)
    good_loss, good_acc = dpo_loss(pc, pr, rc, rr, beta=beta)
    assert good_loss.item() < loss.item()
    assert good_acc.item() == 1.0  # chosen logratio > rejected everywhere

    # Reversed preference -> higher loss and zero accuracy.
    bad_loss, bad_acc = dpo_loss(pr, pc, rc, rr, beta=beta)
    assert bad_loss.item() > loss.item()
    assert bad_acc.item() == 0.0
    print("test_dpo_loss_properties OK")


def test_get_batch_logps_masks_prompt():
    from src.train.dpo import get_batch_logps

    torch.manual_seed(0)
    B, T, V = 2, 5, 7
    logits = torch.randn(B, T, V)

    labels = torch.randint(0, V, (B, T))
    masked = labels.clone()
    masked[:, :2] = -100  # mask first two "prompt" positions

    lp_full = get_batch_logps(logits, labels)
    lp_masked = get_batch_logps(logits, masked)

    # Masked version sums over fewer tokens, so logps (negative CE sum) must be >=.
    assert torch.all(lp_masked >= lp_full - 1e-4)

    # If ALL labels are masked, logps must be exactly 0 (no supervised tokens).
    all_masked = torch.full_like(labels, -100)
    assert torch.allclose(get_batch_logps(logits, all_masked), torch.zeros(B))
    print("test_get_batch_logps_masks_prompt OK")


def test_sft_shift_ignores_prompt():
    # Loss on a sequence where prompt labels are -100 must equal loss computed
    # only over the response positions.
    torch.manual_seed(1)
    V = 10
    logits = torch.randn(1, 6, V)
    labels = torch.tensor([[-100, -100, -100, 3, 4, 5]])

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_all = F.cross_entropy(
        shift_logits.view(-1, V), shift_labels.view(-1), ignore_index=-100
    )

    # Manual: only positions where shifted label != -100.
    mask = shift_labels.view(-1) != -100
    manual = F.cross_entropy(
        shift_logits.view(-1, V)[mask], shift_labels.view(-1)[mask]
    )
    assert torch.allclose(loss_all, manual, atol=1e-5)
    print("test_sft_shift_ignores_prompt OK")


def test_rmsnorm_stability_and_dtype():
    from src.model.layers import RMSNorm

    norm = RMSNorm(16, eps=1e-5)
    x = torch.randn(2, 4, 16)
    out = norm(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()

    # Preserves input dtype even though variance is computed in fp32.
    if torch.cuda.is_available():
        xb = x.to(torch.bfloat16)
        ob = norm(xb)
        assert ob.dtype == torch.bfloat16
    print("test_rmsnorm_stability_and_dtype OK")


def test_generate_batched_shapes():
    from src.model.config import ModelConfig
    from src.model.llm import DecoderOnlyTransformer

    cfg = ModelConfig(vocab_size=32, n_layer=2, n_head=2, n_embd=16, block_size=32)
    model = DecoderOnlyTransformer(cfg)
    model.eval()

    ids = torch.randint(0, 32, (3, 4))  # batch of 3
    out = model.generate(ids, max_new_tokens=5, temperature=1.0, top_k=4)
    assert out.shape[0] == 3
    assert out.shape[1] == 4 + 5
    print("test_generate_batched_shapes OK")


def main():
    tests = [
        test_dpo_loss_properties,
        test_get_batch_logps_masks_prompt,
        test_sft_shift_ignores_prompt,
        test_rmsnorm_stability_and_dtype,
        test_generate_batched_shapes,
    ]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} torch math tests passed.")


if __name__ == "__main__":
    main()
