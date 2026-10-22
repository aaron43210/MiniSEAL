"""
Torch-free evaluation metrics shared across services.

Kept dependency-light (pure Python) so the reward used by the SEAL accept/reject
gate can be unit-tested without importing torch.
"""


def token_f1(prediction: str, reference: str) -> float:
    """Token-level F1 between prediction and reference (SQuAD-style, order-insensitive)."""
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    if not pred_tokens or not ref_tokens:
        return 1.0 if pred_tokens == ref_tokens else 0.0

    common = {}
    for t in ref_tokens:
        common[t] = common.get(t, 0) + 1
    overlap = 0
    for t in pred_tokens:
        if common.get(t, 0) > 0:
            overlap += 1
            common[t] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)
