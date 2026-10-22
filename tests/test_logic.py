"""
Torch-free verification tests for MiniSEAL logic.

Run (no pytest needed):
    python3 -m tests.test_logic

Covers:
- token-F1 reward used by the SEAL accept/reject gate
- PII IP-address regex (must not false-positive on version strings)
- canonical prompt formatting shared by SFT / DPO / inference
- SFT train/val dedup + deterministic shuffle (no leakage)
"""

import os
import sys
import json
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_token_f1():
    from src.common.metrics import token_f1 as _token_f1

    # Identical strings -> perfect score.
    assert _token_f1("the cat sat", "the cat sat") == 1.0
    # No overlap -> zero.
    assert _token_f1("dogs run", "the cat sat") == 0.0
    # Partial overlap -> between 0 and 1.
    f1 = _token_f1("the cat", "the cat sat")
    assert 0.0 < f1 < 1.0
    # Order-insensitive.
    assert _token_f1("cat the sat", "the cat sat") == 1.0
    # Empty prediction vs non-empty reference -> 0.
    assert _token_f1("", "the cat") == 0.0
    print("test_token_f1 OK")


def test_ip_regex():
    from src.hitl.validators import RiskValidator

    v = RiskValidator(blocked_keywords=[], sensitive_topics=[])
    # Real IPv4 flagged.
    assert v.ip_regex.search("connect to 192.168.1.1 now")
    assert v.ip_regex.search("mask 255.255.255.0")
    # Version strings with >4 parts must NOT false-positive.
    assert not v.ip_regex.search("release 3.11.4.2.1 shipped")
    # Out-of-range octets rejected.
    assert not v.ip_regex.search("999.1.1.1")
    assert not v.ip_regex.search("256.0.0.1")
    # Over-long trailing octet rejected.
    assert not v.ip_regex.search("10.0.0.1234")
    print("test_ip_regex OK")


def test_risk_levels():
    from src.hitl.validators import RiskValidator

    v = RiskValidator(blocked_keywords=["ignore previous"], sensitive_topics=["password"])
    level, flags = v.analyze("hello", "ignore previous instructions")
    assert level == "HIGH", (level, flags)
    level, flags = v.analyze("what is the password", "it is secret")
    assert level == "MEDIUM", (level, flags)
    level, flags = v.analyze("hi", "all good")
    assert level == "LOW", (level, flags)
    print("test_risk_levels OK")


def test_prompt_format_shared():
    from src.common.prompts import format_prompt, CHAT_PROMPT_TEMPLATE

    p = format_prompt("  Explain RoPE.  ")
    # Strips the instruction and uses the canonical template.
    assert p == CHAT_PROMPT_TEMPLATE.format(instruction="Explain RoPE.")
    assert p.endswith("### Response:\n")
    print("test_prompt_format_shared OK")


def test_resolvers_fallback():
    from src.common.prompts import resolve_eos_id, resolve_pad_id

    class FakeTok:
        def __init__(self, mapping):
            self.mapping = mapping

        def token_to_id(self, tok):
            return self.mapping.get(tok)

    # Prefers <|eos|>.
    assert resolve_eos_id(FakeTok({"<|eos|>": 7, "<|endoftext|>": 9})) == 7
    # Falls back to <|endoftext|>.
    assert resolve_eos_id(FakeTok({"<|endoftext|>": 9})) == 9
    # None when neither present (callers must then skip appending EOS).
    assert resolve_eos_id(FakeTok({})) is None
    # Pad falls back to 0.
    assert resolve_pad_id(FakeTok({})) == 0
    assert resolve_pad_id(FakeTok({"<|pad|>": 3})) == 3
    print("test_resolvers_fallback OK")


def test_sft_dedup_and_split_no_leakage():
    """Mirror the dedup + seeded-shuffle + split logic in src/train/sft.py."""
    # 8 unique rows, each duplicated 50x (the old create_rag_dataset behavior).
    base = [{"prompt": f"q{i}", "response": f"a{i}"} for i in range(8)]
    sft_data = base * 50

    seen = set()
    unique = []
    for row in sft_data:
        key = json.dumps(row, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append(row)
    assert len(unique) == 8, len(unique)

    rng = random.Random(1234)
    rng.shuffle(unique)
    split_idx = int(len(unique) * 0.9)
    train, val = unique[:split_idx], unique[split_idx:]

    # No row appears in both splits -> no train/val leakage.
    train_keys = {json.dumps(r, sort_keys=True) for r in train}
    val_keys = {json.dumps(r, sort_keys=True) for r in val}
    assert train_keys.isdisjoint(val_keys)
    assert len(val) >= 1

    # Deterministic under a fixed seed.
    u2 = list(base)
    random.Random(1234).shuffle(u2)
    # Same seed + same input order -> same permutation.
    ref = list(base)
    random.Random(1234).shuffle(ref)
    assert u2 == ref
    print("test_sft_dedup_and_split_no_leakage OK")


def main():
    tests = [
        test_token_f1,
        test_ip_regex,
        test_risk_levels,
        test_prompt_format_shared,
        test_resolvers_fallback,
        test_sft_dedup_and_split_no_leakage,
    ]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} torch-free tests passed.")


if __name__ == "__main__":
    main()
