"""
Phase 1A: Train custom BPE tokenizer.

Research grounding:
- BPE is widely used in modern LMs.
- The goal is to learn subword units from your own corpus.
- This satisfies the "self-owned tokenizer" requirement from the PRD.

Input:
    Folder containing .txt files.

Output:
    tokenizer.json
    tokenizer_config.json
"""

import argparse
from pathlib import Path
import os
import sys

# Add project root to python path so 'src' can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC, Sequence
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer

from src.common.utils import ensure_dir, save_json


SPECIAL_TOKENS = {
    "pad_token": "<|pad|>",
    "bos_token": "<|bos|>",
    "eos_token": "<|eos|>",
    "unk_token": "<|unk|>",
}


def collect_text_files(input_dir: str) -> list[str]:
    """
    Input:
        input_dir: Folder path containing .txt files.
    Output:
        List of text file paths.
    """
    paths = sorted(str(p) for p in Path(input_dir).rglob("*.txt"))
    if not paths:
        raise FileNotFoundError(f"No .txt files found in {input_dir}")
    return paths


def train_bpe_tokenizer(
    input_dir: str,
    output_dir: str,
    vocab_size: int,
    min_frequency: int,
) -> None:
    """
    Input:
        input_dir: Folder of raw .txt files.
        output_dir: Folder where tokenizer files will be saved.
        vocab_size: Target vocabulary size.
        min_frequency: Minimum token frequency for BPE merges.

    Output:
        tokenizer.json
        tokenizer_config.json
    """
    ensure_dir(output_dir)
    files = collect_text_files(input_dir)

    tokenizer = Tokenizer(
        BPE(
            unk_token=SPECIAL_TOKENS["unk_token"],
            fuse_unk=True,
        )
    )

    # NFKC normalizes Unicode text consistently.
    tokenizer.normalizer = Sequence([NFKC()])

    # ByteLevel makes the tokenizer robust to unknown characters.
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=list(SPECIAL_TOKENS.values()),
        show_progress=True,
    )

    tokenizer.train(files=files, trainer=trainer)

    tokenizer_path = Path(output_dir) / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    config = {
        "tokenizer_type": "ByteLevel-BPE",
        "vocab_size": tokenizer.get_vocab_size(),
        "special_tokens": SPECIAL_TOKENS,
        "special_token_ids": {
            name: tokenizer.token_to_id(token)
            for name, token in SPECIAL_TOKENS.items()
        },
    }

    save_json(config, Path(output_dir) / "tokenizer_config.json")

    print(f"Saved tokenizer to: {tokenizer_path}")
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--min_frequency", type=int, default=2)
    args = parser.parse_args()

    train_bpe_tokenizer(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )


if __name__ == "__main__":
    main()
