"""
Phase 1B: Encode raw corpus into binary token shards.

Input:
    Raw .txt files
    tokenizer.json
    tokenizer_config.json

Output:
    train.bin
    val.bin
    train.meta.json
    val.meta.json

Each line is treated as one document.
For a cleaner corpus, prepare one paragraph/document per line.
"""

import argparse
import random
from pathlib import Path
import os
import sys

# Add project root to python path so 'src' can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm

from src.common.utils import ensure_dir, load_json, save_json


def collect_text_files(input_dir: str) -> list[Path]:
    """
    Input:
        input_dir: Folder containing .txt files.
    Output:
        List of Path objects.
    """
    files = sorted(Path(input_dir).rglob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files found in {input_dir}")
    return files


def choose_dtype(vocab_size: int):
    """
    Input:
        vocab_size: Tokenizer vocabulary size.
    Output:
        uint16 if possible, otherwise uint32.
    """
    if vocab_size <= 65535:
        return np.uint16
    return np.uint32


def encode_and_write(
    input_dir: str,
    tokenizer_path: str,
    tokenizer_config_path: str,
    output_dir: str,
    val_fraction: float,
    seed: int,
    batch_size: int,
) -> None:
    """
    Input:
        input_dir: Raw corpus folder.
        tokenizer_path: Path to tokenizer.json.
        tokenizer_config_path: Path to tokenizer_config.json.
        output_dir: Destination for .bin files.
        val_fraction: Fraction of lines for validation.
        seed: Random split seed.
        batch_size: Number of lines encoded at once.

    Output:
        train.bin / val.bin and metadata.
    """
    ensure_dir(output_dir)

    tokenizer = Tokenizer.from_file(tokenizer_path)
    tok_cfg = load_json(tokenizer_config_path)

    vocab_size = tok_cfg["vocab_size"]
    eos_id = tok_cfg["special_token_ids"]["eos_token"]
    dtype = choose_dtype(vocab_size)

    train_path = Path(output_dir) / "train.bin"
    val_path = Path(output_dir) / "val.bin"

    # Remove old files to avoid appending to previous runs.
    train_path.unlink(missing_ok=True)
    val_path.unlink(missing_ok=True)

    rng = random.Random(seed)
    files = collect_text_files(input_dir)

    train_tokens = 0
    val_tokens = 0
    train_docs = 0
    val_docs = 0

    def flush_lines(lines: list[str]) -> None:
        nonlocal train_tokens, val_tokens, train_docs, val_docs

        if not lines:
            return

        encodings = tokenizer.encode_batch(lines)

        with open(train_path, "ab") as train_f, open(val_path, "ab") as val_f:
            for enc in encodings:
                ids = enc.ids + [eos_id]
                arr = np.asarray(ids, dtype=dtype)

                if rng.random() < val_fraction:
                    arr.tofile(val_f)
                    val_tokens += len(arr)
                    val_docs += 1
                else:
                    arr.tofile(train_f)
                    train_tokens += len(arr)
                    train_docs += 1

    buffer = []

    for file in tqdm(files, desc="Encoding files"):
        with open(file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                buffer.append(line)

                if len(buffer) >= batch_size:
                    flush_lines(buffer)
                    buffer = []

    flush_lines(buffer)

    train_meta = {
        "path": str(train_path),
        "dtype": np.dtype(dtype).name,
        "n_tokens": train_tokens,
        "n_docs": train_docs,
        "vocab_size": vocab_size,
    }

    val_meta = {
        "path": str(val_path),
        "dtype": np.dtype(dtype).name,
        "n_tokens": val_tokens,
        "n_docs": val_docs,
        "vocab_size": vocab_size,
    }

    save_json(train_meta, Path(output_dir) / "train.meta.json")
    save_json(val_meta, Path(output_dir) / "val.meta.json")

    print("Encoding complete.")
    print(train_meta)
    print(val_meta)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--tokenizer_config_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--val_fraction", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--batch_size", type=int, default=1024)
    args = parser.parse_args()

    encode_and_write(
        input_dir=args.input_dir,
        tokenizer_path=args.tokenizer_path,
        tokenizer_config_path=args.tokenizer_config_path,
        output_dir=args.output_dir,
        val_fraction=args.val_fraction,
        seed=args.seed,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
