from typing import Union
import json
import os
import random
from pathlib import Path

import numpy as np
import torch


def ensure_dir(path: Union[str, Path]) -> None:
    """
    Input:
        path: Directory path.
    Output:
        Creates the directory if it does not exist.
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(obj: dict, path: Union[str, Path]) -> None:
    """
    Input:
        obj: Python dictionary.
        path: Output JSON path.
    Output:
        Saves dictionary as formatted JSON.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: Union[str, Path]) -> dict:
    """
    Input:
        path: JSON file path.
    Output:
        Loaded Python dictionary.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed: int) -> None:
    """
    Input:
        seed: Integer seed.
    Output:
        Makes Python, NumPy, and PyTorch randomness more reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_rank_zero() -> bool:
    """
    Output:
        True only for the main distributed process.
    """
    return int(os.environ.get("RANK", "0")) == 0


def get_world_size() -> int:
    """
    Output:
        Number of distributed processes.
    """
    return int(os.environ.get("WORLD_SIZE", "1"))
