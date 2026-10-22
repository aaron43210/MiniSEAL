from typing import Optional
"""
Alignment datasets and formatting utilities.

Used for:
1. Supervised Fine-Tuning, SFT
2. Direct Preference Optimization, DPO

Input formats:

SFT JSONL:
    {"instruction": "...", "response": "..."}

DPO JSONL:
    {"prompt": "...", "chosen": "...", "rejected": "..."}

Output:
    PyTorch datasets returning tokenized tensors.
"""

import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer


def read_jsonl(path: str) -> list[dict]:
    """
    Input:
        path: JSONL file path.

    Output:
        List of dictionaries.
    """
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            rows.append(json.loads(line))

    return rows


def format_instruction_prompt(instruction: str) -> str:
    """
    Input:
        instruction: User instruction text.

    Output:
        Prompt string used during SFT/alignment. Delegates to the single shared
        formatter so SFT, DPO, and inference never drift apart.
    """
    from src.common.prompts import format_prompt
    return format_prompt(instruction)


def format_response(response: str) -> str:
    """
    Input:
        response: Assistant response.

    Output:
        Clean response text.
    """
    return response.strip()


def build_sft_example(
    tokenizer: Tokenizer,
    instruction: str,
    response: str,
    bos_id: Optional[int],
    eos_id: Optional[int],
    block_size: int,
) -> Optional[dict]:
    """
    Builds one SFT training example.

    The model sees:
        prompt + response

    Loss is calculated only on response tokens.

    Input:
        tokenizer: custom BPE tokenizer.
        instruction: input instruction.
        response: expected answer.
        bos_id: beginning-of-sequence token id.
        eos_id: end-of-sequence token id.
        block_size: max sequence length.

    Output:
        dict with input_ids and labels, or None if invalid.
    """
    prompt = format_instruction_prompt(instruction)
    response = format_response(response)

    prompt_ids = tokenizer.encode(prompt).ids
    response_ids = tokenizer.encode(response).ids

    if bos_id is not None:
        prompt_ids = [bos_id] + prompt_ids

    if eos_id is not None:
        response_ids = response_ids + [eos_id]

    input_ids = prompt_ids + response_ids

    # Ignore prompt tokens in loss.
    labels = [-100] * len(prompt_ids) + response_ids

    if len(input_ids) > block_size:
        input_ids = input_ids[:block_size]
        labels = labels[:block_size]

    # Need at least one supervised response token.
    if all(x == -100 for x in labels):
        return None

    return {
        "input_ids": input_ids,
        "labels": labels,
    }


def build_dpo_pair(
    tokenizer: Tokenizer,
    prompt: str,
    chosen: str,
    rejected: str,
    bos_id: Optional[int],
    eos_id: Optional[int],
    block_size: int,
) -> Optional[dict]:
    """
    Builds one DPO preference pair.

    DPO compares:
        log probability of chosen answer
        vs.
        log probability of rejected answer

    Loss is calculated only on answer tokens.

    Input:
        tokenizer: custom BPE tokenizer.
        prompt: user prompt.
        chosen: preferred response.
        rejected: non-preferred response.
        bos_id/eos_id: optional special token ids.
        block_size: maximum sequence length.

    Output:
        dict containing chosen/rejected input_ids and labels.
    """
    prompt_text = format_instruction_prompt(prompt)

    prompt_ids = tokenizer.encode(prompt_text).ids

    if bos_id is not None:
        prompt_ids = [bos_id] + prompt_ids

    chosen_ids = tokenizer.encode(format_response(chosen)).ids
    rejected_ids = tokenizer.encode(format_response(rejected)).ids

    if eos_id is not None:
        chosen_ids = chosen_ids + [eos_id]
        rejected_ids = rejected_ids + [eos_id]

    chosen_input = prompt_ids + chosen_ids
    rejected_input = prompt_ids + rejected_ids

    chosen_labels = [-100] * len(prompt_ids) + chosen_ids
    rejected_labels = [-100] * len(prompt_ids) + rejected_ids

    if len(chosen_input) > block_size:
        chosen_input = chosen_input[:block_size]
        chosen_labels = chosen_labels[:block_size]

    if len(rejected_input) > block_size:
        rejected_input = rejected_input[:block_size]
        rejected_labels = rejected_labels[:block_size]

    if all(x == -100 for x in chosen_labels):
        return None

    if all(x == -100 for x in rejected_labels):
        return None

    return {
        "chosen_input_ids": chosen_input,
        "chosen_labels": chosen_labels,
        "rejected_input_ids": rejected_input,
        "rejected_labels": rejected_labels,
    }


class SFTDataset(Dataset):
    """
    Supervised fine-tuning dataset.

    Input:
        JSONL file with instruction-response rows.

    Output per item:
        input_ids
        labels
    """

    def __init__(
        self,
        path: str,
        tokenizer: Tokenizer,
        bos_id: Optional[int],
        eos_id: Optional[int],
        block_size: int,
    ):
        rows = read_jsonl(path)
        examples = []

        for row in rows:
            example = build_sft_example(
                tokenizer=tokenizer,
                instruction=row.get("prompt", row.get("instruction", "")),
                response=row["response"],
                bos_id=bos_id,
                eos_id=eos_id,
                block_size=block_size,
            )

            if example is not None:
                examples.append(example)

        if not examples:
            raise ValueError("No valid SFT examples found.")

        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class DPODataset(Dataset):
    """
    DPO preference dataset.

    Input:
        JSONL file with prompt/chosen/rejected rows.

    Output per item:
        chosen_input_ids
        chosen_labels
        rejected_input_ids
        rejected_labels
    """

    def __init__(
        self,
        path: str,
        tokenizer: Tokenizer,
        bos_id: Optional[int],
        eos_id: Optional[int],
        block_size: int,
    ):
        rows = read_jsonl(path)
        examples = []

        for row in rows:
            example = build_dpo_pair(
                tokenizer=tokenizer,
                prompt=row["prompt"],
                chosen=row["chosen"],
                rejected=row["rejected"],
                bos_id=bos_id,
                eos_id=eos_id,
                block_size=block_size,
            )

            if example is not None:
                examples.append(example)

        if not examples:
            raise ValueError("No valid DPO examples found.")

        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def pad_1d(values: List[int], max_len: int, pad_value: int) -> List[int]:
    """
    Pads one integer list to max_len.
    """
    return values + [pad_value] * (max_len - len(values))


def collate_sft(batch: list[dict], pad_id: int) -> dict:
    """
    Input:
        batch of SFT examples.

    Output:
        padded tensors.
    """
    max_len = max(len(x["input_ids"]) for x in batch)

    input_ids = []
    labels = []

    for item in batch:
        input_ids.append(pad_1d(item["input_ids"], max_len, pad_id))
        labels.append(pad_1d(item["labels"], max_len, -100))

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def collate_dpo(batch: list[dict], pad_id: int) -> dict:
    """
    Input:
        batch of DPO examples.

    Output:
        padded chosen/rejected tensors.
    """
    chosen_max = max(len(x["chosen_input_ids"]) for x in batch)
    rejected_max = max(len(x["rejected_input_ids"]) for x in batch)

    chosen_input_ids = []
    chosen_labels = []
    rejected_input_ids = []
    rejected_labels = []

    for item in batch:
        chosen_input_ids.append(
            pad_1d(item["chosen_input_ids"], chosen_max, pad_id)
        )
        chosen_labels.append(
            pad_1d(item["chosen_labels"], chosen_max, -100)
        )
        rejected_input_ids.append(
            pad_1d(item["rejected_input_ids"], rejected_max, pad_id)
        )
        rejected_labels.append(
            pad_1d(item["rejected_labels"], rejected_max, -100)
        )

    return {
        "chosen_input_ids": torch.tensor(chosen_input_ids, dtype=torch.long),
        "chosen_labels": torch.tensor(chosen_labels, dtype=torch.long),
        "rejected_input_ids": torch.tensor(rejected_input_ids, dtype=torch.long),
        "rejected_labels": torch.tensor(rejected_labels, dtype=torch.long),
    }
