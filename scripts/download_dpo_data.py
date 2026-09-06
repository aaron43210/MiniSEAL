import os
import json
from datasets import load_dataset

def main():
    print("=== DOWNLOADING DPO PREFERENCE DATA ===")
    out_dir = "data/alignment"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "dpo.jsonl")
    
    print("Fetching Anthropic/hh-rlhf (subset)...")
    dataset = load_dataset("Anthropic/hh-rlhf", split="train", streaming=True)
    
    # We will take the first 1000 samples to keep training fast for testing
    num_samples = 1000
    
    with open(out_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(dataset):
            if i >= num_samples:
                break
                
            # HH-RLHF has 'chosen' and 'rejected' strings.
            # They already contain the full conversation history.
            # E.g. "Human: <prompt>\n\nAssistant: <response>"
            # We will format them as a dictionary.
            
            chosen = row["chosen"]
            rejected = row["rejected"]
            
            # Since the DPO dataset logic usually needs the prompt and response separated,
            # we can split the text. Anthropic's data ends with "\n\nAssistant: "
            # We'll split on the last Assistant tag to extract the prompt.
            
            split_tag = "\n\nAssistant: "
            if split_tag in chosen and split_tag in rejected:
                prompt_end_idx = chosen.rfind(split_tag) + len(split_tag)
                prompt = chosen[:prompt_end_idx]
                chosen_response = chosen[prompt_end_idx:]
                rejected_response = rejected[rejected.rfind(split_tag) + len(split_tag):]
                
                record = {
                    "prompt": prompt,
                    "chosen": chosen_response,
                    "rejected": rejected_response
                }
                f.write(json.dumps(record) + "\n")
                
    print(f"Saved {num_samples} preference pairs to {out_path}")

if __name__ == "__main__":
    main()
