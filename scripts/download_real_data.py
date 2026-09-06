import os
import sys
import json
from datasets import load_dataset
from tqdm import tqdm

# Ensure project root is in system path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from configs.model_config import PathConfig

def prepare_directories():
    os.makedirs(PathConfig.RAW_DATA_DIR, exist_ok=True)
    os.makedirs(PathConfig.ALIGNMENT_DIR, exist_ok=True)
    os.makedirs(PathConfig.SELF_ADAPT_DIR, exist_ok=True)
    os.makedirs(PathConfig.DEBUG_LOCAL_DIR, exist_ok=True)

def download_pretrain_data():
    """Downloads a rich blend of datasets for robust Pretraining."""
    print("\n[1/4] Downloading Pretraining Data Blend (Stories, Wiki, Web)...")
    
    datasets_to_fetch = [
        {"name": "TinyStories", "path": "roneneldan/TinyStories", "split": "train", "samples": 250_000, "out": "tinystories.txt"},
        {"name": "WikiText-2", "path": "wikitext", "config": "wikitext-2-raw-v1", "split": "train", "samples": 100_000, "out": "wikitext.txt"},
        {"name": "OpenWebText", "path": "Skylion007/openwebtext", "split": "train", "samples": 50_000, "out": "openwebtext.txt"}
    ]
    
    for ds_info in datasets_to_fetch:
        print(f"  -> Fetching {ds_info['name']}...")
        if "config" in ds_info:
            dataset = load_dataset(ds_info["path"], ds_info["config"], split=ds_info["split"], streaming=True)
        else:
            dataset = load_dataset(ds_info["path"], split=ds_info["split"], streaming=True)
            
        out_path = os.path.join(PathConfig.RAW_DATA_DIR, ds_info["out"])
        records_written = 0
        
        with open(out_path, "w", encoding="utf-8") as f:
            for item in tqdm(dataset, desc=f"Writing {ds_info['name']}", total=ds_info["samples"]):
                if records_written >= ds_info["samples"]:
                    break
                text = item["text"].strip()
                if text:
                    f.write(text + "\n<|eos|>\n")
                    records_written += 1
                    
        print(f"    Saved {records_written} documents to {out_path}")

def download_sft_data(num_samples=2000):
    """Downloads Stanford Alpaca and formats it for SFT."""
    print(f"\n[2/4] Downloading Alpaca (SFT) | Target: {num_samples} samples...")
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    
    records_written = 0
    with open(PathConfig.SFT_DATA_PATH, "w", encoding="utf-8") as f:
        for item in tqdm(dataset, desc="Formatting SFT"):
            if records_written >= num_samples:
                break
            
            prompt = item["instruction"]
            if item["input"].strip():
                prompt += f"\nInput: {item['input']}"
                
            record = {
                "prompt": prompt,
                "response": item["output"]
            }
            f.write(json.dumps(record) + "\n")
            records_written += 1
            
    print(f"-> Saved real SFT data to: {PathConfig.SFT_DATA_PATH}")

def download_dpo_data(num_samples=1000):
    """Downloads Orca DPO pairs and formats them for DPO alignment."""
    print(f"\n[3/4] Downloading Orca DPO Pairs | Target: {num_samples} samples...")
    dataset = load_dataset("Intel/orca_dpo_pairs", split="train")
    
    records_written = 0
    with open(PathConfig.PREFERENCES_DATA_PATH, "w", encoding="utf-8") as f:
        for item in tqdm(dataset, desc="Formatting DPO"):
            if records_written >= num_samples:
                break
                
            record = {
                "prompt": item["question"],
                "chosen": item["chosen"],
                "rejected": item["rejected"]
            }
            f.write(json.dumps(record) + "\n")
            records_written += 1
            
    print(f"-> Saved real DPO data to: {PathConfig.PREFERENCES_DATA_PATH}")

def download_squad_adaptation_data(num_samples=500):
    """Downloads SQuAD v1.1 for SEAL knowledge incorporation."""
    print(f"\n[4/4] Downloading SQuAD v1.1 (SEAL) | Target: {num_samples} passages...")
    dataset = load_dataset("squad", split="train")
    
    passage_groups = {}
    for item in tqdm(dataset, desc="Grouping SQuAD Passages"):
        passage = item["context"]
        if passage not in passage_groups:
            passage_groups[passage] = []
        
        passage_groups[passage].append({
            "q": item["question"],
            "a": item["answers"]["text"][0]
        })
        
    records_written = 0
    with open(PathConfig.SQUAD_MINI_PATH, "w", encoding="utf-8") as f:
        for passage, qa_pairs in passage_groups.items():
            if records_written >= num_samples:
                break
                
            record = {
                "passage": passage,
                "qa_pairs": qa_pairs[:3]
            }
            f.write(json.dumps(record) + "\n")
            records_written += 1
            
    print(f"-> Saved real SQuAD data to: {PathConfig.SQUAD_MINI_PATH}")

if __name__ == "__main__":
    print("=== STARTING REAL-WORLD DATA ACQUISITION ===")
    prepare_directories()
    download_pretrain_data()
    download_sft_data()
    download_dpo_data()
    download_squad_adaptation_data()
    print("\n=== DATA ACQUISITION COMPLETE: Ready for training on real data! ===")
