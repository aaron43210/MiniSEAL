import argparse
import os
import yaml
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
import sys

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.common.utils import ensure_dir, set_seed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    set_seed(cfg.get("seed", 42))
    out_dir = cfg["out_dir"]
    ensure_dir(out_dir)

    base_model_path = cfg.get("pretrained_model_path", "Qwen/Qwen2.5-1.5B-Instruct")
    print(f"[SFT] Loading base model: {base_model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # No device_map="auto": the HF Trainer handles device placement itself, and
    # device_map conflicts with the Trainer on some single-/multi-GPU setups.
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Apply PEFT directly
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset_path = cfg["sft_data_path"]
    print(f"[SFT] Loading dataset from {dataset_path}")
    dataset = load_dataset('json', data_files=dataset_path, split='train')
    
    # We will format AND tokenize manually to avoid all third-party TRL bugs
    def tokenize_function(example):
        p = example['prompt']
        r = example['response']
        messages = [{"role": "user", "content": p}, {"role": "assistant", "content": r}]
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False)
        except Exception:
            text = f"User: {p}\nAssistant: {r}{tokenizer.eos_token}"
            
        return tokenizer(text, truncation=True, max_length=cfg.get("block_size", 1024))
        
    dataset = dataset.map(tokenize_function, batched=False)

    sft_config = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=cfg.get("micro_batch_size", 2),
        gradient_accumulation_steps=cfg.get("grad_accum_steps", 4),
        learning_rate=float(cfg.get("learning_rate", 2e-4)),
        logging_steps=cfg.get("log_interval", 10),
        max_steps=cfg.get("max_steps", 1000),
        save_steps=cfg.get("save_interval", 200),
        bf16=True if torch.cuda.is_available() else False,
        fp16=False,
    )

    # Standard HuggingFace Trainer (Bulletproof)
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        args=sft_config,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )

    print("\n[SFT] Starting HuggingFace LoRA Training...\n")
    trainer.train()
    
    final_path = os.path.join(out_dir, "final_adapter")
    trainer.model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\n[SFT] Saved final LoRA adapter to {final_path}")
if __name__ == "__main__":
    main()
