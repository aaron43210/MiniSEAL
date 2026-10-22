import os
import sys
import json
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

def load_chunks(file_path, max_chars=1000, overlap=0):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        if end < len(text):
            next_newline = text.rfind('\n', start, end)
            if next_newline > start + (max_chars // 2):
                end = next_newline
        chunk = text[start:end].strip()
        if len(chunk) > 50:
            chunks.append(chunk)
        start = end
    return chunks

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic SFT QA pairs from a corpus using a TEACHER model."
    )
    parser.add_argument(
        "--teacher-model",
        type=str,
        required=True,
        help=(
            "Path or HF id of the TEACHER model used to generate QA pairs. "
            "This MUST be a stronger, external model (e.g. TinyLlama/TinyLlama-1.1B-Chat, "
            "or another instruction-tuned model). Do NOT point this at this project's own "
            "SFT/DPO checkpoint (e.g. model_weights/dgx_sft_124M): using the student as its "
            "own teacher is circular self-distillation — it cannot inject new knowledge and "
            "only amplifies the student's existing errors and style."
        ),
    )
    parser.add_argument("--data-file", type=str, default=os.path.join(project_root, "data", "duk_data.txt"))
    parser.add_argument("--output-file", type=str, default=os.path.join(project_root, "data", "alignment", "duk_rag_sft_large.jsonl"))
    parser.add_argument("--max-chars", type=int, default=800)
    parser.add_argument("--max-new-tokens", type=int, default=150)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load data
    data_path = args.data_file
    chunks = load_chunks(data_path, max_chars=args.max_chars)
    print(f"Loaded {len(chunks)} text chunks.")

    # 2. Load the TEACHER model
    model_id = args.teacher_model
    if os.path.sep in model_id and not os.path.exists(model_id):
        print(f"Teacher model not found at {model_id}! Run download script first.")
        sys.exit(1)

    print(f"\nLoading teacher model from {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    model.eval()

    # 3. Setup output file
    output_file = args.output_file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"\nGenerating Synthetic QA Pairs. Saving to {output_file}...\n")
    
    success_count = 0
    with open(output_file, "w", encoding="utf-8") as f_out:
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}...")
            
            # Prompt the teacher model to generate a QA pair
            sys_prompt = "You are an automated data generation assistant. Extract exactly one highly accurate Question and Answer based solely on the provided Context."
            user_prompt = f"Context: {chunk}\n\nTask: Write one Question and its Answer based on the context. Format it exactly like this:\nQuestion: <your question>\nAnswer: <your answer>"
            
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
                
            generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            
            # Parse the output
            if "Question:" in generated_text and "Answer:" in generated_text:
                try:
                    q_part = generated_text.split("Question:")[1].split("Answer:")[0].strip()
                    a_part = generated_text.split("Answer:")[1].strip()
                    
                    if q_part and a_part:
                        # Format for the SFT training!
                        full_prompt = f"Use this context to answer the question: {chunk}\nQuestion: {q_part}"
                        
                        json_line = json.dumps({"prompt": full_prompt, "response": a_part})
                        f_out.write(json_line + "\n")
                        f_out.flush()
                        
                        print(f"  -> Q: {q_part}")
                        success_count += 1
                except Exception:
                    pass
                    
    print(f"\nGeneration Complete! Successfully created {success_count} QA pairs.")
    print(f"You can now point your SFT config to {output_file} to train your 124M model on a massive dataset!")

if __name__ == "__main__":
    main()
