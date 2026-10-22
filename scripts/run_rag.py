import os
import sys
import torch
import torch.nn.functional as F

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("CRITICAL: 'sentence-transformers' library is missing.")
    print("Please install it by running: pip3 install sentence-transformers")
    sys.exit(1)

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def load_chunks(file_path, max_chars=1500, overlap=200):
    """Reads the file and creates overlapping chunks of a strict maximum size."""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Sliding window chunking to guarantee no chunk is too large
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        # Try to break at a natural boundary (newline or space) if possible
        if end < len(text):
            next_newline = text.rfind('\n', start, end)
            next_space = text.rfind(' ', start, end)
            # Break at the last newline if it's in the second half of the chunk, else space
            if next_newline > start + (max_chars // 2):
                end = next_newline
            elif next_space > start + (max_chars // 2):
                end = next_space
                
        chunk = text[start:end].strip()
        if len(chunk) > 20:
            chunks.append(chunk)
            
        start = end - overlap
        
    return chunks

def find_latest_adapter(search_roots):
    """Return the newest PEFT adapter directory across the given roots, or None.

    Matches any dir containing adapter_config.json so adapters from any stage
    (SFT final_adapter, SEAL seal_adapter, HITL/DPO lora_adapter_v*) are found.
    """
    found = []
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _dirs, files in os.walk(root):
            if "adapter_config.json" in files:
                found.append(dirpath)
    if not found:
        return None
    found.sort(key=os.path.getmtime, reverse=True)
    return found[0]

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print("\n[1/4] Loading RAG Document Chunks...")
    data_path = os.path.join(project_root, "data", "duk_data.txt")
    chunks = load_chunks(data_path)
    print(f"Loaded {len(chunks)} paragraphs from the document.")
    
    print("\n[2/4] Loading Embedding Model (all-MiniLM-L6-v2)...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    
    print("\n[3/4] Embedding Document Chunks...")
    chunk_embeddings = embedder.encode(chunks, convert_to_tensor=True, device=device)
    
    print("\n[4/4] Loading HuggingFace Model & LoRA Adapters...")
    
    # Load Base Qwen Model
    base_model_path = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    print(f"Loading Base Model: {base_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    # Discover the newest LoRA adapter from any stage (SFT/SEAL/HITL/DPO).
    adapter = find_latest_adapter([
        os.path.join(project_root, "ckpts", "adapters"),
        os.path.join(project_root, "model_weights"),
    ])
    if adapter:
        print(f"Loading latest LoRA adapter: {adapter}")
        model = PeftModel.from_pretrained(model, adapter)
    else:
        print("WARNING: No trained LoRA adapters found. Using base model only.")

    model.eval()
    print("Model loaded successfully!")
    
    print("\n" + "="*50)
    print(" RAG Chat Terminal - Type 'quit' to exit")
    print("="*50)
    
    while True:
        try:
            query = input("\n[You] Ask a question: ").strip()
        except EOFError:
            break
            
        if query.lower() in ['quit', 'exit', 'q']:
            break
        if not query:
            continue
            
        # 1. Embed the user's question
        query_embedding = embedder.encode(query, convert_to_tensor=True, device=device)
        
        # 2. Calculate Cosine Similarity
        similarities = F.cosine_similarity(query_embedding.unsqueeze(0), chunk_embeddings)
        
        # 3. Find the best chunk
        best_idx = torch.argmax(similarities).item()
        retrieved_context = chunks[best_idx]
        score = similarities[best_idx].item()
        
        print(f"\n[Retrieval] Found context (Score: {score:.2f}):")
        print(f"--- {retrieved_context[:100]}... ---")
        
        # 4. Format for Hugging Face Chat Template
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer the user's question accurately using ONLY the provided context."},
            {"role": "user", "content": f"Context: {retrieved_context}\n\nQuestion: {query}"}
        ]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(device)
        
        # 5. Generate Answer
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=150, 
                temperature=0.1,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
        # Decode only the newly generated tokens
        generated_tokens = output_ids[0][inputs.input_ids.shape[1]:]
        answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        print(f"\n[AI Answer]: {answer.strip()}")

if __name__ == "__main__":
    main()
