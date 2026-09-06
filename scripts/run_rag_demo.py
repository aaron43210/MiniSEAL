import os
import sys
import torch
import torch.nn.functional as F

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

try:
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("CRITICAL: Missing libraries.")
    print("Please install them: pip3 install sentence-transformers transformers accelerate")
    sys.exit(1)

def load_chunks(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    raw_chunks = text.split("\n\n")
    chunks = [c.strip() for c in raw_chunks if len(c.strip()) > 20]
    return chunks

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print("\n[1/4] Loading RAG Document Chunks...")
    data_path = os.path.join(project_root, "data", "duk_data.txt")
    chunks = load_chunks(data_path)
    
    print("\n[2/4] Loading Embedding Model (all-MiniLM-L6-v2)...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    
    print("\n[3/4] Embedding Document Chunks...")
    chunk_embeddings = embedder.encode(chunks, convert_to_tensor=True, device=device)
    
    print("\n[4/4] Loading Pre-Trained 'Smart' LLM (TinyLlama-1.1B)...")
    print("This will download the model the first time you run it.")
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    model.eval()
    
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
            
        # Embed question and retrieve context
        query_embedding = embedder.encode(query, convert_to_tensor=True, device=device)
        similarities = F.cosine_similarity(query_embedding.unsqueeze(0), chunk_embeddings)
        best_idx = torch.argmax(similarities).item()
        retrieved_context = chunks[best_idx]
        
        print(f"\n[Retrieval] Found context:")
        print(f"--- {retrieved_context[:150]}... ---")
        
        # Format for TinyLlama Chat model
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer the user's question. Be concise and accurate."},
            {"role": "user", "content": f"Context: {retrieved_context}\n\nQuestion: {query}"}
        ]
        
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
        # Decode only the newly generated tokens
        generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
        answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        print(f"\n[Smart LLM Answer]: {answer.strip()}")

if __name__ == "__main__":
    main()
