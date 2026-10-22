import os
import sys
import torch
import torch.nn.functional as F

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

try:
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer
except ImportError:
    print("CRITICAL: 'sentence-transformers' or 'transformers' library is missing.")
    print("Please install them by running: pip3 install sentence-transformers transformers")
    sys.exit(1)

from src.model.llm import DecoderOnlyTransformer
from src.model.config import ModelConfig

def load_chunks(file_path):
    """Reads the file and splits it into logical chunks (paragraphs)."""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Split by double newlines to get paragraphs
    raw_chunks = text.split("\n\n")
    # Clean up chunks
    chunks = [c.strip() for c in raw_chunks if len(c.strip()) > 20]
    return chunks

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print("\n[1/4] Loading RAG Document Chunks...")
    data_path = os.path.join(project_root, "data", "duk_data.txt")
    chunks = load_chunks(data_path)
    print(f"Loaded {len(chunks)} paragraphs from the document.")
    
    print("\n[2/4] Loading Embedding Model (all-MiniLM-L6-v2)...")
    # This is a tiny, highly efficient embedding model
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    
    print("\n[3/4] Embedding Document Chunks...")
    # Convert all chunks into vector embeddings
    chunk_embeddings = embedder.encode(chunks, convert_to_tensor=True, device=device)
    
    print("\n[4/4] Loading Converted SmolLM-135M Model...")
    
    smollm_dir = os.path.join(project_root, "model_weights", "smollm_135M")
    model_path = os.path.join(smollm_dir, "latest.pt")
    
    if not os.path.exists(model_path):
        print("SmolLM model not found! Please run the conversion script first.")
        return
        
    print(f"Loading Converted Model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(smollm_dir)
    
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model_config = ModelConfig(**ckpt["model_config"])
    
    model = DecoderOnlyTransformer(model_config)
    model.load_state_dict(ckpt["model"])
    model.to(device)
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
        
        # 2. Calculate Cosine Similarity against all document chunks
        # Cosine similarity tells us which chunk vector is closest to our query vector
        similarities = F.cosine_similarity(query_embedding.unsqueeze(0), chunk_embeddings)
        
        # 3. Find the index of the most similar chunk
        best_idx = torch.argmax(similarities).item()
        retrieved_context = chunks[best_idx]
        score = similarities[best_idx].item()
        
        print(f"\n[Retrieval] Found context (Score: {score:.2f}):")
        print(f"--- {retrieved_context[:100]}... ---")
        
        # 4. Format the final prompt to match the model's SFT format exactly
        full_prompt = f"Use this context to answer the question: {retrieved_context}\nQuestion: {query}"
        prompt_text = f"User: {full_prompt}\nAssistant: "
        
        # Tokenize with HF AutoTokenizer
        ids = tokenizer.encode(prompt_text)
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        
        # Generate answer dynamically!
        # eos_token_id=0 for SmolLM's <|endoftext|>
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=150,
            temperature=0.7,
            top_k=50,
            eos_token_id=0
        )
        
        # Decode output
        generated_tokens = output_ids[0][len(ids):].tolist()
        final_answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        print(f"\n[Custom LLM Answer]: {final_answer.strip()}")

if __name__ == "__main__":
    main()
