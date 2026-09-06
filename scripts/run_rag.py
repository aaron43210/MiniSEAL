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

from src.model.llm import DecoderOnlyTransformer
from src.model.config import ModelConfig
from tokenizers import Tokenizer

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
    
    print("\n[4/4] Loading Custom 124M LLM...")
    tokenizer_path = os.path.join(project_root, "artifacts", "tokenizer", "tokenizer.json")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    # Try DPO first, then SFT, then Pretrain
    dpo_path = os.path.join(project_root, "checkpoints", "dgx_dpo_124M", "latest_dpo.pt")
    sft_path = os.path.join(project_root, "checkpoints", "dgx_sft_124M", "latest.pt")
    pretrain_path = os.path.join(project_root, "checkpoints", "dgx_pretrain_124M", "latest.pt")
    
    if os.path.exists(dpo_path):
        model_path = dpo_path
        print(f"Loading fully-aligned DPO model: {model_path}")
    elif os.path.exists(sft_path):
        model_path = sft_path
        print(f"Loading SFT model (DPO not found): {model_path}")
    else:
        model_path = pretrain_path
        print(f"Loading Pretrained model (SFT/DPO not found): {model_path}")
        
    checkpoint = torch.load(model_path, map_location=device)
    model_args = ModelConfig(**checkpoint['args']) if 'args' in checkpoint else ModelConfig(**checkpoint['model_config'])
    model = DecoderOnlyTransformer(model_args).to(device)
    model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint['model'])
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
        
        # 4. Format the final prompt for the LLM
        prompt = f"Context: {retrieved_context}\nQuestion: {query}\nAnswer: "
        
        # 5. Generate Answer
        input_ids = tokenizer.encode(prompt).ids
        x = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
        
        with torch.no_grad():
            output_ids = model.generate(x, max_new_tokens=50, temperature=0.7)
            
        # Decode only the newly generated tokens
        generated_tokens = output_ids[0][len(input_ids):].tolist()
        answer = tokenizer.decode(generated_tokens)
        
        print(f"\n[Custom LLM Answer]: {answer.strip()}")

if __name__ == "__main__":
    main()
