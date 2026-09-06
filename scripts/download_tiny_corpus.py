"""
Generates a tiny local corpus for debugging.
Saves to data/raw/tiny_corpus.txt
"""

import os

def generate_local_corpus(filepath: str, size_mb: float = 1.0):
    print(f"Generating {size_mb}MB local tiny corpus to {filepath}...")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    base_text = (
        "The future of artificial intelligence is full of exciting possibilities. "
        "Generative AI models like Transformers use self-attention mechanisms to understand language. "
        "Byte-Pair Encoding (BPE) helps tokenizers manage vocabulary effectively. "
        "Distributed training across multiple GPUs allows us to scale these models significantly. "
        "Techniques like Rotary Positional Embeddings (RoPE), RMSNorm, and SwiGLU improve performance and stability. "
    )
    
    target_bytes = int(size_mb * 1024 * 1024)
    current_bytes = 0
    
    with open(filepath, 'w', encoding='utf-8') as f:
        while current_bytes < target_bytes:
            f.write(base_text + "\n")
            current_bytes += len(base_text.encode('utf-8')) + 1
            
    print("Generation complete!")

if __name__ == "__main__":
    out_path = os.path.join(os.path.dirname(__file__), "../data/raw/tiny_corpus.txt")
    generate_local_corpus(out_path, size_mb=1.0)
