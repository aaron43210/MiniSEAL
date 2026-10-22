import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.model.config import ModelConfig
from src.model.llm import DecoderOnlyTransformer

def convert():
    print("Downloading HuggingFace model: HuggingFaceTB/SmolLM-135M...")
    hf_model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M")
    hf_state_dict = hf_model.state_dict()

    print("Initializing custom DecoderOnlyTransformer to match SmolLM-135M...")
    config = ModelConfig(
        vocab_size=49152,
        block_size=2048,
        n_layer=30,
        n_head=9,
        n_embd=576,
    )
    custom_model = DecoderOnlyTransformer(config)
    custom_state_dict = custom_model.state_dict()

    print("Mapping weights...")
    mapped_state_dict = {}

    # Token embeddings
    mapped_state_dict["tok_embeddings.weight"] = hf_state_dict["model.embed_tokens.weight"]
    # LM head
    mapped_state_dict["lm_head.weight"] = hf_state_dict["lm_head.weight"]
    # Final norm
    mapped_state_dict["norm.weight"] = hf_state_dict["model.norm.weight"]

    # SmolLM uses Grouped Query Attention (3 KV heads for 9 Q heads). 
    # Your custom model uses Multi-Head Attention (9 KV heads for 9 Q heads).
    # We will seamlessly convert it by duplicating each KV head 3 times!
    n_head = config.n_head
    n_kv_head = 3
    head_dim = config.n_embd // config.n_head
    repeats = n_head // n_kv_head

    for i in range(config.n_layer):
        # Attention - Q and Out are 1:1
        mapped_state_dict[f"blocks.{i}.attn.q_proj.weight"] = hf_state_dict[f"model.layers.{i}.self_attn.q_proj.weight"]
        mapped_state_dict[f"blocks.{i}.attn.out_proj.weight"] = hf_state_dict[f"model.layers.{i}.self_attn.o_proj.weight"]
        
        # Attention - K and V need to be duplicated
        hf_k = hf_state_dict[f"model.layers.{i}.self_attn.k_proj.weight"]
        hf_v = hf_state_dict[f"model.layers.{i}.self_attn.v_proj.weight"]
        
        # Reshape to [kv_heads, head_dim, embd], repeat, reshape back to [n_head * head_dim, embd]
        mapped_k = hf_k.view(n_kv_head, head_dim, -1).repeat_interleave(repeats, dim=0).view(n_head * head_dim, -1)
        mapped_v = hf_v.view(n_kv_head, head_dim, -1).repeat_interleave(repeats, dim=0).view(n_head * head_dim, -1)
        
        mapped_state_dict[f"blocks.{i}.attn.k_proj.weight"] = mapped_k
        mapped_state_dict[f"blocks.{i}.attn.v_proj.weight"] = mapped_v
        
        # Norms
        mapped_state_dict[f"blocks.{i}.attn_norm.weight"] = hf_state_dict[f"model.layers.{i}.input_layernorm.weight"]
        mapped_state_dict[f"blocks.{i}.ffn_norm.weight"] = hf_state_dict[f"model.layers.{i}.post_attention_layernorm.weight"]

        # MLP (SwiGLU)
        mapped_state_dict[f"blocks.{i}.mlp.gate_proj.weight"] = hf_state_dict[f"model.layers.{i}.mlp.gate_proj.weight"]
        mapped_state_dict[f"blocks.{i}.mlp.up_proj.weight"] = hf_state_dict[f"model.layers.{i}.mlp.up_proj.weight"]
        mapped_state_dict[f"blocks.{i}.mlp.down_proj.weight"] = hf_state_dict[f"model.layers.{i}.mlp.down_proj.weight"]

    print("Verifying all weights mapped correctly...")
    for k in custom_state_dict.keys():
        if k not in mapped_state_dict:
            if "rope.inv_freq" in k:
                mapped_state_dict[k] = custom_state_dict[k]
            else:
                print(f"WARNING: Missing key {k}")
                mapped_state_dict[k] = custom_state_dict[k]
                
    custom_model.load_state_dict(mapped_state_dict)

    out_dir = "model_weights/smollm_135M"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "latest.pt")
    
    checkpoint = {
        "model": custom_model.state_dict(),
        "model_config": custom_model.config_dict(),
    }
    
    torch.save(checkpoint, out_path)
    
    # We must also save the tokenizer config so app.py knows to load the HF tokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    tokenizer.save_pretrained(out_dir)
    
    print(f"Success! Saved converted model and tokenizer to {out_dir}")

if __name__ == "__main__":
    convert()
