import os
import torch

class SystemConfig:
    """Global system paths and hardware auto-detection."""
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Auto-detect target compute device
    if torch.cuda.is_available():
        DEVICE = "cuda"
        NUM_GPUS = torch.cuda.device_count()
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
        NUM_GPUS = 1
    else:
        DEVICE = "cpu"
        NUM_GPUS = 0

    print(f"[SYSTEM] Selected Device: {DEVICE} | Available GPUs: {NUM_GPUS}")


class PathConfig:
    """Centralized workspace directory structure."""
    DATA_DIR = os.path.join(SystemConfig.PROJECT_ROOT, "data")
    RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
    ALIGNMENT_DIR = os.path.join(DATA_DIR, "alignment")
    SELF_ADAPT_DIR = os.path.join(DATA_DIR, "self_adapt")
    
    RUNS_DIR = os.path.join(SystemConfig.PROJECT_ROOT, "runs")
    DEBUG_LOCAL_DIR = os.path.join(RUNS_DIR, "debug_local")
    CHECKPOINTS_DIR = os.path.join(SystemConfig.PROJECT_ROOT, "checkpoints")
    
    # Tokenizer & Corpus
    TOKENIZER_PATH = os.path.join(SystemConfig.PROJECT_ROOT, "artifacts/tokenizer/tokenizer.json")
    RAW_CORPUS_PATH = os.path.join(RAW_DATA_DIR, "wikitext-2.txt")
    
    # Dataset Outputs
    SFT_DATA_PATH = os.path.join(ALIGNMENT_DIR, "sft.jsonl")
    PREFERENCES_DATA_PATH = os.path.join(ALIGNMENT_DIR, "preferences.jsonl")
    SQUAD_MINI_PATH = os.path.join(SELF_ADAPT_DIR, "squad_mini.jsonl")
    
    # Model Weights Progression
    PRETRAINED_BASE_MODEL = os.path.join(DEBUG_LOCAL_DIR, "latest.pt")
    SFT_MODEL_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, "model_sft.pt")
    DPO_MODEL_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, "model_aligned_dpo.pt")
    ADAPTED_MODEL_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, "model_adapted_seal.pt")


class TrainingHyperparameters:
    """Hyperparameters matching our model dimensions and budget."""
    VOCAB_SIZE = 32000
    MAX_SEQ_LEN = 512
    D_MODEL = 512
    N_LAYERS = 8
    N_HEADS = 8
    D_FF = 1024
    BATCH_SIZE = 4
    LEARNING_RATE = 3e-4


# Automatic directory setup on module import
os.makedirs(PathConfig.RAW_DATA_DIR, exist_ok=True)
os.makedirs(PathConfig.ALIGNMENT_DIR, exist_ok=True)
os.makedirs(PathConfig.SELF_ADAPT_DIR, exist_ok=True)
os.makedirs(PathConfig.DEBUG_LOCAL_DIR, exist_ok=True)
os.makedirs(PathConfig.CHECKPOINTS_DIR, exist_ok=True)
