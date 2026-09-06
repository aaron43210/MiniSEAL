import os
import sys

try:
    from datasets import load_dataset
except ImportError:
    print("CRITICAL: 'datasets' library is missing.")
    print("Please install it: pip3 install datasets")
    sys.exit(1)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(project_root, "data")
wiki_path = os.path.join(data_dir, "wikipedia_and_duk.txt")
duk_path = os.path.join(data_dir, "duk_data.txt")

def main():
    print("Downloading a subset of Wikipedia (this might take a minute)...")
    # Load 10,000 articles from wikipedia
    dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train[:10000]", trust_remote_code=True)
    
    print("Extracting text and saving to disk...")
    with open(wiki_path, "w", encoding="utf-8") as out_f:
        for item in dataset:
            out_f.write(item["text"] + "\n\n")
            
    print("Appending original DUK data to the end...")
    if os.path.exists(duk_path):
        with open(duk_path, "r", encoding="utf-8") as duk_f:
            duk_data = duk_f.read()
            
        with open(wiki_path, "a", encoding="utf-8") as out_f:
            # Append it 50 times so the model heavily memorizes it during pre-training
            for _ in range(50):
                out_f.write("\n\n" + duk_data + "\n\n")
    
    print(f"Dataset successfully created at: {wiki_path}")
    print("You can now train the tokenizer and pre-train the model on this massive file!")

if __name__ == "__main__":
    main()
