import mlx.core as mx
import os
import sys

# Add parent dir to path so we can import src modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.model.transformer import TransformerLanguageModel
from src.tokenizer.bpe import SimpleTokenizer
from src.inference.generate import generate

def test_prompts():
    seq_length = 256
    embed_size = 256
    num_layers = 6
    heads = 8
    forward_expansion = 4
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    tokenizer_path = os.path.join(base_dir, "tokenizer.json")
    model_path = os.path.join(base_dir, "model_weights.safetensors")
    
    if not os.path.exists(tokenizer_path) or not os.path.exists(model_path):
        print("Error: Model weights or tokenizer not found.")
        return
        
    tokenizer = SimpleTokenizer()
    tokenizer.load(tokenizer_path)
    
    model = TransformerLanguageModel(
        vocab_size=tokenizer.vocab_size,
        embed_size=embed_size,
        num_layers=num_layers,
        heads=heads,
        forward_expansion=forward_expansion,
        max_length=seq_length
    )
    
    model.load_weights(model_path)
    mx.eval(model.parameters())
    print("Model loaded successfully.\n")
    
    prompts = [
        "To make charcoal, one must",
        "To gather iron ore, look for",
        "The process of building a stone tool starts with",
        "To smelt iron, the bloomery requires"
    ]
    
    for prompt in prompts:
        print("="*60)
        print(f"PROMPT: {prompt}")
        print("="*60)
        generate(model, tokenizer, prompt, max_new_tokens=400, temperature=0.7)
        print("\n")

if __name__ == "__main__":
    test_prompts()
