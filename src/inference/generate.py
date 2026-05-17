import mlx.core as mx
import os
import sys

# Add parent dir to path so we can import other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.transformer import TransformerLanguageModel
from tokenizer.bpe import SimpleTokenizer

def generate(model, tokenizer, prompt, max_new_tokens=500, temperature=0.8, top_k=None):
    # Encode prompt
    context = tokenizer.encode(prompt)
    x = mx.array([context])
    
    print(prompt, end="", flush=True)
    
    generated = []
    
    for _ in range(max_new_tokens):
        # Crop context if it gets too long for our model (seq_length=256)
        x_cond = x if x.shape[1] <= 256 else x[:, -256:]
        
        # Forward pass
        logits = model(x_cond)
        
        # Take the logits at the last step
        logits = logits[:, -1, :] / temperature
        
        # Apply top-k sampling if specified
        if top_k is not None:
            # Need to implement top_k logic for MLX, for now just sample from softmax
            pass
            
        # Sample from the distribution
        probs = mx.softmax(logits, axis=-1)
        next_token = mx.random.categorical(logits)
        
        # Append to sequence
        next_token_val = next_token.item()
        generated.append(next_token_val)
        
        # Add to input for next step
        x = mx.concatenate([x, mx.array([[next_token_val]])], axis=1)
        
        # Decode and print
        print(tokenizer.decode([next_token_val]), end="", flush=True)
        
    print()
    return tokenizer.decode(generated)

def run_inference():
    # Model parameters must match training!
    seq_length = 256
    embed_size = 256
    num_layers = 6
    heads = 8
    forward_expansion = 4
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    tokenizer_path = os.path.join(base_dir, "tokenizer.json")
    model_path = os.path.join(base_dir, "model_weights.safetensors")
    
    if not os.path.exists(tokenizer_path) or not os.path.exists(model_path):
        print("Error: Model weights or tokenizer not found. Please run the training script first.")
        return
        
    # Load tokenizer
    tokenizer = SimpleTokenizer()
    tokenizer.load(tokenizer_path)
    
    # Initialize model
    model = TransformerLanguageModel(
        vocab_size=tokenizer.vocab_size,
        embed_size=embed_size,
        num_layers=num_layers,
        heads=heads,
        forward_expansion=forward_expansion,
        max_length=seq_length
    )
    
    # Load weights
    model.load_weights(model_path)
    mx.eval(model.parameters())
    print("Model loaded successfully.")
    
    # Interactive generation loop
    print("\n--- STC Blueprint Generator ---")
    print("Type a prompt to generate technical text (or 'quit' to exit)")
    
    while True:
        prompt = input("\nPrompt: ")
        if prompt.lower() in ['quit', 'exit']:
            break
            
        if not prompt:
            prompt = "The high-carbon steel "
            
        print("\nGenerating...\n")
        generate(model, tokenizer, prompt)

if __name__ == "__main__":
    run_inference()
