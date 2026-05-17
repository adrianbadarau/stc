import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils
import time
import math
import sys
import os

# Add parent dir to path so we can import other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.transformer import TransformerLanguageModel
from tokenizer.bpe import SimpleTokenizer

def get_batch(data, seq_length, batch_size):
    # Pick random starting points
    ix = mx.random.randint(0, len(data) - seq_length, (batch_size,))
    
    # Create batches of inputs (x) and targets (y)
    # y is x shifted by 1
    x = mx.stack([data[i:i+seq_length] for i in ix.tolist()])
    y = mx.stack([data[i+1:i+seq_length+1] for i in ix.tolist()])
    return x, y

def train():
    # Hyperparameters
    batch_size = 32
    seq_length = 256
    embed_size = 256
    num_layers = 6
    heads = 8
    forward_expansion = 4
    learning_rate = 3e-4
    max_iters = 5000
    eval_interval = 500
    
    # Load dataset
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "stc_training_data.txt")
    if not os.path.exists(data_path):
        print(f"Error: Training data not found at {data_path}. Please run the scraper first.")
        return
        
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
        
    # Initialize and train tokenizer
    tokenizer = SimpleTokenizer()
    tokenizer.train(text)
    
    # Save tokenizer for inference
    tokenizer_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tokenizer.json")
    tokenizer.save(tokenizer_path)
    
    vocab_size = tokenizer.vocab_size
    print(f"Vocabulary size: {vocab_size}")
    
    # Encode all data
    data = mx.array(tokenizer.encode(text))
    
    # Train/val split
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    # Initialize model
    model = TransformerLanguageModel(
        vocab_size=vocab_size,
        embed_size=embed_size,
        num_layers=num_layers,
        heads=heads,
        forward_expansion=forward_expansion,
        max_length=seq_length
    )
    
    # MLX initialization (lazy evaluation, so we need to force it)
    mx.eval(model.parameters())
    
    print(f"Model initialized with {sum(v.size for _, v in mlx.utils.tree_flatten(model.parameters())) / 1e6:.2f}M parameters")
    
    # Loss function (cross entropy)
    def loss_fn(model, x, y):
        logits = model(x)
        # Reshape logits to (batch_size * seq_length, vocab_size)
        # Reshape y to (batch_size * seq_length)
        logits = logits.reshape(-1, vocab_size)
        y = y.reshape(-1)
        return mx.mean(nn.losses.cross_entropy(logits, y))

    # Optimizer
    optimizer = optim.AdamW(learning_rate=learning_rate)
    
    state = [model.state, optimizer.state]
    
    # Value and grad function
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    def step(x, y):
        loss, grads = loss_and_grad_fn(model, x, y)
        optimizer.update(model, grads)
        return loss

    step_compiled = mx.compile(step, inputs=[model, optimizer], outputs=[model, optimizer])

    def evaluate(data, num_batches=10):
        total_loss = 0.0
        for _ in range(num_batches):
            x, y = get_batch(data, seq_length, batch_size)
            total_loss += loss_fn(model, x, y)
        return total_loss / num_batches
        
    evaluate_compiled = mx.compile(evaluate, inputs=model)

    # Training loop
    print("Starting training...")
    t0 = time.time()
    
    for iteration in range(max_iters):
        # Evaluate performance on train and val sets
        if iteration % eval_interval == 0 or iteration == max_iters - 1:
            train_loss = evaluate_compiled(train_data).item()
            val_loss = evaluate_compiled(val_data).item()
            print(f"step {iteration}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")
            
        # Sample a batch of data
        x, y = get_batch(train_data, seq_length, batch_size)
        
        # Take a step
        loss = step_compiled(x, y)
        mx.eval(model, optimizer, loss) # Force execution
        
    t1 = time.time()
    print(f"Training completed in {t1-t0:.2f} seconds")
    
    # Save the model
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "model_weights.safetensors")
    model.save_weights(model_path)
    print(f"Model weights saved to {model_path}")

if __name__ == "__main__":
    train()
