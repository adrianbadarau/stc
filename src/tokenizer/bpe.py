import json
import os

class SimpleTokenizer:
    """A basic character-level tokenizer as described in Phase 2."""
    
    def __init__(self):
        self.vocab = {}
        self.inverse_vocab = {}
        self.vocab_size = 0
        
    def train(self, text):
        # Find all unique characters
        unique_chars = sorted(list(set(text)))
        self.vocab_size = len(unique_chars)
        
        # Build vocabulary mapping
        self.vocab = {ch: i for i, ch in enumerate(unique_chars)}
        self.inverse_vocab = {i: ch for i, ch in enumerate(unique_chars)}
        
    def encode(self, text):
        return [self.vocab.get(ch, self.vocab.get('<UNK>', 0)) for ch in text]
        
    def decode(self, ids):
        return "".join([self.inverse_vocab.get(i, '') for i in ids])
        
    def save(self, filepath):
        with open(filepath, 'w') as f:
            json.dump({
                'vocab': self.vocab,
                'inverse_vocab': self.inverse_vocab
            }, f)
            
    def load(self, filepath):
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
                self.vocab = data['vocab']
                # JSON keys are strings, convert back to integers for inverse
                self.inverse_vocab = {int(k): v for k, v in data['inverse_vocab'].items()}
                self.vocab_size = len(self.vocab)

# For a production LLM, BPE would be implemented here, but character-level 
# is highly suitable for learning the architecture first.
