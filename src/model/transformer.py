import mlx.core as mx
import mlx.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.q_proj = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.k_proj = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.v_proj = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.fc_out = nn.Linear(self.embed_size, self.embed_size)

    def __call__(self, x, mask=None):
        B, seq_length, _ = x.shape

        # Linear projections
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        # Reshape to (B, seq_length, heads, head_dim) and transpose to (B, heads, seq_length, head_dim)
        queries = queries.reshape(B, seq_length, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, seq_length, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        values = values.reshape(B, seq_length, self.heads, self.head_dim).transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        energy = queries @ keys.transpose(0, 1, 3, 2) / math.sqrt(self.head_dim)

        if mask is not None:
            energy = energy + mask

        attention = mx.softmax(energy, axis=-1)
        
        # Apply attention to values
        out = attention @ values
        
        # Reshape back to (B, seq_length, embed_size)
        out = out.transpose(0, 2, 1, 3).reshape(B, seq_length, self.embed_size)

        return self.fc_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion):
        super().__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.GELU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

    def __call__(self, x, mask=None):
        attention = self.attention(self.norm1(x), mask)
        x = x + attention
        forward = self.feed_forward(self.norm2(x))
        x = x + forward
        return x


class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, heads, forward_expansion, max_length):
        super().__init__()
        self.embed_size = embed_size
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = [
            TransformerBlock(embed_size, heads, forward_expansion)
            for _ in range(num_layers)
        ]
        self.norm_out = nn.LayerNorm(embed_size)
        self.fc_out = nn.Linear(embed_size, vocab_size, bias=False)

    def __call__(self, x):
        B, seq_length = x.shape
        
        # Create positions
        positions = mx.arange(0, seq_length)[None, :]
        
        # Create causal mask for language modeling
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_length)

        # Embeddings
        out = self.word_embedding(x) + self.position_embedding(positions)

        # Forward through layers
        for layer in self.layers:
            out = layer(out, mask)

        out = self.norm_out(out)
        return self.fc_out(out)
