import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def sinusoidal_positional_encoding(seq_len, d_model, device):
    """Standard sinusoidal PE (Vaswani et al., 2017)."""
    position = torch.arange(seq_len, device=device).unsqueeze(1)  # (seq_len, 1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device) *
                         (-math.log(10000.0) / d_model))
    pe = torch.zeros(seq_len, d_model, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # (1, seq_len, d_model)
# end sinusoidal_positional_encoding

# ======== Transformer Blocks ========

class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.last_attn = None  # store for visualization

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        out, attn_weights = self.attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=True)
        self.last_attn = attn_weights  # (batch, heads, seq, seq)
        return out
# end SelfAttention

class CrossAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.last_attn = None

    def forward(self, q, kv, attn_mask=None, key_padding_mask=None):
        out, attn_weights = self.attn(q, kv, kv, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=True)
        self.last_attn = attn_weights  # (batch, heads, q_len, kv_len)
        return out
# end CrossAttention

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dim_ff):
        super().__init__()
        self.self_attn = SelfAttention(d_model, n_heads)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        # Self-attention
        sa = self.self_attn(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = self.norm1(x + sa)
        # Feedforward
        ff = self.ff(x)
        x = self.norm2(x + ff)
        return x
# end TransformerBlock

class CrossTransformerBlock(nn.Module):
    """For H-encoder with self + cross attention"""
    def __init__(self, d_model, n_heads, dim_ff):
        super().__init__()
        self.self_attn = SelfAttention(d_model, n_heads)
        self.cross_attn = CrossAttention(d_model, n_heads)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, mem, attn_mask=None, key_padding_mask=None):
        # Self-attention on H
        sa = self.self_attn(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = self.norm1(x + sa)
        # Cross-attention to M
        ca = self.cross_attn(x, mem)
        x = self.norm2(x + ca)
        # Feedforward
        ff = self.ff(x)
        x = self.norm3(x + ff)
        return x
# end CrossTransformerBlock

# ======== Models ========

class DualEncoderModel(nn.Module):
    def __init__(self, m_vocab_size, h_vocab_size, seq_len, d_model=128, n_heads=4, num_layers=2, dim_ff=256):
        super().__init__()
        self.pos = sinusoidal_positional_encoding(
            seq_len, d_model
        )
        self.m_embed = nn.Embedding(m_vocab_size, d_model)
        self.h_embed = nn.Embedding(h_vocab_size, d_model)

        self.melody_encoder = nn.ModuleList([TransformerBlock(d_model, n_heads, dim_ff) for _ in range(num_layers)])
        self.harmony_encoder = nn.ModuleList([CrossTransformerBlock(d_model, n_heads, dim_ff) for _ in range(num_layers)])

        self.out_proj = nn.Linear(d_model, h_vocab_size)
    # end init

    def forward(self, m_seq, h_seq, h_attn_mask=None):
        m = self.m_embed(m_seq)
        h = self.h_embed(h_seq)

        m = m + self.pos
        h = h + self.pos

        # Melody encoder
        for layer in self.melody_encoder:
            m = layer(m)

        # Harmony encoder with cross-attn
        for layer in self.harmony_encoder:
            h = layer(h, mem=m, attn_mask=h_attn_mask)

        logits = self.out_proj(h)
        return logits
    # end forward
# end DualEncoderModel

class SingleEncoderModel(nn.Module):
    def __init__(self, m_vocab_size, h_vocab_size, seq_len, d_model=128, n_heads=4, num_layers=4, dim_ff=256):
        super().__init__()
        self.pos = sinusoidal_positional_encoding(
            seq_len, d_model
        )
        self.m_embed = nn.Embedding(m_vocab_size, d_model)
        self.h_embed = nn.Embedding(h_vocab_size, d_model)
        self.encoder = nn.ModuleList([TransformerBlock(d_model, n_heads, dim_ff) for _ in range(num_layers)])
        self.out_proj = nn.Linear(d_model, h_vocab_size)
    # end init

    def forward(self, m_seq, h_seq, attn_mask=None):
        m = self.m_embed(m_seq)
        h = self.h_embed(h_seq)

        m = m + self.pos
        h = h + self.pos

        x = torch.cat([m, h], dim=1)
        for layer in self.encoder:
            x = layer(x, attn_mask=attn_mask)
        logits = self.out_proj(x)
        return logits
    # end forward
# end SingleEncoderModel