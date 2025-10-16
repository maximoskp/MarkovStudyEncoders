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
    def __init__(self, d_model, n_heads, device='cpu'):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, device=device)
        self.last_attn = None  # store for visualization

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        out, attn_weights = self.attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False
        )
        self.last_attn = attn_weights  # (batch, heads, seq, seq)
        return out
# end SelfAttention

class CrossAttention(nn.Module):
    def __init__(self, d_model, n_heads, device='cpu'):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, device=device)
        self.last_attn = None

    def forward(self, q, kv, attn_mask=None, key_padding_mask=None):
        out, attn_weights = self.attn(
            q, kv, kv,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False
        )
        self.last_attn = attn_weights  # (batch, heads, q_len, kv_len)
        return out
# end CrossAttention

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dim_ff, device='cpu'):
        super().__init__()
        self.self_attn = SelfAttention(d_model, n_heads, device=device)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff, device=device),
            nn.GELU(),
            nn.Linear(dim_ff, d_model, device=device),
        )
        self.norm1 = nn.LayerNorm(d_model, device=device)
        self.norm2 = nn.LayerNorm(d_model, device=device)
        self.last_attn_weights = None  # place to store the weights
    # end init

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        # Self-attention
        sa = self.self_attn(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        self.last_attn_weights = self.self_attn.last_attn.detach()  # store for later
        x = self.norm1(x + sa)
        # Feedforward
        ff = self.ff(x)
        x = self.norm2(x + ff)
        return x
# end TransformerBlock

class CrossTransformerBlock(nn.Module):
    """For H-encoder with self + cross attention"""
    def __init__(self, d_model, n_heads, dim_ff, device='cpu'):
        super().__init__()
        self.self_attn = SelfAttention(d_model, n_heads, device=device)
        self.cross_attn = CrossAttention(d_model, n_heads, device=device)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff, device=device),
            nn.GELU(),
            nn.Linear(dim_ff, d_model, device=device),
        )
        self.norm1 = nn.LayerNorm(d_model, device=device)
        self.norm2 = nn.LayerNorm(d_model, device=device)
        self.norm3 = nn.LayerNorm(d_model, device=device)
        self.self_attn_weights = None  # place to store the weights
        self.cross_attn_weights = None  # place to store the weights
    # end init

    def forward(self, x, mem, attn_mask=None, key_padding_mask=None):
        # Self-attention on H
        sa = self.self_attn(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        self.self_attn_weights = self.self_attn.last_attn.detach()  # store for later
        x = self.norm1(x + sa)
        # Cross-attention to M
        ca = self.cross_attn(x, mem)
        self.cross_attn_weights = self.cross_attn.last_attn.detach()  # store for later
        x = self.norm2(x + ca)
        # Feedforward
        ff = self.ff(x)
        x = self.norm3(x + ff)
        return x
# end CrossTransformerBlock

class OnlyCrossTransformerBlock(nn.Module):
    """For H-encoder with self + cross attention"""
    def __init__(self, d_model, n_heads, dim_ff, device='cpu'):
        super().__init__()
        self.cross_attn = CrossAttention(d_model, n_heads, device=device)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff, device=device),
            nn.GELU(),
            nn.Linear(dim_ff, d_model, device=device),
        )
        self.norm1 = nn.LayerNorm(d_model, device=device)
        self.norm2 = nn.LayerNorm(d_model, device=device)
        self.norm3 = nn.LayerNorm(d_model, device=device)
        self.cross_attn_weights = None  # place to store the weights
    # end init

    def forward(self, x, mem, attn_mask=None, key_padding_mask=None):
        # no self-attention on H, keep only as residual
        x = self.norm1(x)
        # Cross-attention to M
        ca = self.cross_attn(x, mem)
        self.cross_attn_weights = self.cross_attn.last_attn.detach()  # store for later
        x = self.norm2(x + ca)
        # Feedforward
        ff = self.ff(x)
        x = self.norm3(x + ff)
        return x
# end OnlyCrossTransformerBlock

# ======== Models ========

class DualEncoderModel(nn.Module):
    def __init__(
            self, 
            m_vocab_size, 
            h_vocab_size, 
            seq_len, 
            d_model=128, 
            n_heads=4, 
            num_layers=2, 
            dim_ff=256,
            device='cpu'
        ):
        super().__init__()
        self.pos = sinusoidal_positional_encoding(
            seq_len, d_model, device=device
        )
        self.m_embed = nn.Embedding(m_vocab_size, d_model, device=device)
        self.h_embed = nn.Embedding(h_vocab_size, d_model, device=device)

        self.melody_encoder = nn.ModuleList([TransformerBlock(d_model, n_heads, dim_ff, device=device) for _ in range(num_layers)])
        self.harmony_encoder = nn.ModuleList([CrossTransformerBlock(d_model, n_heads, dim_ff, device=device) for _ in range(num_layers)])

        self.out_proj = nn.Linear(d_model, h_vocab_size, device=device)
        self.device = device
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

    # optionally add helpers to extract attention maps across layers:
    def get_attention_maps(self):
        """
        Returns lists of per-layer attention tensors for self and cross attentions.
            self_attns = [layer.last_self_attn, ...]
            cross_attns = [layer.last_cross_attn, ...]
        Each element can be None (if not computed) or a tensor (B, nhead, Lh, Lh)/(B, nhead, Lh, Lm).
        """
        self_attns = []
        cross_attns = []
        for layer in self.harmony_encoder:
            self_attns.append(layer.self_attn_weights)
            cross_attns.append(layer.cross_attn_weights)
        return self_attns, cross_attns
    # end get_attention_maps
# end DualEncoderModel

class DE_no_cross(nn.Module):
    def __init__(
            self, 
            m_vocab_size, 
            h_vocab_size, 
            seq_len, 
            d_model=128, 
            n_heads=4, 
            num_layers=2, 
            dim_ff=256,
            device='cpu'
        ):
        super().__init__()
        self.pos = sinusoidal_positional_encoding(
            seq_len, d_model, device=device
        )
        self.m_embed = nn.Embedding(m_vocab_size, d_model, device=device)
        self.h_embed = nn.Embedding(h_vocab_size, d_model, device=device)

        self.melody_encoder = nn.ModuleList([TransformerBlock(d_model, n_heads, dim_ff, device=device) for _ in range(num_layers)])
        self.harmony_encoder = nn.ModuleList([OnlyCrossTransformerBlock(d_model, n_heads, dim_ff, device=device) for _ in range(num_layers)])

        self.out_proj = nn.Linear(d_model, h_vocab_size, device=device)
        self.device = device
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

    # optionally add helpers to extract attention maps across layers:
    def get_attention_maps(self):
        """
        Returns lists of per-layer attention tensors for self and cross attentions.
            self_attns = [layer.last_self_attn, ...]
            cross_attns = [layer.last_cross_attn, ...]
        Each element can be None (if not computed) or a tensor (B, nhead, Lh, Lh)/(B, nhead, Lh, Lm).
        """
        self_attns = []
        cross_attns = []
        for layer in self.harmony_encoder:
            cross_attns.append(layer.cross_attn_weights)
        return self_attns, cross_attns
    # end get_attention_maps
# end DE_no_cross

class DE_learned_pos(nn.Module):
    def __init__(
            self, 
            m_vocab_size, 
            h_vocab_size, 
            seq_len, 
            d_model=128, 
            n_heads=4, 
            num_layers=2, 
            dim_ff=256,
            device='cpu'
        ):
        super().__init__()
        self.pos = sinusoidal_positional_encoding(
            seq_len, d_model, device=device
        )
        self.learned_pos = nn.Parameter(torch.zeros(1, seq_len, d_model, device=device))
        self.m_embed = nn.Embedding(m_vocab_size, d_model, device=device)
        self.h_embed = nn.Embedding(h_vocab_size, d_model, device=device)

        self.melody_encoder = nn.ModuleList([TransformerBlock(d_model, n_heads, dim_ff, device=device) for _ in range(num_layers)])
        self.harmony_encoder = nn.ModuleList([CrossTransformerBlock(d_model, n_heads, dim_ff, device=device) for _ in range(num_layers)])

        self.out_proj = nn.Linear(d_model, h_vocab_size, device=device)
        self.device = device
    # end init

    def forward(self, m_seq, h_seq, h_attn_mask=None):
        m = self.m_embed(m_seq)
        h = self.h_embed(h_seq)

        # m = m + self.pos
        m = m + self.learned_pos
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

    # optionally add helpers to extract attention maps across layers:
    def get_attention_maps(self):
        """
        Returns lists of per-layer attention tensors for self and cross attentions.
            self_attns = [layer.last_self_attn, ...]
            cross_attns = [layer.last_cross_attn, ...]
        Each element can be None (if not computed) or a tensor (B, nhead, Lh, Lh)/(B, nhead, Lh, Lm).
        """
        self_attns = []
        cross_attns = []
        for layer in self.harmony_encoder:
            self_attns.append(layer.self_attn_weights)
            cross_attns.append(layer.cross_attn_weights)
        return self_attns, cross_attns
    # end get_attention_maps
# end DE_learned_pos

class DE_no_pos(nn.Module):
    def __init__(
            self, 
            m_vocab_size, 
            h_vocab_size, 
            seq_len, 
            d_model=128, 
            n_heads=4, 
            num_layers=2, 
            dim_ff=256,
            device='cpu'
        ):
        super().__init__()
        self.pos = sinusoidal_positional_encoding(
            seq_len, d_model, device=device
        )
        self.m_embed = nn.Embedding(m_vocab_size, d_model, device=device)
        self.h_embed = nn.Embedding(h_vocab_size, d_model, device=device)

        self.melody_encoder = nn.ModuleList([TransformerBlock(d_model, n_heads, dim_ff, device=device) for _ in range(num_layers)])
        self.harmony_encoder = nn.ModuleList([CrossTransformerBlock(d_model, n_heads, dim_ff, device=device) for _ in range(num_layers)])

        self.out_proj = nn.Linear(d_model, h_vocab_size, device=device)
        self.device = device
    # end init

    def forward(self, m_seq, h_seq, h_attn_mask=None):
        m = self.m_embed(m_seq)
        h = self.h_embed(h_seq)
        
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

    # optionally add helpers to extract attention maps across layers:
    def get_attention_maps(self):
        """
        Returns lists of per-layer attention tensors for self and cross attentions.
            self_attns = [layer.last_self_attn, ...]
            cross_attns = [layer.last_cross_attn, ...]
        Each element can be None (if not computed) or a tensor (B, nhead, Lh, Lh)/(B, nhead, Lh, Lm).
        """
        self_attns = []
        cross_attns = []
        for layer in self.harmony_encoder:
            self_attns.append(layer.self_attn_weights)
            cross_attns.append(layer.cross_attn_weights)
        return self_attns, cross_attns
    # end get_attention_maps
# end DE_no_pos

class SingleEncoderModel(nn.Module):
    def __init__(
            self, 
            m_vocab_size, 
            h_vocab_size, 
            seq_len, 
            d_model=128, 
            n_heads=4, 
            num_layers=2, 
            dim_ff=256,
            device='cpu'
        ):
        super().__init__()
        self.pos = sinusoidal_positional_encoding(
            seq_len, d_model, device=device
        )
        self.seq_len = seq_len
        self.m_embed = nn.Embedding(m_vocab_size, d_model, device=device)
        self.h_embed = nn.Embedding(h_vocab_size, d_model, device=device)
        self.encoder = nn.ModuleList([TransformerBlock(d_model, n_heads, dim_ff, device=device) for _ in range(num_layers)])
        self.out_proj = nn.Linear(d_model, h_vocab_size, device=device)
        self.device = device
    # end init

    def forward(self, m_seq, h_seq, attn_mask=None):
        m = self.m_embed(m_seq)
        h = self.h_embed(h_seq)

        m = m + self.pos
        h = h + self.pos

        x = torch.cat([m, h], dim=1)
        for layer in self.encoder:
            x = layer(x, attn_mask=attn_mask)
        logits = self.out_proj(x[:, -self.seq_len:, :])
        return logits
    # end forward

    # optionally add helpers to extract attention maps across layers:
    def get_attention_maps(self):
        """
        Returns lists of per-layer attention tensors for self and cross attentions.
            self_attns = [layer.last_self_attn, ...]
        Each element can be None (if not computed) or a tensor (B, nhead, Lh, Lh)/(B, nhead, Lh, Lm).
        """
        self_attns = []
        for layer in self.encoder:
            self_attns.append(layer.last_attn_weights)
        return self_attns
    # end get_attention_maps
# end SingleEncoderModel

class SE_learned_pos(nn.Module):
    def __init__(
            self, 
            m_vocab_size, 
            h_vocab_size, 
            seq_len, 
            d_model=128, 
            n_heads=4, 
            num_layers=2, 
            dim_ff=256,
            device='cpu'
        ):
        super().__init__()
        self.pos = sinusoidal_positional_encoding(
            seq_len, d_model, device=device
        )
        self.learned_pos = nn.Parameter(torch.zeros(1, seq_len, d_model, device=device))
        self.seq_len = seq_len
        self.m_embed = nn.Embedding(m_vocab_size, d_model, device=device)
        self.h_embed = nn.Embedding(h_vocab_size, d_model, device=device)
        self.encoder = nn.ModuleList([TransformerBlock(d_model, n_heads, dim_ff, device=device) for _ in range(num_layers)])
        self.out_proj = nn.Linear(d_model, h_vocab_size, device=device)
        self.device = device
    # end init

    def forward(self, m_seq, h_seq, attn_mask=None):
        m = self.m_embed(m_seq)
        h = self.h_embed(h_seq)

        # m = m + self.pos
        m = m + self.learned_pos
        h = h + self.pos

        x = torch.cat([m, h], dim=1)
        for layer in self.encoder:
            x = layer(x, attn_mask=attn_mask)
        logits = self.out_proj(x[:, -self.seq_len:, :])
        return logits
    # end forward

    # optionally add helpers to extract attention maps across layers:
    def get_attention_maps(self):
        """
        Returns lists of per-layer attention tensors for self and cross attentions.
            self_attns = [layer.last_self_attn, ...]
        Each element can be None (if not computed) or a tensor (B, nhead, Lh, Lh)/(B, nhead, Lh, Lm).
        """
        self_attns = []
        for layer in self.encoder:
            self_attns.append(layer.last_attn_weights)
        return self_attns
    # end get_attention_maps
# end SE_learned_pos

class SE_no_pos(nn.Module):
    def __init__(
            self, 
            m_vocab_size, 
            h_vocab_size, 
            seq_len, 
            d_model=128, 
            n_heads=4, 
            num_layers=2, 
            dim_ff=256,
            device='cpu'
        ):
        super().__init__()
        self.pos = sinusoidal_positional_encoding(
            seq_len, d_model, device=device
        )
        self.seq_len = seq_len
        self.m_embed = nn.Embedding(m_vocab_size, d_model, device=device)
        self.h_embed = nn.Embedding(h_vocab_size, d_model, device=device)
        self.encoder = nn.ModuleList([TransformerBlock(d_model, n_heads, dim_ff, device=device) for _ in range(num_layers)])
        self.out_proj = nn.Linear(d_model, h_vocab_size, device=device)
        self.device = device
    # end init

    def forward(self, m_seq, h_seq, attn_mask=None):
        m = self.m_embed(m_seq)
        h = self.h_embed(h_seq)
        
        h = h + self.pos

        x = torch.cat([m, h], dim=1)
        for layer in self.encoder:
            x = layer(x, attn_mask=attn_mask)
        logits = self.out_proj(x[:, -self.seq_len:, :])
        return logits
    # end forward

    # optionally add helpers to extract attention maps across layers:
    def get_attention_maps(self):
        """
        Returns lists of per-layer attention tensors for self and cross attentions.
            self_attns = [layer.last_self_attn, ...]
        Each element can be None (if not computed) or a tensor (B, nhead, Lh, Lh)/(B, nhead, Lh, Lm).
        """
        self_attns = []
        for layer in self.encoder:
            self_attns.append(layer.last_attn_weights)
        return self_attns
    # end get_attention_maps
# end SE_no_pos