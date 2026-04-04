"""
model.py — Production-ready Transformer backbone
Fixes applied vs v1:
  - nn.LayerNorm instead of custom (more numerically stable)
  - Weight tying between input embedding and output projection
  - Device-safe mask utilities
  - Dropout after embeddings (matches paper)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────
# EMBEDDINGS
# ──────────────────────────────────────────────

class InputEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.d_model   = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe       = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_seq_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ──────────────────────────────────────────────
# ATTENTION
# ──────────────────────────────────────────────

def scaled_dot_product_attention(query, key, value, mask=None, dropout=None):

    d_k    = query.size(-1)
    # (Q*K.transpose)/sqrt(d_k)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    # softmax((Q*K.transpose)/sqrt(d_k))
    weights = F.softmax(scores, dim=-1)

    # Guard against all-inf rows (full padding) → NaN → zero
    weights = torch.nan_to_num(weights, nan=0.0)

    if dropout is not None:
        weights = dropout(weights)

    return torch.matmul(weights, value), weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % h == 0
        self.h   = h
        self.d_k = d_model // h

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(p=dropout)

    def _split_heads(self, t):
        B, S, _ = t.size()
        # ( B, S, d_model ) -> ( B, h, S, d_k )
        return t.view(B, S, self.h, self.d_k).transpose(1, 2)

    def forward(self, query, key, value, mask=None):
        Q = self._split_heads(self.W_q(query))
        K = self._split_heads(self.W_k(key))
        V = self._split_heads(self.W_v(value))
        # dimensions of Q, K and V : ( B, h, S, d_k )

        x, self.attn_weights = scaled_dot_product_attention(Q, K, V, mask, self.dropout)

        # ( B, h, S, d_k ) -> ( B, S, d_model: h*d_K )
        B, _, S, _ = x.size()
        x = x.transpose(1, 2).contiguous().view(B, S, self.h * self.d_k)
        return self.W_o(x) # Let all heads stacked mix together -> W_o * (softmax((Q*K_transpose)/sqrt(d_k))*V)


# ──────────────────────────────────────────────
# FEED-FORWARD
# ──────────────────────────────────────────────

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),                  # GELU slightly outperforms ReLU in practice
            nn.Dropout(p=dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(p=dropout),
        )

    def forward(self, x):
        return self.net(x)


# ──────────────────────────────────────────────
# RESIDUAL + NORM
# ──────────────────────────────────────────────

class ResidualConnection(nn.Module):
    """Pre-LN residual (more stable than post-LN for deep stacks)."""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.norm    = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        #sublayers -> can be MultiHeadAttention, FeedForward
        return x + self.dropout(sublayer(self.norm(x)))


# ──────────────────────────────────────────────
# ENCODER
# ──────────────────────────────────────────────

class EncoderLayer(nn.Module):
    def __init__(self, d_model, h, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, h, dropout)
        self.ffn       = FeedForward(d_model, d_ff, dropout)
        self.res1      = ResidualConnection(d_model, dropout)
        self.res2      = ResidualConnection(d_model, dropout)

    def forward(self, x, src_mask=None):
        x = self.res1(x, lambda x: self.self_attn(x, x, x, src_mask))
        x = self.res2(x, self.ffn)
        return x


class Encoder(nn.Module):
    def __init__(self, d_model, h, d_ff, N, dropout):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, h, d_ff, dropout) for _ in range(N)])
        self.norm   = nn.LayerNorm(d_model)

    def forward(self, x, src_mask=None):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)


# ──────────────────────────────────────────────
# DECODER
# ──────────────────────────────────────────────

class DecoderLayer(nn.Module):
    def __init__(self, d_model, h, d_ff, dropout):
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, h, dropout)
        self.cross_attn = MultiHeadAttention(d_model, h, dropout)
        self.ffn        = FeedForward(d_model, d_ff, dropout)
        self.res        = nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range(3)])

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        x = self.res[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.res[1](x, lambda x: self.cross_attn(x, enc_out, enc_out, src_mask))
        x = self.res[2](x, self.ffn)
        return x


class Decoder(nn.Module):
    def __init__(self, d_model, h, d_ff, N, dropout):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, h, d_ff, dropout) for _ in range(N)])
        self.norm   = nn.LayerNorm(d_model)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return self.norm(x)


# ──────────────────────────────────────────────
# MASK UTILITIES  (device-safe)
# ──────────────────────────────────────────────

def make_src_mask(src: torch.Tensor, pad_idx: int = 0):
    """(B, 1, 1, S)  — 1 where token is real, 0 where padding."""
    return (src != pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt: torch.Tensor, pad_idx: int = 0):
    """(B, 1, T, T)  — combined causal + padding mask."""
    T        = tgt.size(1)
    pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)           # (B,1,1,T)
    causal   = torch.tril(torch.ones(T, T, device=tgt.device, dtype=torch.bool))  # ← same device
    return pad_mask & causal                                          # (B,1,T,T)


# ──────────────────────────────────────────────
# TRANSFORMER  (full model)
# ──────────────────────────────────────────────

class Transformer(nn.Module):
    """
    Encoder-Decoder Transformer — training-ready.

    Key features:
      • Weight tying between target embedding and output projection
      • Pre-LN residuals for stable deep training
      • Device-safe causal masks
      • GELU activations
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model:    int   = 512,
        h:          int   = 8,
        N:          int   = 6,
        d_ff:       int   = 2048,
        max_seq_len:int   = 5000,
        dropout:    float = 0.1,
        pad_idx:    int   = 0,
        tie_weights:bool  = True,
    ):
        super().__init__()
        self.pad_idx = pad_idx

        # ── Embeddings ──
        self.src_embed = InputEmbedding(src_vocab_size, d_model)
        self.tgt_embed = InputEmbedding(tgt_vocab_size, d_model)
        self.pos_enc   = PositionalEncoding(d_model, max_seq_len, dropout)

        # ── Encoder / Decoder ──
        self.encoder = Encoder(d_model, h, d_ff, N, dropout)
        self.decoder = Decoder(d_model, h, d_ff, N, dropout)

        # ── Output head ──
        self.output_head = nn.Linear(d_model, tgt_vocab_size, bias=False)

        # Weight tying: output projection shares weights with target embedding
        if tie_weights:
            self.output_head.weight = self.tgt_embed.embedding.weight

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "embedding" in name:
                nn.init.normal_(p, mean=0, std=0.02)
            elif p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ── Sub-steps (useful for inference) ──────────

    def encode(self, src):
        mask = make_src_mask(src, self.pad_idx)
        return self.encoder(self.pos_enc(self.src_embed(src)), mask), mask

    def decode(self, tgt, enc_out, src_mask):
        tgt_mask = make_tgt_mask(tgt, self.pad_idx)
        return self.decoder(self.pos_enc(self.tgt_embed(tgt)), enc_out, src_mask, tgt_mask)

    # ── Full forward (training) ────────────────────

    def forward(self, src, tgt):
        """
        src : (B, S)   — source token IDs
        tgt : (B, T)   — target token IDs  (shifted right; <sos> prepended)
        returns logits : (B, T, tgt_vocab_size)
        """
        enc_out, src_mask = self.encode(src)
        dec_out           = self.decode(tgt, enc_out, src_mask)
        return self.output_head(dec_out)
