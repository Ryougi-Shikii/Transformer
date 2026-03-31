"""
Full Transformer Architecture from Scratch
Based on "Attention Is All You Need" (2017)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# 1. INPUT EMBEDDING
# ─────────────────────────────────────────────
class InputEmbedding(nn.Module):
    """Maps token indices to dense vectors and scales by sqrt(d_model)."""

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # x: (batch, seq_len)  →  (batch, seq_len, d_model)
        return self.embedding(x) * math.sqrt(self.d_model)


# ─────────────────────────────────────────────
# 2. POSITIONAL ENCODING
# ─────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    """
    Injects fixed sinusoidal position information into the embeddings.
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, max_seq_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_seq_len, d_model)               # (max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()  # (max_seq_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)         # even indices
        pe[:, 1::2] = torch.cos(position * div_term)         # odd  indices
        pe = pe.unsqueeze(0)                                  # (1, max_seq_len, d_model)
        self.register_buffer("pe", pe)                        # not a learnable param

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ─────────────────────────────────────────────
# 3. SCALED DOT-PRODUCT ATTENTION
# ─────────────────────────────────────────────
def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor = None,
    dropout: nn.Dropout = None,
):
    """
    Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V

    Shapes (single head):
        query, key, value : (batch, heads, seq_len, d_k)
        mask              : (batch, 1,     1,        seq_len)  or  (batch, 1, seq_len, seq_len)
    Returns:
        output : (batch, heads, seq_len, d_k)
        weights: (batch, heads, seq_len, seq_len)
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # (B, H, S, S)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    weights = F.softmax(scores, dim=-1)

    if dropout is not None:
        weights = dropout(weights)

    output = torch.matmul(weights, value)
    return output, weights


# ─────────────────────────────────────────────
# 4. MULTI-HEAD ATTENTION
# ─────────────────────────────────────────────
class MultiHeadAttention(nn.Module):
    """
    Splits Q/K/V into `h` heads, runs attention in parallel,
    then re-projects the concatenated result.
    """

    def __init__(self, d_model: int, h: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % h == 0, "d_model must be divisible by h"

        self.h = h
        self.d_k = d_model // h

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        # Linear projections
        Q = self.W_q(query)   # (B, S, d_model)
        K = self.W_k(key)
        V = self.W_v(value)

        # Split into heads  →  (B, h, S, d_k)
        def split_heads(t):
            B, S, _ = t.size()
            return t.view(B, S, self.h, self.d_k).transpose(1, 2)

        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)

        # Attention per head
        x, self.attention_weights = scaled_dot_product_attention(
            Q, K, V, mask=mask, dropout=self.dropout
        )

        # Merge heads  →  (B, S, d_model)
        B, _, S, _ = x.size()
        x = x.transpose(1, 2).contiguous().view(B, S, self.h * self.d_k)

        return self.W_o(x)


# ─────────────────────────────────────────────
# 5. POSITION-WISE FEED-FORWARD NETWORK
# ─────────────────────────────────────────────
class FeedForward(nn.Module):
    """Two linear layers with a ReLU in between (per position independently)."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────
# 6. LAYER NORMALISATION + RESIDUAL CONNECTION
# ─────────────────────────────────────────────
class LayerNorm(nn.Module):
    """Standard layer normalisation with learnable scale & bias."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta  = nn.Parameter(torch.zeros(d_model))
        self.eps   = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std  = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class ResidualConnection(nn.Module):
    """Pre-LN residual: LayerNorm → sublayer → dropout → add."""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.norm    = LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


# ─────────────────────────────────────────────
# 7. ENCODER LAYER  &  FULL ENCODER
# ─────────────────────────────────────────────
class EncoderLayer(nn.Module):
    """One Transformer encoder block: Self-Attention → FFN (each with residual)."""

    def __init__(self, d_model: int, h: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, h, dropout)
        self.feed_fwd   = FeedForward(d_model, d_ff, dropout)
        self.residual1  = ResidualConnection(d_model, dropout)
        self.residual2  = ResidualConnection(d_model, dropout)

    def forward(self, x, src_mask=None):
        x = self.residual1(x, lambda x: self.self_attn(x, x, x, src_mask))
        x = self.residual2(x, self.feed_fwd)
        return x


class Encoder(nn.Module):
    """Stack of N encoder layers followed by a final layer norm."""

    def __init__(self, d_model: int, h: int, d_ff: int, N: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, h, d_ff, dropout) for _ in range(N)])
        self.norm   = LayerNorm(d_model)

    def forward(self, x, src_mask=None):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)


# ─────────────────────────────────────────────
# 8. DECODER LAYER  &  FULL DECODER
# ─────────────────────────────────────────────
class DecoderLayer(nn.Module):
    """
    One Transformer decoder block:
      1. Masked Self-Attention (causal)
      2. Cross-Attention with encoder output
      3. FFN
    """

    def __init__(self, d_model: int, h: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn   = MultiHeadAttention(d_model, h, dropout)
        self.cross_attn  = MultiHeadAttention(d_model, h, dropout)
        self.feed_fwd    = FeedForward(d_model, d_ff, dropout)
        self.residuals   = nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range(3)])

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # Masked self-attention (look only at previous tokens)
        x = self.residuals[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # Cross-attention (queries from decoder, keys/values from encoder)
        x = self.residuals[1](x, lambda x: self.cross_attn(x, enc_output, enc_output, src_mask))
        # Feed-forward
        x = self.residuals[2](x, self.feed_fwd)
        return x


class Decoder(nn.Module):
    """Stack of N decoder layers followed by a final layer norm."""

    def __init__(self, d_model: int, h: int, d_ff: int, N: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, h, d_ff, dropout) for _ in range(N)])
        self.norm   = LayerNorm(d_model)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return self.norm(x)


# ─────────────────────────────────────────────
# 9. PROJECTION (LINEAR + SOFTMAX) HEAD
# ─────────────────────────────────────────────
class ProjectionHead(nn.Module):
    """Maps the d_model hidden state to vocabulary logits."""

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x: (batch, seq_len, d_model)  →  (batch, seq_len, vocab_size)
        return self.proj(x)


# ─────────────────────────────────────────────
# 10. MASK UTILITIES
# ─────────────────────────────────────────────
def make_src_mask(src: torch.Tensor, pad_idx: int = 0):
    """
    Padding mask for the source sequence.
    Returns: (batch, 1, 1, src_len)  — True where token is NOT padding.
    """
    return (src != pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt: torch.Tensor, pad_idx: int = 0):
    """
    Combined padding + causal (look-ahead) mask for the target.
    Returns: (batch, 1, tgt_len, tgt_len)
    """
    tgt_len  = tgt.size(1)
    pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)             # (B, 1, 1, T)
    causal   = torch.tril(torch.ones(tgt_len, tgt_len, device=tgt.device)).bool()  # (T, T)
    return pad_mask & causal                                           # (B, 1, T, T)


# ─────────────────────────────────────────────
# 11. FULL TRANSFORMER MODEL
# ─────────────────────────────────────────────
class Transformer(nn.Module):
    """
    Encoder-Decoder Transformer for sequence-to-sequence tasks.

    Args:
        src_vocab_size : source vocabulary size
        tgt_vocab_size : target vocabulary size
        d_model        : embedding / hidden dimension  (default 512)
        h              : number of attention heads      (default 8)
        N              : number of encoder/decoder layers (default 6)
        d_ff           : inner feed-forward dimension   (default 2048)
        max_seq_len    : maximum sequence length        (default 5000)
        dropout        : dropout probability            (default 0.1)
        pad_idx        : padding token index            (default 0)
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        h: int = 8,
        N: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 5000,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.pad_idx = pad_idx

        # Embeddings
        self.src_embed = InputEmbedding(src_vocab_size, d_model)
        self.tgt_embed = InputEmbedding(tgt_vocab_size, d_model)
        self.pos_enc   = PositionalEncoding(d_model, max_seq_len, dropout)

        # Encoder & Decoder stacks
        self.encoder = Encoder(d_model, h, d_ff, N, dropout)
        self.decoder = Decoder(d_model, h, d_ff, N, dropout)

        # Output projection
        self.output_head = ProjectionHead(d_model, tgt_vocab_size)

        # Weight initialisation (Xavier uniform as in the original paper)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor):
        return self.encoder(self.pos_enc(self.src_embed(src)), src_mask)

    def decode(
        self,
        tgt: torch.Tensor,
        enc_output: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ):
        return self.decoder(self.pos_enc(self.tgt_embed(tgt)), enc_output, src_mask, tgt_mask)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        src_mask = make_src_mask(src, self.pad_idx)
        tgt_mask = make_tgt_mask(tgt, self.pad_idx)

        enc_output = self.encode(src, src_mask)
        dec_output = self.decode(tgt, enc_output, src_mask, tgt_mask)
        logits     = self.output_head(dec_output)    # (B, T, tgt_vocab_size)
        return logits


# ─────────────────────────────────────────────
# 12. QUICK SMOKE TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Hyper-parameters (small for demonstration)
    SRC_VOCAB = 8000
    TGT_VOCAB = 8000
    BATCH     = 4
    SRC_LEN   = 20
    TGT_LEN   = 18

    model = Transformer(
        src_vocab_size=SRC_VOCAB,
        tgt_vocab_size=TGT_VOCAB,
        d_model=256,
        h=8,
        N=3,
        d_ff=512,
        dropout=0.1,
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Transformer built  |  trainable params: {total_params:,}")

    # Fake token sequences (batch of 4)
    src = torch.randint(1, SRC_VOCAB, (BATCH, SRC_LEN))   # no PAD tokens here
    tgt = torch.randint(1, TGT_VOCAB, (BATCH, TGT_LEN))

    logits = model(src, tgt)
    print(f"Input  shape : src={tuple(src.shape)}, tgt={tuple(tgt.shape)}")
    print(f"Output shape : {tuple(logits.shape)}")   # (4, 18, 8000)
    print("Forward pass  ✓")
