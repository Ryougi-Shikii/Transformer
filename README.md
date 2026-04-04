# Transformer Model (`model.py`) — From Tokens to Probabilities

This file implements a **Pre-LayerNorm Transformer**, the modern standard for stable deep Transformer training.

The goal: transform a sequence of token IDs into a probability distribution over a vocabulary.

---

## 1. Input Stage: Embedding + Position

### Input Embedding
- Converts token IDs into dense vectors of size `d_model` (e.g., 512).
- Scaled by:

\[
\sqrt{d_{model}}
\]

- Ensures embedding variance aligns with positional encoding.

---

### Positional Encoding
Adds positional information using sine/cosine functions:

\[
PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})
\]

\[
PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})
\]

- Enables the model to understand token order.
- No recurrence or convolution is used.

---

### Dropout
- Applied after embedding + positional encoding.
- Typical value: `0.1`
- Prevents over-reliance on specific dimensions.

---

## 2. Core Mechanism: Multi-Head Attention (MHA)

### Linear Projections
Input is projected into:
- Query (Q)
- Key (K)
- Value (V)

---

### Multi-Head Split
If:
- `d_model = 512`
- `heads = 8`

Then:
- Each head processes `d_k = 64`

This allows parallel attention over different representation subspaces.

---

### Scaled Dot-Product Attention

\[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

- \(QK^T\): similarity scores
- Scaling prevents gradient issues
- Softmax produces attention weights

---

### Masking
- Implemented via `masked_fill`
- Sets invalid positions to `-∞`
- Ensures:
  - No attention to padding
  - No "future token" leakage in decoder

---

## 3. Feed-Forward Network (FFN)

Operates independently on each token.

### Structure:
1. **Expansion**:
   \[
   512 \rightarrow 2048
   \]

2. **Activation**:
   - GELU (smooth, better gradient flow)

3. **Projection Back**:
   \[
   2048 \rightarrow 512
   \]

---

## 4. Pre-LayerNorm Architecture

Each sublayer follows:

\[
x + \text{Dropout}(\text{Sublayer}(\text{LayerNorm}(x)))
\]

### Why Pre-LN?
- Stabilizes training
- Prevents vanishing/exploding gradients
- Enables deeper stacks

---

## 5. Decoder Cross-Attention

Decoder contains an additional attention layer:

### Mechanism:
- Query → Decoder
- Key, Value → Encoder

### Purpose:
- Allows decoder to attend to source sequence
- Enables alignment (e.g., translation)

---

## 6. Output Stage

### Linear Projection
- Maps `d_model → vocab_size`
- Produces logits for each token

---

### Weight Tying

```python
self.output_head.weight = self.tgt_embed.embedding.weight