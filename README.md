# Transformer

Experiments with transformers

1. The Entry Point: Embeddings and Position
Computers don't understand words; they understand vectors.
- Input Embedding: This converts a token ID into a vector of size [d_model] (usually 512).
The code multiplies these vectors by [d_model]^1/2 (sqrt) to ensure that the variance of the embeddings matches the variance of the positional encodings.
- Positional Encoding: Since the Transformer processes all tokens at once, it has no idea where a word sits in a sentence. This component adds a fixed "signal" to the embedding using sine and cosine waves
- Dropout: A 10% dropout is applied immediately after adding the position signal to prevent the model from over-relying on specific dimensions.

2. The Engine: Multi-Head Attention (MHA)
This is where the model "looks" at other words to understand context.
- Linear Projections: The input is projected into three different spaces: Query (Q), Key (K), and Value (V).
The "Split": If [d_model] is 512 and there are 8 heads (h), the code reshapes the tensor so that each head handles a vector of size 64 (d_k).
This allows the model to attend to different types of information (e.g., one head for grammar, one for meaning) simultaneously.
- Scaled Dot-Product: The core math is Softmax((QK^T)/([d_k]^1/2)).
-QK^T calculates a score for how much every word should care about every other word.
-Dividing by [d_k]^1/2 prevents the scores from getting too large, which would "kill" the gradients during training.
-Masking: The code uses masked_fill to set scores for padding or "future" tokens to negative infinity, effectively making their influence zero after the softmax.

