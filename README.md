# Transformer

Experiments with transformers

1. The Entry Point: Embeddings and Position
Computers don't understand words; they understand vectors.
- Input Embedding: This converts a token ID into a vector of size [d_model] (usually 512).
The code multiplies these vectors by sqrt{d_{model}} to ensure that the variance of the embeddings matches the variance of the positional encodings.
- Positional Encoding: Since the Transformer processes all tokens at once, it has no idea where a word sits in a sentence. This component adds a fixed "signal" to the embedding using sine and cosine waves
- Dropout: A 10% dropout is applied immediately after adding the position signal to prevent the model from over-relying on specific dimensions.

