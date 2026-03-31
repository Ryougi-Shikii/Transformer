"""
dataset.py — Dataset utilities + full example usage

Provides:
  • TranslationDataset   — generic seq2seq dataset wrapper
  • Vocabulary           — token ↔ index mapping with special tokens
  • collate_fn           — dynamic padding for DataLoader batching
  • run_example()        — end-to-end demo you can copy and adapt
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple, Dict, Optional
from collections import Counter


# ──────────────────────────────────────────────
# VOCABULARY
# ──────────────────────────────────────────────

class Vocabulary:
    """
    Simple token ↔ index vocabulary.

    Special tokens:
      <pad> = 0   (always index 0)
      <sos> = 1
      <eos> = 2
      <unk> = 3
    """

    PAD, SOS, EOS, UNK = 0, 1, 2, 3

    def __init__(self):
        self.token2idx: Dict[str, int] = {
            "<pad>": self.PAD,
            "<sos>": self.SOS,
            "<eos>": self.EOS,
            "<unk>": self.UNK,
        }
        self.idx2token: Dict[int, str] = {v: k for k, v in self.token2idx.items()}

    def build(self, sentences: List[List[str]], min_freq: int = 1):
        """Build vocabulary from tokenized sentences."""
        counter = Counter(tok for sent in sentences for tok in sent)
        for token, freq in sorted(counter.items()):
            if freq >= min_freq and token not in self.token2idx:
                idx = len(self.token2idx)
                self.token2idx[token] = idx
                self.idx2token[idx]   = token

    def encode(self, tokens: List[str]) -> List[int]:
        return (
            [self.SOS]
            + [self.token2idx.get(t, self.UNK) for t in tokens]
            + [self.EOS]
        )

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        special = {self.PAD, self.SOS, self.EOS, self.UNK}
        tokens  = [
            self.idx2token.get(i, "<unk>")
            for i in ids
            if not (skip_special and i in special)
        ]
        return " ".join(tokens)

    def __len__(self) -> int:
        return len(self.token2idx)


# ──────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────

class TranslationDataset(Dataset):
    """
    Generic seq2seq dataset.

    Args:
        src_sentences : list of pre-tokenized source sentences
        tgt_sentences : list of pre-tokenized target sentences
        src_vocab     : Vocabulary for source
        tgt_vocab     : Vocabulary for target
        max_src_len   : drop samples longer than this
        max_tgt_len   : drop samples longer than this
    """

    def __init__(
        self,
        src_sentences:  List[List[str]],
        tgt_sentences:  List[List[str]],
        src_vocab:      Vocabulary,
        tgt_vocab:      Vocabulary,
        max_src_len:    int = 128,
        max_tgt_len:    int = 128,
    ):
        assert len(src_sentences) == len(tgt_sentences)

        self.pairs = []
        for src_toks, tgt_toks in zip(src_sentences, tgt_sentences):
            if len(src_toks) > max_src_len or len(tgt_toks) > max_tgt_len:
                continue
            src_ids = src_vocab.encode(src_toks)
            tgt_ids = tgt_vocab.encode(tgt_toks)
            self.pairs.append((
                torch.tensor(src_ids, dtype=torch.long),
                torch.tensor(tgt_ids, dtype=torch.long),
            ))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        return {"src": src, "tgt": tgt}


def collate_fn(batch, pad_idx: int = 0):
    """
    Pads variable-length sequences in a batch to the same length.
    Compatible with DataLoader's collate_fn argument.
    """
    src_list = [item["src"] for item in batch]
    tgt_list = [item["tgt"] for item in batch]

    src_padded = pad_sequence(src_list, batch_first=True, padding_value=pad_idx)
    tgt_padded = pad_sequence(tgt_list, batch_first=True, padding_value=pad_idx)

    return {"src": src_padded, "tgt": tgt_padded}


# ──────────────────────────────────────────────
# END-TO-END EXAMPLE
# ──────────────────────────────────────────────

def run_example():
    """
    Minimal runnable demo.  Replace the toy sentences with your real data.
    """
    from functools import partial
    from model     import Transformer
    from train     import train
    from inference import greedy_decode, beam_search

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ── 1. Toy data (replace with your real corpus) ───────────────────────
    src_sents = [
        ["hello", "world"],
        ["how", "are", "you"],
        ["good", "morning"],
        ["the", "cat", "sat", "on", "the", "mat"],
    ]
    tgt_sents = [
        ["hola", "mundo"],
        ["como", "estas", "tu"],
        ["buenos", "dias"],
        ["el", "gato", "se", "sento", "en", "el", "tapete"],
    ]

    # ── 2. Build vocabularies ─────────────────────────────────────────────
    src_vocab = Vocabulary()
    tgt_vocab = Vocabulary()
    src_vocab.build(src_sents)
    tgt_vocab.build(tgt_sents)

    print(f"Source vocab size : {len(src_vocab)}")
    print(f"Target vocab size : {len(tgt_vocab)}")

    # ── 3. Build dataset + DataLoader ─────────────────────────────────────
    dataset = TranslationDataset(src_sents, tgt_sents, src_vocab, tgt_vocab)
    loader  = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=partial(collate_fn, pad_idx=Vocabulary.PAD),
    )

    # ── 4. Build model ────────────────────────────────────────────────────
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model   = 64,       # small for demo
        h         = 4,
        N         = 2,
        d_ff      = 128,
        dropout   = 0.1,
        pad_idx   = Vocabulary.PAD,
        tie_weights=True,
    ).to(DEVICE)

    # ── 5. Train ──────────────────────────────────────────────────────────
    train(
        model,
        train_loader    = loader,
        val_loader      = loader,     # use a real val split in production
        sos_idx         = Vocabulary.SOS,
        eos_idx         = Vocabulary.EOS,
        epochs          = 5,
        d_model         = 64,
        warmup_steps    = 50,
        label_smoothing = 0.1,
        checkpoint_dir  = "checkpoints",
        device          = DEVICE,
    )

    # ── 6. Inference ──────────────────────────────────────────────────────
    model.eval()
    test_src  = torch.tensor(
        [src_vocab.encode(["hello", "world"])], dtype=torch.long, device=DEVICE
    )

    greedy_ids  = greedy_decode(model, test_src, Vocabulary.SOS, Vocabulary.EOS, device=DEVICE)
    beam_ids    = beam_search(model, test_src, Vocabulary.SOS, Vocabulary.EOS, beam_size=3, device=DEVICE)

    print("\n─── Inference ───────────────────────────────────")
    print(f"  Input  : hello world")
    print(f"  Greedy : {tgt_vocab.decode(greedy_ids)}")
    print(f"  Beam   : {tgt_vocab.decode(beam_ids)}")


if __name__ == "__main__":
    run_example()
