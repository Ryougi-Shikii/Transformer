"""
inference.py — Greedy decoding & Beam Search for the Transformer
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Optional


# ──────────────────────────────────────────────
# GREEDY DECODING
# ──────────────────────────────────────────────

@torch.no_grad()
def greedy_decode(
    model,
    src:        torch.Tensor,
    sos_idx:    int,
    eos_idx:    int,
    max_len:    int = 100,
    device:     str = "cpu",
) -> List[int]:
    """
    Autoregressive greedy decoding (picks highest-prob token each step).

    Args:
        model   : trained Transformer
        src     : (1, S) source token tensor (already on device)
        sos_idx : start-of-sequence token index
        eos_idx : end-of-sequence token index
        max_len : maximum generation length
        device  : 'cpu' or 'cuda'

    Returns:
        List of generated token IDs (without <sos>)
    """
    model.eval()
    src = src.to(device)

    enc_out, src_mask = model.encode(src)

    # Start with <sos>
    generated = torch.tensor([[sos_idx]], dtype=torch.long, device=device)

    for _ in range(max_len):
        dec_out = model.decode(generated, enc_out, src_mask)    # (1, T, d_model)
        logits  = model.output_head(dec_out[:, -1, :])          # (1, vocab)
        next_id = logits.argmax(dim=-1).item()

        generated = torch.cat(
            [generated, torch.tensor([[next_id]], device=device)], dim=1
        )

        if next_id == eos_idx:
            break

    return generated[0, 1:].tolist()   # strip <sos>


# ──────────────────────────────────────────────
# BEAM SEARCH
# ──────────────────────────────────────────────

@dataclass
class BeamHypothesis:
    token_ids:  List[int]
    log_prob:   float
    is_done:    bool = False

    def __len__(self):
        return len(self.token_ids)


@torch.no_grad()
def beam_search(
    model,
    src:           torch.Tensor,
    sos_idx:       int,
    eos_idx:       int,
    beam_size:     int   = 4,
    max_len:       int   = 100,
    length_penalty:float = 0.7,   # α in ((5+|hyp|)/(5+1))^α
    device:        str   = "cpu",
) -> List[int]:
    """
    Beam search decoding.

    Args:
        model          : trained Transformer
        src            : (1, S) source token tensor
        sos_idx        : <sos> token index
        eos_idx        : <eos> token index
        beam_size      : number of beams
        max_len        : max generation length
        length_penalty : higher → favour longer sequences
        device         : 'cpu' or 'cuda'

    Returns:
        Best sequence as a list of token IDs (without <sos>)
    """
    model.eval()
    src = src.to(device)

    enc_out, src_mask = model.encode(src)

    # Expand encoder output for all beams: (beam_size, S, d_model)
    enc_out  = enc_out.expand(beam_size, -1, -1)
    src_mask = src_mask.expand(beam_size, -1, -1, -1)

    # Initialize beams
    beams: List[BeamHypothesis] = [
        BeamHypothesis(token_ids=[sos_idx], log_prob=0.0)
    ]
    completed: List[BeamHypothesis] = []

    for step in range(max_len):
        active = [b for b in beams if not b.is_done]
        if not active:
            break

        # Build current sequence tensor  (active_beams, step+1)
        tgt = torch.tensor(
            [b.token_ids for b in active], dtype=torch.long, device=device
        )
        n   = tgt.size(0)

        dec_out = model.decode(tgt, enc_out[:n], src_mask[:n])   # (n, T, d_model)
        logits  = model.output_head(dec_out[:, -1, :])           # (n, vocab)
        log_probs = F.log_softmax(logits, dim=-1)                # (n, vocab)

        # Expand each beam by top-k tokens
        new_beams: List[BeamHypothesis] = []
        top_log_probs, top_ids = log_probs.topk(beam_size, dim=-1)  # (n, beam)

        for i, beam in enumerate(active):
            for j in range(beam_size):
                token   = top_ids[i, j].item()
                lp      = beam.log_prob + top_log_probs[i, j].item()
                new_seq = beam.token_ids + [token]

                hyp = BeamHypothesis(token_ids=new_seq, log_prob=lp)
                if token == eos_idx:
                    hyp.is_done = True
                    completed.append(hyp)
                else:
                    new_beams.append(hyp)

        # Keep top beam_size active beams (by length-penalised score)
        def score(h: BeamHypothesis) -> float:
            lp = ((5 + len(h)) / 6) ** length_penalty
            return h.log_prob / lp

        new_beams.sort(key=score, reverse=True)
        beams = new_beams[:beam_size]

    # If nothing completed, use best active beam
    if not completed:
        completed = beams

    best = max(completed, key=lambda h: h.log_prob / (len(h) ** length_penalty))
    return best.token_ids[1:]   # strip <sos>
