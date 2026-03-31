"""
train.py — Complete training pipeline for the Transformer

Features:
  • Teacher forcing with correct label shifting (<sos> y1 y2 → y1 y2 <eos>)
  • Label smoothing cross-entropy loss
  • Warmup + inverse-sqrt LR schedule (matches original paper)
  • Gradient clipping
  • Checkpoint save / resume
  • Per-epoch validation with perplexity
"""

import os
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from typing import Optional


# ──────────────────────────────────────────────
# LABEL SMOOTHING LOSS
# ──────────────────────────────────────────────

class LabelSmoothingLoss(nn.Module):
    """
    Cross-entropy with label smoothing.
    Smoothing ε distributes probability mass away from the gold label
    to prevent overconfident predictions.
    """

    def __init__(self, vocab_size: int, pad_idx: int = 0, smoothing: float = 0.1):
        super().__init__()
        self.pad_idx    = pad_idx
        self.smoothing  = smoothing
        self.vocab_size = vocab_size
        self.criterion  = nn.KLDivLoss(reduction="sum")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        logits  : (B*T, vocab_size)  — raw model output
        targets : (B*T,)             — gold token IDs
        """
        log_probs = F.log_softmax(logits, dim=-1)

        # Build smooth target distribution
        with torch.no_grad():
            smooth_dist = torch.full_like(log_probs, self.smoothing / (self.vocab_size - 2))
            smooth_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
            smooth_dist[:, self.pad_idx] = 0.0            # zero out padding

        # Mask padding positions
        non_pad = targets != self.pad_idx
        loss = self.criterion(log_probs[non_pad], smooth_dist[non_pad])
        return loss / non_pad.sum().float()


# ──────────────────────────────────────────────
# WARMUP LR SCHEDULER  (paper §5.3)
# ──────────────────────────────────────────────

class TransformerLRScheduler:
    """
    lrate = d_model^(-0.5) · min(step^(-0.5), step · warmup^(-1.5))
    """

    def __init__(self, optimizer, d_model: int, warmup_steps: int = 4000):
        self.optimizer    = optimizer
        self.d_model      = d_model
        self.warmup_steps = warmup_steps
        self._step        = 0

    def step(self):
        self._step += 1
        lr = self._get_lr()
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def _get_lr(self) -> float:
        s, w = self._step, self.warmup_steps
        return self.d_model ** (-0.5) * min(s ** (-0.5), s * w ** (-1.5))

    @property
    def current_lr(self) -> float:
        return self._get_lr()


# ──────────────────────────────────────────────
# HELPER: shift targets for teacher forcing
# ──────────────────────────────────────────────

def make_teacher_forcing_pair(tgt_full: torch.Tensor, sos_idx: int, eos_idx: int):
    """
    Given full target sequence (including <sos> and <eos>):
      tgt_input  = <sos> y1 y2 … yN          (feed to decoder)
      tgt_output = y1   y2 … yN <eos>        (predict this)

    Args:
        tgt_full : (B, T+2) — includes <sos> and <eos>
    Returns:
        tgt_input  : (B, T+1)
        tgt_output : (B, T+1)
    """
    tgt_input  = tgt_full[:, :-1]   # drop last token  (<eos>)
    tgt_output = tgt_full[:, 1:]    # drop first token (<sos>)
    return tgt_input, tgt_output


# ──────────────────────────────────────────────
# TRAIN ONE EPOCH
# ──────────────────────────────────────────────

def train_epoch(
    model,
    loader:     DataLoader,
    criterion:  LabelSmoothingLoss,
    optimizer:  torch.optim.Optimizer,
    scheduler:  TransformerLRScheduler,
    device:     str,
    sos_idx:    int,
    eos_idx:    int,
    clip_grad:  float = 1.0,
    log_every:  int   = 50,
):
    model.train()
    total_loss  = 0.0
    total_toks  = 0
    start       = time.time()

    for batch_idx, batch in enumerate(loader):
        # ── Unpack batch ─────────────────────────────
        # Expects batch to be a dict with "src" and "tgt" tensors
        src      = batch["src"].to(device)           # (B, S)
        tgt_full = batch["tgt"].to(device)           # (B, T)  includes <sos>/<eos>

        tgt_input, tgt_output = make_teacher_forcing_pair(tgt_full, sos_idx, eos_idx)

        # ── Forward pass ─────────────────────────────
        logits = model(src, tgt_input)               # (B, T, vocab)

        B, T, V = logits.size()
        loss = criterion(
            logits.reshape(B * T, V),
            tgt_output.reshape(B * T),
        )

        # ── Backward ─────────────────────────────────
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        scheduler.step()

        # ── Logging ──────────────────────────────────
        non_pad     = (tgt_output != 0).sum().item()
        total_loss += loss.item() * non_pad
        total_toks += non_pad

        if (batch_idx + 1) % log_every == 0:
            elapsed  = time.time() - start
            avg_loss = total_loss / total_toks
            ppl      = math.exp(min(avg_loss, 100))
            print(
                f"  step {batch_idx+1:>5} | loss {avg_loss:.4f} | ppl {ppl:.2f} "
                f"| lr {scheduler.current_lr:.2e} | {elapsed:.1f}s"
            )

    avg_loss = total_loss / max(total_toks, 1)
    return avg_loss, math.exp(min(avg_loss, 100))


# ──────────────────────────────────────────────
# VALIDATION
# ──────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, criterion, device, sos_idx, eos_idx):
    model.eval()
    total_loss = 0.0
    total_toks = 0

    for batch in loader:
        src      = batch["src"].to(device)
        tgt_full = batch["tgt"].to(device)
        tgt_input, tgt_output = make_teacher_forcing_pair(tgt_full, sos_idx, eos_idx)

        logits = model(src, tgt_input)
        B, T, V = logits.size()
        loss = criterion(logits.reshape(B * T, V), tgt_output.reshape(B * T))

        non_pad     = (tgt_output != 0).sum().item()
        total_loss += loss.item() * non_pad
        total_toks += non_pad

    avg_loss = total_loss / max(total_toks, 1)
    return avg_loss, math.exp(min(avg_loss, 100))


# ──────────────────────────────────────────────
# CHECKPOINT UTILITIES
# ──────────────────────────────────────────────

def save_checkpoint(path: str, model, optimizer, scheduler, epoch: int, val_loss: float):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "epoch":      epoch,
            "val_loss":   val_loss,
            "model":      model.state_dict(),
            "optimizer":  optimizer.state_dict(),
            "scheduler_step": scheduler._step,
        },
        path,
    )
    print(f"  ✓ Checkpoint saved → {path}")


def load_checkpoint(path: str, model, optimizer=None, scheduler=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler:
        scheduler._step = ckpt.get("scheduler_step", 0)
    print(f"  ✓ Resumed from epoch {ckpt['epoch']}  (val_loss={ckpt['val_loss']:.4f})")
    return ckpt["epoch"]


# ──────────────────────────────────────────────
# MAIN TRAINING LOOP
# ──────────────────────────────────────────────

def train(
    model,
    train_loader:   DataLoader,
    val_loader:     DataLoader,
    sos_idx:        int,
    eos_idx:        int,
    epochs:         int   = 20,
    d_model:        int   = 512,
    warmup_steps:   int   = 4000,
    label_smoothing:float = 0.1,
    clip_grad:      float = 1.0,
    log_every:      int   = 50,
    checkpoint_dir: str   = "checkpoints",
    resume_from:    Optional[str] = None,
    device:         Optional[str] = None,
):
    """
    Full training loop.

    Usage example:
        from model import Transformer
        from train import train

        model = Transformer(src_vocab=8000, tgt_vocab=8000)
        train(model, train_loader, val_loader, sos_idx=1, eos_idx=2)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    print(f"Training on: {device.upper()}")
    print(f"Parameters : {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    vocab_size = model.output_head.weight.size(0)
    criterion  = LabelSmoothingLoss(vocab_size, pad_idx=model.pad_idx, smoothing=label_smoothing)
    optimizer  = Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler  = TransformerLRScheduler(optimizer, d_model=d_model, warmup_steps=warmup_steps)

    start_epoch = 0
    if resume_from and os.path.exists(resume_from):
        start_epoch = load_checkpoint(resume_from, model, optimizer, scheduler, device)

    best_val_loss = float("inf")

    for epoch in range(start_epoch, epochs):
        print(f"\n{'─'*55}")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"{'─'*55}")

        train_loss, train_ppl = train_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            device, sos_idx, eos_idx, clip_grad, log_every,
        )
        val_loss, val_ppl = evaluate(model, val_loader, criterion, device, sos_idx, eos_idx)

        print(f"\n  Train → loss {train_loss:.4f}  ppl {train_ppl:.2f}")
        print(f"  Val   → loss {val_loss:.4f}  ppl {val_ppl:.2f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                f"{checkpoint_dir}/best.pt", model, optimizer, scheduler, epoch + 1, val_loss
            )

        # Save periodic checkpoint
        save_checkpoint(
            f"{checkpoint_dir}/epoch_{epoch+1}.pt", model, optimizer, scheduler, epoch + 1, val_loss
        )

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
