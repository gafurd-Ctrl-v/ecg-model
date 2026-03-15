"""
train.py — Training loop for ECGAttentionNet.

All hyperparameters and paths come from config.py.
You should never need to edit this file — change config.py instead.

Run:
  python train.py

Metrics printed each epoch:
  Loss        : train_loss, val_loss, and a trend arrow (↓ improving / ↑ worsening / → flat)
  AUC         : per-class + macro (train and val)
  F1          : per-class + macro (val only, at threshold 0.5)
  LR          : current learning rate from the scheduler
  VRAM        : GPU memory used / total (MB)
  Time        : seconds per epoch

End-of-training summary table shows best epoch and all final metrics.
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from tqdm import tqdm

from config import (
    DATA_DIR,
    CHECKPOINT_DIR,
    CHECKPOINT_PATH,
    CLASSES,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    LR_WARMUP_PCT,
    GRAD_CLIP_NORM,
    NUM_WORKERS,
    NUM_CLASSES,
    BASE_CH,
    N_HEADS,
    DROPOUT,
)
from dataset import get_dataloaders
from model   import ECGAttentionNet

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── Device info ────────────────────────────────────────────────────────────────

def print_device_info():
    print(f"Device : {DEVICE}")
    if DEVICE.type == 'cuda':
        props = torch.cuda.get_device_properties(0)
        print(f"GPU    : {props.name}")
        print(f"VRAM   : {props.total_memory / 1e9:.1f} GB")
        print(f"CUDA   : {torch.version.cuda}")


def get_vram_str() -> str:
    """Return current VRAM usage as 'used/total MB' string."""
    if DEVICE.type != 'cuda':
        return 'N/A'
    used  = torch.cuda.memory_allocated()  / 1e6
    total = torch.cuda.get_device_properties(0).total_memory / 1e6
    return f"{used:.0f}/{total:.0f} MB"


# ── Metrics ────────────────────────────────────────────────────────────────────

THRESHOLD = 0.5   # sigmoid threshold for F1 / precision / recall


def compute_aucs(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Per-class AUC + macro average. Skips classes with no positive labels."""
    aucs = {}
    for i, cls in enumerate(CLASSES):
        if y_true[:, i].sum() > 0:
            aucs[cls] = roc_auc_score(y_true[:, i], y_pred[:, i])
    aucs['macro'] = float(np.mean(list(aucs.values())))
    return aucs


def compute_f1s(y_true: np.ndarray, y_pred_prob: np.ndarray) -> dict:
    """Per-class F1 + macro F1 at THRESHOLD."""
    y_bin = (y_pred_prob >= THRESHOLD).astype(int)
    f1s = {}
    for i, cls in enumerate(CLASSES):
        if y_true[:, i].sum() > 0:
            f1s[cls] = f1_score(y_true[:, i], y_bin[:, i], zero_division=0)
    f1s['macro'] = float(np.mean(list(f1s.values())))
    return f1s


def compute_precision_recall(y_true: np.ndarray, y_pred_prob: np.ndarray) -> tuple[dict, dict]:
    """Per-class precision and recall at THRESHOLD."""
    y_bin = (y_pred_prob >= THRESHOLD).astype(int)
    prec, rec = {}, {}
    for i, cls in enumerate(CLASSES):
        if y_true[:, i].sum() > 0:
            prec[cls] = precision_score(y_true[:, i], y_bin[:, i], zero_division=0)
            rec[cls]  = recall_score(   y_true[:, i], y_bin[:, i], zero_division=0)
    prec['macro'] = float(np.mean(list(prec.values())))
    rec['macro']  = float(np.mean(list(rec.values())))
    return prec, rec


def trend_arrow(current: float, previous: float | None, higher_is_better: bool = True) -> str:
    """Return ↓ ↑ → based on whether the metric moved in the desired direction."""
    if previous is None:
        return ' '
    delta = current - previous
    if abs(delta) < 1e-4:
        return '→'
    improving = (delta > 0) if higher_is_better else (delta < 0)
    return '↓' if (not higher_is_better and improving) or (higher_is_better and not improving) else '↑' if (not higher_is_better and not improving) or (higher_is_better and improving) else '→'


def loss_trend(current: float, previous: float | None) -> str:
    return trend_arrow(current, previous, higher_is_better=False)


def auc_trend(current: float, previous: float | None) -> str:
    return trend_arrow(current, previous, higher_is_better=True)


# ── Train / eval passes ───────────────────────────────────────────────────────

def run_epoch(model, loader, criterion,
              optimizer=None, scaler=None, scheduler=None,
              training: bool = False) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Unified train/eval pass.

    Returns:
        (avg_loss, all_labels, all_pred_probs)
        shapes: scalar, (N, 5), (N, 5)
    """
    model.train() if training else model.eval()
    total_loss  = 0.0
    all_preds   = []
    all_labels  = []

    ctx = torch.enable_grad() if training else torch.no_grad()

    with ctx:
        for signals, labels in loader:
            signals = signals.to(DEVICE, dtype=torch.float32, non_blocking=True)
            labels  = labels.to(DEVICE,  dtype=torch.float32, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=DEVICE.type):
                logits = model(signals)
                loss   = criterion(logits, labels)

            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            total_loss += loss.item()
            all_preds.append(torch.sigmoid(logits).float().detach().cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    return (
        total_loss / len(loader),
        np.vstack(all_labels),
        np.vstack(all_preds),
    )


# ── Pretty printing ────────────────────────────────────────────────────────────

def print_epoch_header():
    print()
    print(f"{'Ep':>4}  {'T-Loss':>7}  {'V-Loss':>7}  "
          f"{'T-AUC':>6}  {'V-AUC':>6}  {'V-F1':>6}  "
          f"{'LR':>8}  {'VRAM':>12}  {'Time':>6}")
    print("─" * 90)


def print_epoch_row(epoch, epochs,
                    t_loss, v_loss,
                    t_aucs, v_aucs, v_f1s,
                    lr, vram, elapsed,
                    prev_v_loss=None, prev_v_auc=None):
    l_arrow   = loss_trend(v_loss,         prev_v_loss)
    auc_arrow = auc_trend(v_aucs['macro'], prev_v_auc)
    print(
        f"{epoch:>3}/{epochs:<3}  "
        f"{t_loss:>7.4f}  {v_loss:>6.4f}{l_arrow}  "
        f"{t_aucs['macro']:>6.3f}  {v_aucs['macro']:>5.3f}{auc_arrow}  "
        f"{v_f1s['macro']:>6.3f}  "
        f"{lr:>8.2e}  {vram:>12}  {elapsed:>5.1f}s"
    )


def print_per_class(v_aucs, v_f1s, v_prec, v_rec):
    print()
    print(f"  {'Class':<6}  {'AUC':>6}  {'F1':>6}  {'Prec':>6}  {'Recall':>6}")
    print("  " + "─" * 38)
    for cls in CLASSES:
        auc  = v_aucs.get(cls, float('nan'))
        f1   = v_f1s.get(cls,  float('nan'))
        p    = v_prec.get(cls, float('nan'))
        r    = v_rec.get(cls,  float('nan'))
        print(f"  {cls:<6}  {auc:>6.3f}  {f1:>6.3f}  {p:>6.3f}  {r:>6.3f}")
    print(f"  {'macro':<6}  {v_aucs['macro']:>6.3f}  {v_f1s['macro']:>6.3f}  "
          f"{v_prec['macro']:>6.3f}  {v_rec['macro']:>6.3f}")


def print_summary(history: list[dict], best_epoch: int):
    print()
    print("═" * 90)
    print("TRAINING COMPLETE — SUMMARY")
    print("═" * 90)
    best = history[best_epoch - 1]
    print(f"  Best epoch      : {best_epoch}")
    print(f"  Best val AUC    : {best['v_auc']:.4f}")
    print(f"  Best val F1     : {best['v_f1']:.4f}")
    print(f"  Val loss at best: {best['v_loss']:.4f}")
    print()
    print("  Epoch-by-epoch val macro-AUC:")
    bar_max = max(h['v_auc'] for h in history)
    for h in history:
        filled = int((h['v_auc'] / bar_max) * 30)
        marker = '★' if h['epoch'] == best_epoch else ' '
        print(f"  Ep {h['epoch']:>2} {marker} {'█' * filled}{'░' * (30 - filled)}  {h['v_auc']:.3f}")
    print("═" * 90)


# ── Main training loop ─────────────────────────────────────────────────────────

def train(data_dir:       str   = DATA_DIR,
          epochs:         int   = EPOCHS,
          batch_size:     int   = BATCH_SIZE,
          lr:             float = LEARNING_RATE,
          checkpoint_dir: str   = CHECKPOINT_DIR,
          num_workers:    int   = NUM_WORKERS):

    print_device_info()
    os.makedirs(checkpoint_dir, exist_ok=True)

    print('\nLoading PTB-XL...')
    train_loader, val_loader, _ = get_dataloaders(
        data_dir    = data_dir,
        batch_size  = batch_size,
        num_workers = num_workers,
    )

    model = ECGAttentionNet(
        num_classes = NUM_CLASSES,
        base_ch     = BASE_CH,
        nhead       = N_HEADS,
        dropout     = DROPOUT,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Parameters: {n_params:,}')

    criterion   = nn.BCEWithLogitsLoss()
    optimizer   = AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    total_steps = epochs * len(train_loader)
    scheduler   = OneCycleLR(
        optimizer,
        max_lr          = lr,
        total_steps     = total_steps,
        pct_start       = LR_WARMUP_PCT,
        anneal_strategy = 'cos',
    )
    scaler    = GradScaler(device=DEVICE.type)
    save_path = os.path.join(checkpoint_dir, os.path.basename(CHECKPOINT_PATH))

    print(f'\nTraining {epochs} epochs  batch={batch_size}  lr={lr}')
    print_epoch_header()

    best_val_auc  = 0.0
    best_epoch    = 1
    history       = []
    prev_v_loss   = None
    prev_v_auc    = None

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # ── Train pass ───────────────────────────────────────────────────────
        t_loss, t_labels, t_preds = run_epoch(
            model, train_loader, criterion,
            optimizer=optimizer, scaler=scaler, scheduler=scheduler,
            training=True
        )

        # ── Val pass ─────────────────────────────────────────────────────────
        v_loss, v_labels, v_preds = run_epoch(
            model, val_loader, criterion, training=False
        )

        elapsed = time.time() - t0

        # ── Compute all metrics ───────────────────────────────────────────────
        t_aucs         = compute_aucs(t_labels, t_preds)
        v_aucs         = compute_aucs(v_labels, v_preds)
        v_f1s          = compute_f1s(v_labels, v_preds)
        v_prec, v_rec  = compute_precision_recall(v_labels, v_preds)
        current_lr     = scheduler.get_last_lr()[0]
        vram           = get_vram_str()

        # ── Print summary row ─────────────────────────────────────────────────
        print_epoch_row(
            epoch, epochs,
            t_loss, v_loss,
            t_aucs, v_aucs, v_f1s,
            current_lr, vram, elapsed,
            prev_v_loss, prev_v_auc
        )

        # ── Print per-class breakdown every 5 epochs and at the last epoch ───
        if epoch % 5 == 0 or epoch == epochs:
            print_per_class(v_aucs, v_f1s, v_prec, v_rec)
            print()

        # ── Save best checkpoint ──────────────────────────────────────────────
        if v_aucs['macro'] > best_val_auc:
            best_val_auc = v_aucs['macro']
            best_epoch   = epoch
            torch.save({
                'epoch':            epoch,
                'model_state_dict': model.state_dict(),
                'val_auc':          best_val_auc,
                'val_aucs':         v_aucs,
                'val_f1s':          v_f1s,
                'config': {
                    'BASE_CH':    BASE_CH,
                    'N_HEADS':    N_HEADS,
                    'DROPOUT':    DROPOUT,
                    'BATCH_SIZE': batch_size,
                    'EPOCHS':     epochs,
                    'LR':         lr,
                }
            }, save_path)
            print(f"  ✓ New best  macro-AUC={best_val_auc:.4f}  macro-F1={v_f1s['macro']:.4f}"
                  f"  → {save_path}")

        # ── Store history for end summary ─────────────────────────────────────
        history.append({
            'epoch':  epoch,
            't_loss': t_loss,
            'v_loss': v_loss,
            't_auc':  t_aucs['macro'],
            'v_auc':  v_aucs['macro'],
            'v_f1':   v_f1s['macro'],
        })

        prev_v_loss = v_loss
        prev_v_auc  = v_aucs['macro']

    print_summary(history, best_epoch)
    return model


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train ECGAttentionNet. Defaults come from config.py.')
    parser.add_argument('--data_dir',       type=str,   default=DATA_DIR)
    parser.add_argument('--epochs',         type=int,   default=EPOCHS)
    parser.add_argument('--batch_size',     type=int,   default=BATCH_SIZE,
                        help='Lower to 8 if CUDA OOM on RTX 3050')
    parser.add_argument('--lr',             type=float, default=LEARNING_RATE)
    parser.add_argument('--checkpoint_dir', type=str,   default=CHECKPOINT_DIR)
    parser.add_argument('--num_workers',    type=int,   default=NUM_WORKERS)
    args = parser.parse_args()

    train(
        data_dir       = args.data_dir,
        epochs         = args.epochs,
        batch_size     = args.batch_size,
        lr             = args.lr,
        checkpoint_dir = args.checkpoint_dir,
        num_workers    = args.num_workers,
    )