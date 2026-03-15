"""
evaluate.py — Rigorous test-set evaluation of the trained ECGAttentionNet.

Runs the best checkpoint against the 1,711 held-out test records and
produces a full report including:
  - Per-class AUC, F1, Precision, Recall
  - Confusion matrix per class
  - Confidence distribution (how sure is the model when it's right vs wrong)
  - Worst predictions (highest-confidence wrong calls)
  - Comparison against the val metrics saved in the checkpoint

Run:
  python evaluate.py

This script uses only the test fold (fold 10) — records the model has
never seen during training or validation.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, roc_curve, confusion_matrix,
    average_precision_score,
)

from config import (
    DATA_DIR, CHECKPOINT_PATH, CLASSES, CLASS_NAMES,
    NUM_CLASSES, BASE_CH, N_HEADS, DROPOUT, BATCH_SIZE, NUM_WORKERS,
)
from dataset import get_dataloaders, load_ptbxl, PTBXLDataset
from model   import ECGAttentionNet
from torch.utils.data import DataLoader

DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
THRESHOLD = 0.5


# ── Load model ─────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str = CHECKPOINT_PATH) -> tuple:
    """Load checkpoint and return (model, checkpoint_dict)."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Run train.py first."
        )
    ckpt  = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    model = ECGAttentionNet(
        num_classes = NUM_CLASSES,
        base_ch     = BASE_CH,
        nhead       = N_HEADS,
        dropout     = DROPOUT,
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(DEVICE).eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}  "
          f"(val macro-AUC={ckpt['val_auc']:.4f})")
    return model, ckpt


# ── Inference on test set ──────────────────────────────────────────────────────

@torch.no_grad()
def run_test(model, data_dir: str = DATA_DIR) -> tuple:
    """Run model on test fold. Returns (labels, probs, record_ids).

    Returns:
        y_true : (N, 5) int array
        y_prob : (N, 5) float array  — sigmoid probabilities
        ids    : list of ecg_id strings
    """
    _, _, test_df = load_ptbxl(data_dir)
    test_ds = PTBXLDataset(test_df, data_dir)
    loader  = DataLoader(
        test_ds,
        batch_size  = BATCH_SIZE,
        shuffle     = False,
        num_workers = NUM_WORKERS,
        pin_memory  = True,
    )

    all_probs, all_labels = [], []

    for signals, labels in tqdm(loader, desc='Testing', ncols=80):
        signals = signals.to(DEVICE, dtype=torch.float32, non_blocking=True)
        with autocast(device_type=DEVICE.type):
            logits = model(signals)
        all_probs.append(torch.sigmoid(logits).float().cpu().numpy())
        all_labels.append(labels.numpy())

    y_prob  = np.vstack(all_probs)
    y_true  = np.vstack(all_labels).astype(int)
    ids     = test_df.index.astype(str).tolist()
    return y_true, y_prob, ids


# ── Metric helpers ─────────────────────────────────────────────────────────────

def compute_all_metrics(y_true: np.ndarray,
                         y_prob: np.ndarray) -> dict:
    """Compute AUC, AP, F1, Precision, Recall per class and macro."""
    y_bin = (y_prob >= THRESHOLD).astype(int)
    metrics = {}
    for i, cls in enumerate(CLASSES):
        if y_true[:, i].sum() == 0:
            continue
        metrics[cls] = {
            'auc':       roc_auc_score(y_true[:, i], y_prob[:, i]),
            'ap':        average_precision_score(y_true[:, i], y_prob[:, i]),
            'f1':        f1_score(y_true[:, i], y_bin[:, i], zero_division=0),
            'precision': precision_score(y_true[:, i], y_bin[:, i], zero_division=0),
            'recall':    recall_score(y_true[:, i], y_bin[:, i], zero_division=0),
            'support':   int(y_true[:, i].sum()),
        }

    # Macro averages
    for metric in ['auc', 'ap', 'f1', 'precision', 'recall']:
        metrics['macro_' + metric] = float(
            np.mean([metrics[c][metric] for c in CLASSES if c in metrics])
        )
    return metrics


def compute_confusion(y_true: np.ndarray,
                       y_prob: np.ndarray) -> dict:
    """Per-class 2×2 confusion matrix (binary: class vs rest)."""
    y_bin = (y_prob >= THRESHOLD).astype(int)
    cms   = {}
    for i, cls in enumerate(CLASSES):
        cm = confusion_matrix(y_true[:, i], y_bin[:, i], labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        cms[cls] = {'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp)}
    return cms


def find_worst_predictions(y_true: np.ndarray,
                            y_prob: np.ndarray,
                            ids:    list,
                            top_k:  int = 5) -> list:
    """
    Find the top_k most confident WRONG predictions per class.
    A wrong prediction is where the model was confident (prob > 0.7)
    but the true label disagreed.
    """
    worst = []
    y_bin = (y_prob >= THRESHOLD).astype(int)
    for i, cls in enumerate(CLASSES):
        # False positives — model said yes (high conf), label said no
        fp_mask = (y_bin[:, i] == 1) & (y_true[:, i] == 0)
        fp_idx  = np.where(fp_mask)[0]
        if len(fp_idx) > 0:
            fp_sorted = fp_idx[np.argsort(-y_prob[fp_idx, i])][:top_k]
            for idx in fp_sorted:
                worst.append({
                    'ecg_id':    ids[idx],
                    'class':     cls,
                    'type':      'False Positive',
                    'confidence': float(y_prob[idx, i]),
                })
        # False negatives — model said no (low prob), label said yes
        fn_mask = (y_bin[:, i] == 0) & (y_true[:, i] == 1)
        fn_idx  = np.where(fn_mask)[0]
        if len(fn_idx) > 0:
            fn_sorted = fn_idx[np.argsort(y_prob[fn_idx, i])][:top_k]
            for idx in fn_sorted:
                worst.append({
                    'ecg_id':    ids[idx],
                    'class':     cls,
                    'type':      'False Negative',
                    'confidence': float(y_prob[idx, i]),
                })
    return worst


def confidence_distribution(y_true: np.ndarray,
                              y_prob: np.ndarray) -> dict:
    """
    For each class: mean confidence when correct vs when wrong.
    A well-calibrated model should be more confident when it's right.
    """
    dist = {}
    y_bin = (y_prob >= THRESHOLD).astype(int)
    for i, cls in enumerate(CLASSES):
        correct_mask = (y_bin[:, i] == y_true[:, i])
        wrong_mask   = ~correct_mask
        dist[cls] = {
            'mean_conf_correct': float(y_prob[correct_mask, i].mean()) if correct_mask.sum() > 0 else 0.0,
            'mean_conf_wrong':   float(y_prob[wrong_mask,   i].mean()) if wrong_mask.sum()   > 0 else 0.0,
            'pct_correct':       float(correct_mask.mean()) * 100,
        }
    return dist


# ── Printing ───────────────────────────────────────────────────────────────────

def print_section(title: str):
    print()
    print("═" * 70)
    print(f"  {title}")
    print("═" * 70)


def print_metrics(metrics: dict, ckpt: dict):
    print_section("PER-CLASS METRICS ON TEST SET (N=1,711)")
    print(f"\n  {'Class':<6}  {'AUC':>6}  {'AP':>6}  {'F1':>6}  "
          f"{'Prec':>6}  {'Recall':>6}  {'Support':>8}")
    print("  " + "─" * 54)
    for cls in CLASSES:
        if cls not in metrics:
            continue
        m = metrics[cls]
        print(f"  {cls:<6}  {m['auc']:>6.3f}  {m['ap']:>6.3f}  {m['f1']:>6.3f}  "
              f"{m['precision']:>6.3f}  {m['recall']:>6.3f}  {m['support']:>8,}")
    print("  " + "─" * 54)
    print(f"  {'macro':<6}  {metrics['macro_auc']:>6.3f}  {metrics['macro_ap']:>6.3f}  "
          f"{metrics['macro_f1']:>6.3f}  {metrics['macro_precision']:>6.3f}  "
          f"{metrics['macro_recall']:>6.3f}")

    # Compare against val metrics from checkpoint
    val_auc = ckpt.get('val_auc', None)
    if val_auc:
        delta = metrics['macro_auc'] - val_auc
        direction = '↑ better than val' if delta >= 0 else '↓ worse than val'
        print(f"\n  Test macro-AUC vs best val macro-AUC: "
              f"{metrics['macro_auc']:.4f} vs {val_auc:.4f}  "
              f"(Δ={delta:+.4f}  {direction})")
        if abs(delta) < 0.01:
            print("  ✓ Gap < 0.01 — model generalises well, no significant overfit")
        elif delta < -0.02:
            print("  ⚠ Gap > 0.02 — some overfitting to the val set")


def print_confusion_matrices(cms: dict):
    print_section("CONFUSION MATRICES (per class, binary: class vs rest)")
    print("\n  How to read: rows = actual, cols = predicted")
    for cls in CLASSES:
        if cls not in cms:
            continue
        cm = cms[cls]
        total   = cm['TP'] + cm['TN'] + cm['FP'] + cm['FN']
        spec    = cm['TN'] / (cm['TN'] + cm['FP']) if (cm['TN'] + cm['FP']) > 0 else 0
        sens    = cm['TP'] / (cm['TP'] + cm['FN']) if (cm['TP'] + cm['FN']) > 0 else 0
        print(f"\n  {cls} ({CLASS_NAMES[cls]})  —  {cm['TP']+cm['FN']} positive cases")
        print(f"                Pred NO    Pred YES")
        print(f"  Actual NO  :  {cm['TN']:>7,}    {cm['FP']:>7,}    (Specificity: {spec:.3f})")
        print(f"  Actual YES :  {cm['FN']:>7,}    {cm['TP']:>7,}    (Sensitivity: {sens:.3f})")


def print_confidence(dist: dict):
    print_section("CONFIDENCE CALIBRATION")
    print(f"\n  {'Class':<6}  {'%Correct':>9}  {'Conf(correct)':>14}  {'Conf(wrong)':>12}")
    print("  " + "─" * 46)
    for cls in CLASSES:
        d = dist[cls]
        calibration = '✓ well calibrated' if d['mean_conf_correct'] > d['mean_conf_wrong'] + 0.1 else '⚠ check'
        print(f"  {cls:<6}  {d['pct_correct']:>8.1f}%  "
              f"{d['mean_conf_correct']:>14.3f}  "
              f"{d['mean_conf_wrong']:>12.3f}  {calibration}")
    print("\n  Conf(correct) should be substantially higher than Conf(wrong).")
    print("  If they are similar, the model is guessing rather than knowing.")


def print_worst(worst: list):
    print_section("MOST CONFIDENT WRONG PREDICTIONS (top 3 per class per type)")
    print("\n  These are the records where the model was most confidently wrong.")
    print("  Use these ecg_ids to inspect the raw signals for edge cases.\n")
    current_cls = None
    count = 0
    for w in sorted(worst, key=lambda x: (x['class'], x['type'], -x['confidence'])):
        if w['class'] != current_cls:
            current_cls = w['class']
            count = 0
            print(f"  {w['class']} ({CLASS_NAMES[w['class']]}):")
        if count < 3:
            print(f"    ecg_id={w['ecg_id']:>6}  {w['type']:<16}  confidence={w['confidence']:.3f}")
            count += 1


# ── Main ───────────────────────────────────────────────────────────────────────

def evaluate():
    print("ECGAttentionNet — Test Set Evaluation")
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")

    model, ckpt = load_model()

    print(f"\nRunning inference on test set...")
    y_true, y_prob, ids = run_test(model)

    print(f"\nComputing metrics over {len(y_true):,} test records...")
    metrics = compute_all_metrics(y_true, y_prob)
    cms     = compute_confusion(y_true, y_prob)
    dist    = confidence_distribution(y_true, y_prob)
    worst   = find_worst_predictions(y_true, y_prob, ids)

    print_metrics(metrics, ckpt)
    print_confusion_matrices(cms)
    print_confidence(dist)
    print_worst(worst)

    print()
    print("═" * 70)
    print("  VERDICT")
    print("═" * 70)
    auc = metrics['macro_auc']
    if auc >= 0.93:
        verdict = "Excellent — competitive with published PTB-XL benchmarks"
    elif auc >= 0.90:
        verdict = "Good — solid clinical screening performance"
    elif auc >= 0.85:
        verdict = "Moderate — usable but room for improvement"
    else:
        verdict = "Needs improvement — consider more epochs or architecture changes"
    print(f"\n  Test macro-AUC : {auc:.4f}")
    print(f"  Assessment     : {verdict}")
    print()


if __name__ == '__main__':
    evaluate()
