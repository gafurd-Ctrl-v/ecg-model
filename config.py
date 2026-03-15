"""
config.py — Single source of truth for the entire ECG pipeline.

Every path, hyperparameter, and constant lives here.
All other files import from this module — nothing is hardcoded elsewhere.

HOW TO USE:
  - Change DATA_DIR to point at your PTB-XL folder.
  - Change CHECKPOINT_DIR if you want checkpoints saved elsewhere.
  - Adjust BATCH_SIZE to 8 if you get CUDA out-of-memory on RTX 3050.
  - Set ANTHROPIC_API_KEY here OR export it as an environment variable
    (the environment variable takes precedence).
"""

import os

# ═══════════════════════════════════════════════════════════════════════════════
# PATHS — change these to match your system
# ═══════════════════════════════════════════════════════════════════════════════

# Root folder of your PTB-XL download.
# Must contain: ptbxl_database.csv  and  records500/
DATA_DIR = r'C:\Users\soham\Downloads\MJPR\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3'

# Where best_model.pt will be saved during training and loaded during inference.
CHECKPOINT_DIR = 'checkpoints'
CHECKPOINT_FILE = 'best_model.pt'   # filename only; combined with CHECKPOINT_DIR below
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, CHECKPOINT_FILE)

# ═══════════════════════════════════════════════════════════════════════════════
# DATASET CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Sampling rate of the records500/ files (Hz)
FS = 500

# Number of time samples per recording (10 s × 500 Hz)
SIGNAL_LEN = 5000

# PTB-XL diagnostic superclasses used as classification targets
CLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']

# Human-readable class names (used in reports and UI)
CLASS_NAMES = {
    'NORM': 'Normal Sinus Rhythm',
    'MI':   'Myocardial Infarction',
    'STTC': 'ST/T Change',
    'CD':   'Conduction Disturbance',
    'HYP':  'Hypertrophy',
}

# Standard 12-lead ECG lead ordering as stored in PTB-XL
LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
              'V1', 'V2', 'V3',  'V4',  'V5',  'V6']

# Index of Lead II in LEAD_NAMES — used for NeuroKit2 interval extraction
LEAD_II_INDEX = 1

# PTB-XL stratified fold assignments
TEST_FOLD  = 10
VAL_FOLD   =  9
# Training folds: everything not in TEST_FOLD or VAL_FOLD (i.e. 1–8)

# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

# Butterworth bandpass filter
FILTER_LOWCUT  =  0.5   # Hz — removes baseline wander (breathing artefact)
FILTER_HIGHCUT = 45.0   # Hz — removes EMG noise (muscle artefact)
FILTER_ORDER   =  4     # filter order; 4th-order is standard for ECG

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════════

NUM_CLASSES = len(CLASSES)  # 5

# ResNet backbone base channels.
# Channel progression: BASE_CH → BASE_CH×2 → BASE_CH×4 → BASE_CH×8
# With BASE_CH=32: 32 → 64 → 128 → 256 channels (~2.1M parameters, fits in 4 GB VRAM)
BASE_CH  = 32

# Number of attention heads in MultiheadAttention.
# Must evenly divide (BASE_CH × 8). With BASE_CH=32: 256 / 4 = 64 per head.
# Increase to 8 if you have more VRAM.
N_HEADS  = 4

# Dropout rate used throughout (residual blocks + attention + classifier head)
DROPOUT  = 0.2

# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING HYPERPARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

# Samples per training step.
# RTX 3050 4 GB: use 16 (peaks ~2.5 GB VRAM with AMP).
# If CUDA OOM: lower to 8.
BATCH_SIZE   = 64

# Total training epochs
EPOCHS       = 49

# Peak learning rate for OneCycleLR
LEARNING_RATE = 3e-4

# AdamW weight decay
WEIGHT_DECAY  = 1e-4

# Fraction of total steps used for linear warm-up before cosine decay
LR_WARMUP_PCT = 0.1

# Gradient clipping max norm (prevents exploding gradients with AMP)
GRAD_CLIP_NORM = 1.0

# DataLoader worker processes.
# Windows users: set to 0 (Windows doesn't support forked worker processes).
# Linux users: 4 is a good default.
NUM_WORKERS = 0

# ═══════════════════════════════════════════════════════════════════════════════
# EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════════════════════

# Number of top leads to highlight in the attention rollout bar chart
TOP_K_LEADS = 3

# ═══════════════════════════════════════════════════════════════════════════════
# ANTHROPIC API
# ═══════════════════════════════════════════════════════════════════════════════

# Claude model used for clinical report generation.
ANTHROPIC_MODEL     = 'claude-sonnet-4-20250514'
ANTHROPIC_MAX_TOKENS = 512

# API key — leave as None to read from the ANTHROPIC_API_KEY environment variable.
# If you set it here, it takes precedence over the environment variable.
# WARNING: do not commit a real key to version control.
ANTHROPIC_API_KEY = None   # e.g. "sk-ant-api03-..."
