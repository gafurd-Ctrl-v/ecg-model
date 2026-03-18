import os

# ═══════════════════════════════════════════════════════════════════════════════
# PATHS — change these to match your system
# ═══════════════════════════════════════════════════════════════════════════════

DATA_DIR        = r"C:\Users\soham\Downloads\MJPR\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
CHECKPOINT_DIR  = 'checkpoints'
CHECKPOINT_FILE = 'best_model.pt'
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, CHECKPOINT_FILE)

# ═══════════════════════════════════════════════════════════════════════════════
# DATASET CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

FS         = 500
SIGNAL_LEN = 5000

CLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']

CLASS_NAMES = {
    'NORM': 'Normal Sinus Rhythm',
    'MI':   'Myocardial Infarction',
    'STTC': 'ST/T Change',
    'CD':   'Conduction Disturbance',
    'HYP':  'Hypertrophy',
}

LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
              'V1', 'V2', 'V3',  'V4',  'V5',  'V6']

LEAD_II_INDEX = 1

TEST_FOLD = 10
VAL_FOLD  =  9

# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

FILTER_LOWCUT  =  0.5
FILTER_HIGHCUT = 45.0
FILTER_ORDER   =  4

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════════

NUM_CLASSES = len(CLASSES)   # 5
BASE_CH     = 32
N_HEADS     = 4
DROPOUT     = 0.2

# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING HYPERPARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

BATCH_SIZE      = 64
EPOCHS          = 49
LEARNING_RATE   = 3e-4
WEIGHT_DECAY    = 1e-4
LR_WARMUP_PCT   = 0.1
GRAD_CLIP_NORM  = 1.0
NUM_WORKERS     = 0

# ═══════════════════════════════════════════════════════════════════════════════
# EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════════════════════

TOP_K_LEADS = 3