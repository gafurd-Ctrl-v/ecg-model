import os
import ast
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt
import torch
from torch.utils.data import Dataset, DataLoader

from config import (
    DATA_DIR, FS, SIGNAL_LEN, CLASSES,
    FILTER_LOWCUT, FILTER_HIGHCUT, FILTER_ORDER,
    TEST_FOLD, VAL_FOLD, BATCH_SIZE, NUM_WORKERS,
)


def bandpass_filter(signal: np.ndarray) -> np.ndarray:
    nyq  = FS / 2.0
    b, a = butter(FILTER_ORDER, [FILTER_LOWCUT / nyq, FILTER_HIGHCUT / nyq], btype='band')
    return filtfilt(b, a, signal, axis=-1)


def instance_normalize(signal: np.ndarray) -> np.ndarray:
    mean = signal.mean(axis=-1, keepdims=True)
    std  = signal.std(axis=-1,  keepdims=True) + 1e-8
    return (signal - mean) / std


def pad_or_crop(signal: np.ndarray) -> np.ndarray:
    T = signal.shape[1]
    if T > SIGNAL_LEN:
        return signal[:, :SIGNAL_LEN]
    if T < SIGNAL_LEN:
        return np.pad(signal, ((0, 0), (0, SIGNAL_LEN - T)), mode='constant')
    return signal


class PTBXLDataset(Dataset):
    def __init__(self, df: pd.DataFrame, data_dir: str = DATA_DIR):
        self.df       = df.reset_index(drop=True)
        self.data_dir = data_dir

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row         = self.df.iloc[idx]
        rel_path    = row['filename_hr'].replace('.hea', '')
        record_path = os.path.join(self.data_dir, rel_path)
        record      = wfdb.rdrecord(record_path)
        signal      = record.p_signal.T.astype(np.float32)

        signal = pad_or_crop(signal)
        signal = bandpass_filter(signal)
        signal = instance_normalize(signal)
        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

        label = row[CLASSES].values.astype(np.float32)
        return torch.from_numpy(signal), torch.from_numpy(label)


def load_ptbxl(data_dir: str = DATA_DIR) -> tuple:
    csv_path = os.path.join(data_dir, 'ptbxl_database.csv')
    scp_path = os.path.join(data_dir, 'scp_statements.csv')

    df  = pd.read_csv(csv_path, index_col='ecg_id')
    scp = pd.read_csv(scp_path, index_col=0)

    # Keep only codes that have a known diagnostic class
    scp = scp[scp['diagnostic_class'].notna()]

    # Build lookup: scp_code → superclass  e.g. 'IMI' → 'MI'
    code_to_superclass = scp['diagnostic_class'].to_dict()

    def parse_scp(raw):
        if not isinstance(raw, str):
            return []
        try:
            codes = ast.literal_eval(raw)
        except Exception:
            return []
        found = set()
        for code, confidence in codes.items():
            if confidence >= 100.0 and code in code_to_superclass:
                sc = code_to_superclass[code]
                if sc in CLASSES:
                    found.add(sc)
        return list(found)

    df['superclass_list'] = df['scp_codes'].apply(parse_scp)

    for cls in CLASSES:
        df[cls] = df['superclass_list'].apply(lambda x: 1.0 if cls in x else 0.0)

    df = df[df[CLASSES].sum(axis=1) > 0].copy()

    test_df  = df[df['strat_fold'] == TEST_FOLD]
    val_df   = df[df['strat_fold'] == VAL_FOLD]
    train_df = df[~df['strat_fold'].isin([TEST_FOLD, VAL_FOLD])]

    print(f"Split sizes — Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")
    return train_df, val_df, test_df


def get_dataloaders(data_dir:    str = DATA_DIR,
                    batch_size:  int = BATCH_SIZE,
                    num_workers: int = NUM_WORKERS) -> tuple:
    train_df, val_df, test_df = load_ptbxl(data_dir)

    kwargs = dict(pin_memory=True, persistent_workers=(num_workers > 0),
                  num_workers=num_workers)

    train_loader = DataLoader(PTBXLDataset(train_df, data_dir),
                              batch_size=batch_size, shuffle=True,  **kwargs)
    val_loader   = DataLoader(PTBXLDataset(val_df,   data_dir),
                              batch_size=batch_size, shuffle=False, **kwargs)
    test_loader  = DataLoader(PTBXLDataset(test_df,  data_dir),
                              batch_size=batch_size, shuffle=False, **kwargs)

    return train_loader, val_loader, test_loader
