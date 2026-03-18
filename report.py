"""
report.py — NeuroKit2 ECG interval extraction.

Anthropic/LLM report generation has been removed and will be added
back later as a Gemini-based feature. This module now only exposes
extract_clinical_metrics(), which is used by app.py to populate the
HR / PR / QRS / QTc metric cards.
"""

import numpy as np
import neurokit2 as nk

from config import FS, LEAD_II_INDEX


def extract_clinical_metrics(signal_np: np.ndarray) -> dict:
    """Extract standard ECG intervals using NeuroKit2.

    Operates on Lead II (index LEAD_II_INDEX from config.py).
    Sampling rate FS is read from config.py.

    Args:
        signal_np: (12, SIGNAL_LEN) normalised numpy array.
    Returns:
        dict with keys: hr, pr_interval, qrs_duration, qtc  (None if failed).
    """
    lead   = signal_np[LEAD_II_INDEX]
    result = {'hr': None, 'pr_interval': None, 'qrs_duration': None, 'qtc': None}

    try:
        cleaned = nk.ecg_clean(lead, sampling_rate=FS, method='pantompkins1985')
        signals, info = nk.ecg_process(cleaned, sampling_rate=FS)

        if 'ECG_Rate' in signals:
            result['hr'] = round(float(signals['ECG_Rate'].mean()), 1)

        r_peaks = info.get('ECG_R_Peaks',   np.array([]))
        p_peaks = info.get('ECG_P_Peaks',   np.array([]))
        q_peaks = info.get('ECG_Q_Peaks',   np.array([]))
        s_peaks = info.get('ECG_S_Peaks',   np.array([]))
        t_ends  = info.get('ECG_T_Offsets', np.array([]))

        # PR interval
        if len(p_peaks) > 1 and len(r_peaks) > 1:
            n     = min(len(p_peaks), len(r_peaks))
            pr_ms = [(r - p) / FS * 1000 for p, r in zip(p_peaks[:n], r_peaks[:n])
                     if 60 < (r - p) / FS * 1000 < 300]
            if pr_ms:
                result['pr_interval'] = round(float(np.median(pr_ms)), 1)

        # QRS duration
        if len(q_peaks) > 1 and len(s_peaks) > 1:
            n      = min(len(q_peaks), len(s_peaks))
            qrs_ms = [(s - q) / FS * 1000 for q, s in zip(q_peaks[:n], s_peaks[:n])
                      if 40 < (s - q) / FS * 1000 < 200]
            if qrs_ms:
                result['qrs_duration'] = round(float(np.median(qrs_ms)), 1)

        # QTc — Bazett's formula
        if len(t_ends) > 1 and len(r_peaks) > 2:
            rr_sec     = np.diff(r_peaks) / FS
            rr_median  = float(np.median(rr_sec))
            qt_ms_list = [(t - r) / FS * 1000 for t, r in zip(t_ends, r_peaks)
                          if 200 < (t - r) / FS * 1000 < 600]
            if qt_ms_list and rr_median > 0:
                result['qtc'] = round(float(np.median(qt_ms_list)) / np.sqrt(rr_median), 1)

    except Exception as exc:
        print(f"[NeuroKit2] Interval extraction failed: {exc}")

    return result