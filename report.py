"""
report.py — NeuroKit2 interval extraction + Anthropic API clinical summary.

All constants (CLASSES, CLASS_NAMES, LEAD_NAMES, FS, ANTHROPIC_MODEL,
ANTHROPIC_MAX_TOKENS, ANTHROPIC_API_KEY, LEAD_II_INDEX) imported from config.py.
"""

import os
import numpy as np
import neurokit2 as nk
from anthropic import Anthropic

from config import (
    CLASSES,
    CLASS_NAMES,
    LEAD_NAMES,
    FS,
    LEAD_II_INDEX,
    ANTHROPIC_MODEL,
    ANTHROPIC_MAX_TOKENS,
    ANTHROPIC_API_KEY,
)


# ── Clinical metrics ──────────────────────────────────────────────────────────

def extract_clinical_metrics(signal_np: np.ndarray) -> dict:
    """Extract standard ECG intervals using NeuroKit2.

    Operates on Lead II (index LEAD_II_INDEX from config.py).
    Sampling rate FS is read from config.py.

    Args:
        signal_np: (12, SIGNAL_LEN) normalised numpy array.
    Returns:
        dict with keys: hr, pr_interval, qrs_duration, qtc  (None if failed).
    """
    lead = signal_np[LEAD_II_INDEX]
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
            n      = min(len(p_peaks), len(r_peaks))
            pr_ms  = [(r - p) / FS * 1000 for p, r in zip(p_peaks[:n], r_peaks[:n])
                      if 60 < (r - p) / FS * 1000 < 300]
            if pr_ms:
                result['pr_interval'] = round(float(np.median(pr_ms)), 1)

        # QRS duration
        if len(q_peaks) > 1 and len(s_peaks) > 1:
            n       = min(len(q_peaks), len(s_peaks))
            qrs_ms  = [(s - q) / FS * 1000 for q, s in zip(q_peaks[:n], s_peaks[:n])
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


# ── LLM report ────────────────────────────────────────────────────────────────

def _build_prompt(predictions:      dict,
                  lead_importance:  dict,
                  clinical_metrics: dict) -> str:
    """Construct the structured prompt sent to Claude."""

    pred_lines = '\n'.join(
        f"  • {CLASS_NAMES[cls]}: {prob * 100:.1f}%"
        for cls, prob in sorted(predictions.items(), key=lambda x: -x[1])
        if prob > 0.05
    )

    top3     = sorted(lead_importance.items(), key=lambda x: -x[1])[:3]
    top3_str = ', '.join(f"Lead {l} ({v * 100:.1f}%)" for l, v in top3)

    m = clinical_metrics
    metric_lines = []
    if m.get('hr'):           metric_lines.append(f"  • Heart Rate:    {m['hr']} bpm")
    if m.get('pr_interval'):  metric_lines.append(f"  • PR Interval:   {m['pr_interval']} ms  (normal 120–200 ms)")
    if m.get('qrs_duration'): metric_lines.append(f"  • QRS Duration:  {m['qrs_duration']} ms  (normal <120 ms)")
    if m.get('qtc'):          metric_lines.append(f"  • QTc:           {m['qtc']} ms  (normal <450 ms)")
    metrics_str = '\n'.join(metric_lines) if metric_lines else '  • Unavailable'

    return f"""You are an AI Cardiology Assistant interpreting a 12-lead ECG analysed by a ResNet-Attention deep learning model.

ANALYSIS RESULTS
────────────────
Model Predictions (sigmoid probabilities):
{pred_lines}

Most Significant Leads (attention rollout):
  {top3_str}

Extracted Clinical Metrics:
{metrics_str}

TASK
────
Write a clinical ECG summary of exactly 3–4 sentences that:
1. States the primary finding with its confidence level.
2. Explains the anatomical significance of the most-attended leads.
3. Comments on rhythm and conduction based on the interval values.
4. Ends with a concise recommended clinical action.

Finish with a one-sentence disclaimer that this summary is AI-generated and requires review by a licensed physician before any clinical decision.

Respond with the summary only — no headers, no bullet points."""


def generate_clinical_report(predictions:      dict,
                              lead_importance:  dict,
                              clinical_metrics: dict) -> str:
    """Call Claude to generate a plain-language clinical summary.

    API key precedence:
      1. config.ANTHROPIC_API_KEY  (if not None)
      2. ANTHROPIC_API_KEY environment variable
    Model and token limit come from config.py.
    """
    api_key = ANTHROPIC_API_KEY or os.environ.get('ANTHROPIC_API_KEY')
    client  = Anthropic(api_key=api_key) if api_key else Anthropic()

    message = client.messages.create(
        model      = ANTHROPIC_MODEL,
        max_tokens = ANTHROPIC_MAX_TOKENS,
        messages   = [{"role": "user", "content": _build_prompt(
            predictions, lead_importance, clinical_metrics
        )}]
    )
    return message.content[0].text
