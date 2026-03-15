"""
report.py — NeuroKit2 interval extraction + Gemini API clinical report.

Two responsibilities:
  1. extract_clinical_metrics() — HR, PR, QRS, QTc via NeuroKit2.
  2. generate_clinical_report() — Structured markdown report via Gemini API.

Get a free Gemini API key at: https://aistudio.google.com/app/apikey
Set it in config.py as GEMINI_API_KEY or export GEMINI_API_KEY in your terminal.
"""

import os
import re
import numpy as np
import neurokit2 as nk
import google.generativeai as genai

from config import (
    CLASSES, CLASS_NAMES, LEAD_NAMES, FS, LEAD_II_INDEX,
    GEMINI_MODEL, GEMINI_MAX_TOKENS, GEMINI_API_KEY,
)


# ── Clinical metrics ──────────────────────────────────────────────────────────

def extract_clinical_metrics(signal_np: np.ndarray) -> dict:
    """Extract standard ECG intervals from Lead II using NeuroKit2.

    Args:
        signal_np: (12, SIGNAL_LEN) normalised numpy array.
    Returns:
        dict with keys: hr, pr_interval, qrs_duration, qtc (None if failed).
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

        if len(p_peaks) > 1 and len(r_peaks) > 1:
            n     = min(len(p_peaks), len(r_peaks))
            pr_ms = [(r - p) / FS * 1000 for p, r in zip(p_peaks[:n], r_peaks[:n])
                     if 60 < (r - p) / FS * 1000 < 300]
            if pr_ms:
                result['pr_interval'] = round(float(np.median(pr_ms)), 1)

        if len(q_peaks) > 1 and len(s_peaks) > 1:
            n      = min(len(q_peaks), len(s_peaks))
            qrs_ms = [(s - q) / FS * 1000 for q, s in zip(q_peaks[:n], s_peaks[:n])
                      if 40 < (s - q) / FS * 1000 < 200]
            if qrs_ms:
                result['qrs_duration'] = round(float(np.median(qrs_ms)), 1)

        if len(t_ends) > 1 and len(r_peaks) > 2:
            rr_sec    = np.diff(r_peaks) / FS
            rr_median = float(np.median(rr_sec))
            qt_list   = [(t - r) / FS * 1000 for t, r in zip(t_ends, r_peaks)
                         if 200 < (t - r) / FS * 1000 < 600]
            if qt_list and rr_median > 0:
                result['qtc'] = round(float(np.median(qt_list)) / np.sqrt(rr_median), 1)

    except Exception as exc:
        print(f"[NeuroKit2] Interval extraction failed: {exc}")

    return result


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_prompt(predictions:      dict,
                  lead_importance:  dict,
                  clinical_metrics: dict,
                  sub_diagnoses:    list | None = None) -> str:
    """Build a rich structured prompt for Gemini."""

    # Predictions block
    pred_lines = '\n'.join(
        f"- {CLASS_NAMES[cls]}: {prob * 100:.1f}%"
        for cls, prob in sorted(predictions.items(), key=lambda x: -x[1])
        if prob > 0.05
    )

    # Top leads
    top3     = sorted(lead_importance.items(), key=lambda x: -x[1])[:3]
    top3_str = ', '.join(f"Lead {l} ({v * 100:.1f}%)" for l, v in top3)

    # Metrics
    m = clinical_metrics
    metrics_lines = []
    if m.get('hr'):           metrics_lines.append(f"- Heart Rate: **{m['hr']} bpm** (normal 60–100)")
    if m.get('pr_interval'):  metrics_lines.append(f"- PR Interval: **{m['pr_interval']} ms** (normal 120–200)")
    if m.get('qrs_duration'): metrics_lines.append(f"- QRS Duration: **{m['qrs_duration']} ms** (normal <120)")
    if m.get('qtc'):          metrics_lines.append(f"- QTc: **{m['qtc']} ms** (normal <450)")
    metrics_str = '\n'.join(metrics_lines) if metrics_lines else '- Unavailable'

    # Sub-diagnoses block
    sub_dx_str = ''
    if sub_diagnoses:
        lines = []
        for d in sub_diagnoses:
            lines.append(f"- **{d.name}** ({d.confidence} confidence)")
            lines.append(f"  - Territory: {d.territory}")
            lines.append(f"  - Artery: {d.artery}")
            ev_supporting = [e.factor for e in d.evidence if e.supporting][:3]
            if ev_supporting:
                lines.append(f"  - Key evidence: {'; '.join(ev_supporting)}")
        sub_dx_str = '\n'.join(lines)

    return f"""You are an expert AI Cardiology Assistant generating a structured clinical ECG report.
The ECG was analysed by a deep learning model (ResNet-1D + Multi-Head Attention trained on PTB-XL).

---
## INPUT DATA

### Model Predictions
{pred_lines}

### Most Attended Leads (Attention Rollout)
{top3_str}

### Extracted Clinical Metrics
{metrics_str}

{'### Detailed Sub-Diagnoses' + chr(10) + sub_dx_str if sub_dx_str else ''}

---
## YOUR TASK

Generate a **well-formatted clinical ECG report** in Markdown with exactly these sections:

### 🔍 Primary Finding
One sentence stating the most likely diagnosis with confidence percentage.

### 📍 Localisation & Anatomy
2–3 sentences explaining which cardiac territory is involved, which coronary artery supplies it, and why the highlighted leads point to this region.

### 📊 Rhythm & Conduction
2–3 sentences interpreting the heart rate, PR interval, QRS duration, and QTc. Flag any abnormal values explicitly with their normal ranges.

### 🔬 Supporting Evidence
A bullet list (4–6 items) of the specific leads and signal features that led to this conclusion. Each bullet should name the lead, what was observed, and its clinical significance.

### ⚠️ Differential Diagnoses
A short bullet list (2–4 items) of conditions that should be considered and ruled out.

### ✅ Recommended Action
2–3 sentences describing the immediate clinical action recommended based on this ECG.

### ⚕️ Disclaimer
*One sentence stating this report is AI-generated, for educational and screening purposes only, and must be reviewed by a licensed physician before any clinical decision.*

---
Rules:
- Use proper Markdown formatting (headers, bold, bullets, italics)
- Be clinically precise — use correct cardiology terminology
- Flag any critical findings with ⚠️
- Keep each section concise — no padding or repetition
- Never invent findings not supported by the input data"""


# ── Gemini API call ───────────────────────────────────────────────────────────

def generate_clinical_report(predictions:      dict,
                              lead_importance:  dict,
                              clinical_metrics: dict,
                              sub_diagnoses:    list | None = None) -> str:
    """Call Gemini to generate a structured markdown clinical report.

    API key precedence:
      1. config.GEMINI_API_KEY  (if not None)
      2. GEMINI_API_KEY environment variable

    Args:
        predictions:      {class_name: probability}
        lead_importance:  {lead_name: fraction}
        clinical_metrics: dict from extract_clinical_metrics()
        sub_diagnoses:    list of SubDiagnosis from diagnosis_engine (optional)

    Returns:
        Markdown-formatted report string.
    """
    api_key = GEMINI_API_KEY or os.environ.get('GEMINI_API_KEY')
    if not api_key:
        raise ValueError(
            "No Gemini API key found. Set GEMINI_API_KEY in config.py or as an environment variable.\n"
            "Get a free key at: https://aistudio.google.com/app/apikey"
        )

    genai.configure(api_key=api_key)

    model = genai.GenerativeModel(
        model_name     = GEMINI_MODEL,
        generation_config = genai.types.GenerationConfig(
            max_output_tokens = GEMINI_MAX_TOKENS,
            temperature       = 0.3,   # lower = more factual, less creative
        )
    )

    prompt   = _build_prompt(predictions, lead_importance, clinical_metrics, sub_diagnoses)
    response = model.generate_content(prompt)
    return response.text