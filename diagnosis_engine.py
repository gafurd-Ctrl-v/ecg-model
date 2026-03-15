"""
diagnosis_engine.py — Detailed sub-diagnosis engine with evidence chains.

Combines three information sources to produce specific diagnoses:
  1. Model predictions (which of the 5 superclasses is most likely)
  2. Lead importance (attention rollout — which leads the model focused on)
  3. Clinical metrics (HR, PR, QRS, QTc from NeuroKit2)

For each confident prediction, the engine:
  - Maps it to a specific sub-diagnosis (e.g. MI → Anterior STEMI)
  - Builds an evidence chain: which leads contributed and why
  - Assigns a confidence level (High / Moderate / Low)
  - Lists supporting and opposing factors
  - Provides a brief clinical interpretation

Sub-diagnosis taxonomy:
  NORM  → Normal Sinus Rhythm, Sinus Tachycardia, Sinus Bradycardia
  MI    → Anterior STEMI, Inferior STEMI, Lateral STEMI,
          Anteroseptal MI, Posterior MI, NSTEMI (non-localised)
  STTC  → Global Ischemia / Left Main, Anterior Ischemia,
          Inferior Ischemia, Lateral Ischemia, Early Repolarisation
  CD    → LBBB, RBBB, Left Anterior Fascicular Block,
          Left Posterior Fascicular Block, 1st-degree AV Block,
          Non-specific IVCD
  HYP   → Left Ventricular Hypertrophy, Right Ventricular Hypertrophy,
          Biventricular Hypertrophy
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from config import CLASSES, LEAD_NAMES


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class EvidenceItem:
    """A single piece of evidence supporting or opposing a conclusion."""
    lead:        Optional[str]   # e.g. 'V3', or None if metric-based
    factor:      str             # e.g. 'High attention (34.2%)'
    detail:      str             # e.g. 'Anterior wall — LAD territory'
    weight:      float           # 0–1, how strong this evidence is
    supporting:  bool            # True = supports diagnosis, False = opposes


@dataclass
class SubDiagnosis:
    """A specific localised diagnosis with full evidence chain."""
    superclass:       str              # e.g. 'MI'
    name:             str              # e.g. 'Anterior STEMI'
    confidence:       str              # 'High' / 'Moderate' / 'Low'
    confidence_score: float            # 0–1
    territory:        str              # e.g. 'Anterior wall (V3, V4)'
    artery:           str              # e.g. 'Left Anterior Descending (LAD)'
    key_leads:        list[str]        # leads most relevant to this diagnosis
    evidence:         list[EvidenceItem] = field(default_factory=list)
    interpretation:   str             = ''
    differentials:    list[str]        = field(default_factory=list)
    clinical_action:  str             = ''


# ── Lead group definitions ─────────────────────────────────────────────────────

INFERIOR_LEADS   = {'II', 'III', 'aVF'}
ANTERIOR_LEADS   = {'V3', 'V4'}
SEPTAL_LEADS     = {'V1', 'V2'}
LATERAL_LEADS    = {'I', 'aVL', 'V5', 'V6'}
HIGH_LAT_LEADS   = {'I', 'aVL'}
LOW_LAT_LEADS    = {'V5', 'V6'}
ANTERO_SEP_LEADS = {'V1', 'V2', 'V3', 'V4'}
GLOBAL_LEADS     = {'aVR'}
RIGHT_LEADS      = {'V1', 'V2', 'III', 'aVF'}


def top_leads_in_group(lead_importance: dict,
                        group: set,
                        threshold: float = 0.08) -> list[tuple[str, float]]:
    """Return leads from a group that exceed the importance threshold, sorted descending."""
    return sorted(
        [(l, v) for l, v in lead_importance.items() if l in group and v >= threshold],
        key=lambda x: -x[1]
    )


def group_importance(lead_importance: dict, group: set) -> float:
    """Sum of importance scores for all leads in a group."""
    return sum(lead_importance.get(l, 0) for l in group)


def top_n_leads(lead_importance: dict, n: int = 3) -> list[str]:
    return [l for l, _ in sorted(lead_importance.items(), key=lambda x: -x[1])[:n]]


# ── Sub-diagnosis rules ────────────────────────────────────────────────────────

def _diagnose_mi(lead_importance: dict,
                 metrics:          dict,
                 model_prob:       float) -> SubDiagnosis:
    """Map MI superclass to a specific territory and sub-type."""

    inf_score   = group_importance(lead_importance, INFERIOR_LEADS)
    ant_score   = group_importance(lead_importance, ANTERIOR_LEADS)
    sep_score   = group_importance(lead_importance, SEPTAL_LEADS)
    lat_score   = group_importance(lead_importance, LATERAL_LEADS)
    ant_sep_sc  = group_importance(lead_importance, ANTERO_SEP_LEADS)

    scores = {
        'inferior':    inf_score,
        'anterior':    ant_score,
        'anteroseptal': ant_sep_sc * 0.7,
        'lateral':     lat_score,
        'septal':      sep_score,
    }
    dominant = max(scores, key=scores.get)

    evidence = []

    if dominant == 'inferior' or inf_score > 0.25:
        key_leads  = [l for l, _ in top_leads_in_group(lead_importance, INFERIOR_LEADS, 0.05)]
        name       = 'Inferior STEMI / Inferior MI'
        territory  = 'Inferior wall of left ventricle (diaphragmatic surface)'
        artery     = 'Right Coronary Artery (RCA) — dominant in 85% of patients'
        action     = 'Urgent cardiology review. Check right-sided leads (V4R) to rule out RV involvement.'
        differentials = ['Posterior MI (check V7-V9 if available)', 'Pericarditis (if diffuse)']

        for l, v in top_leads_in_group(lead_importance, INFERIOR_LEADS, 0.05):
            evidence.append(EvidenceItem(
                lead=l, factor=f'High attention ({v*100:.1f}%)',
                detail='Inferior lead — directly faces inferior LV wall',
                weight=v, supporting=True
            ))
        for l, v in top_leads_in_group(lead_importance, HIGH_LAT_LEADS, 0.05):
            evidence.append(EvidenceItem(
                lead=l, factor=f'Reciprocal attention ({v*100:.1f}%)',
                detail='High lateral leads show reciprocal depression in inferior MI',
                weight=v * 0.5, supporting=True
            ))
        interp = (
            f'Model confidence {model_prob*100:.1f}% for MI with dominant inferior lead '
            f'attention ({inf_score*100:.1f}% of total). Inferior MI pattern implicates '
            f'RCA territory. Right ventricular involvement should be excluded.'
        )

    elif dominant in ('anterior', 'anteroseptal') or ant_sep_sc > 0.25:
        ant_leads = top_leads_in_group(lead_importance, ANTERO_SEP_LEADS, 0.05)
        key_leads = [l for l, _ in ant_leads]

        if sep_score > ant_score:
            name      = 'Anteroseptal MI'
            territory = 'Interventricular septum + adjacent anterior wall (V1-V3)'
            artery    = 'Proximal Left Anterior Descending (LAD) — septal perforators'
            action    = 'Urgent cardiology review. Proximal LAD occlusion — large territory at risk.'
            differentials = ['LBBB (can mimic anterior MI)', 'Anterior STEMI (if V3-V4 also involved)']
            interp = (
                f'High septal lead attention (V1-V2: {sep_score*100:.1f}%) suggests '
                f'anteroseptal involvement. Proximal LAD territory. Bundle branch block '
                f'should be excluded as it can mimic this pattern.'
            )
        else:
            name      = 'Anterior STEMI / Anterior MI'
            territory = 'Anterior wall of left ventricle (V3-V4, often V1-V6)'
            artery    = 'Left Anterior Descending (LAD) — mid to distal'
            action    = 'Urgent cardiology review. LAD territory — largest MI territory, highest risk.'
            differentials = ['Early repolarisation (in younger patients)', 'LBBB with Sgarbossa criteria']
            interp = (
                f'Dominant anterior lead attention (V3-V4: {ant_score*100:.1f}%) indicates '
                f'anterior wall involvement. LAD territory. Anterior MI carries highest '
                f'in-hospital mortality due to large myocardial territory.'
            )

        for l, v in ant_leads:
            evidence.append(EvidenceItem(
                lead=l, factor=f'High attention ({v*100:.1f}%)',
                detail='Anterior/septal lead — faces LAD territory',
                weight=v, supporting=True
            ))

    elif dominant == 'lateral' or lat_score > 0.20:
        key_leads = [l for l, _ in top_leads_in_group(lead_importance, LATERAL_LEADS, 0.05)]
        name      = 'Lateral STEMI / Lateral MI'
        territory = 'Lateral wall of left ventricle'
        artery    = 'Left Circumflex (LCx) — the "electrically silent" artery'
        action    = 'Cardiology review. LCx occlusion is often missed — low voltage changes.'
        differentials = ['High lateral MI (I, aVL only)', 'Anterolateral MI (V4-V6 + I, aVL)']
        interp = (
            f'Lateral lead dominance ({lat_score*100:.1f}%) implicates LCx territory. '
            f'Circumflex occlusions produce smaller ECG changes and are more frequently '
            f'missed — clinical correlation essential.'
        )
        for l, v in top_leads_in_group(lead_importance, LATERAL_LEADS, 0.05):
            evidence.append(EvidenceItem(
                lead=l, factor=f'High attention ({v*100:.1f}%)',
                detail='Lateral lead — faces LCx territory',
                weight=v, supporting=True
            ))

    else:
        # Non-localised — insufficient lead dominance
        key_leads  = top_n_leads(lead_importance, 3)
        name       = 'Non-localised MI / NSTEMI pattern'
        territory  = 'Not clearly localised — multi-territory or subendocardial'
        artery     = 'Unable to localise — may be multivessel disease'
        action     = 'Cardiology review. Troponin and serial ECGs recommended.'
        differentials = ['NSTEMI', 'Demand ischemia (Type 2 MI)', 'Multivessel disease']
        interp     = (
            f'Model confidence {model_prob*100:.1f}% for MI but no dominant lead territory '
            f'(max group score {max(scores.values())*100:.1f}%). Consider NSTEMI or '
            f'non-localised subendocardial ischemia.'
        )

    # Add metric-based evidence
    qrs = metrics.get('qrs_duration')
    if qrs and qrs > 120:
        evidence.append(EvidenceItem(
            lead=None,
            factor=f'Prolonged QRS ({qrs} ms)',
            detail='Wide QRS may indicate bundle branch block masking MI — apply Sgarbossa criteria',
            weight=0.6, supporting=False
        ))
    hr = metrics.get('hr')
    if hr and hr > 100:
        evidence.append(EvidenceItem(
            lead=None,
            factor=f'Tachycardia ({hr} bpm)',
            detail='Compensatory tachycardia — consistent with significant MI',
            weight=0.4, supporting=True
        ))

    conf_score = model_prob * min(1.0, max(scores.values()) * 4)
    confidence = 'High' if conf_score > 0.65 else ('Moderate' if conf_score > 0.40 else 'Low')

    return SubDiagnosis(
        superclass=CLASSES[1], name=name, confidence=confidence,
        confidence_score=conf_score, territory=territory, artery=artery,
        key_leads=key_leads, evidence=evidence, interpretation=interp,
        differentials=differentials, clinical_action=action,
    )


def _diagnose_sttc(lead_importance: dict,
                   metrics:          dict,
                   model_prob:       float) -> SubDiagnosis:
    """Map STTC superclass to specific ischemia pattern."""

    inf_score = group_importance(lead_importance, INFERIOR_LEADS)
    ant_score = group_importance(lead_importance, ANTERO_SEP_LEADS)
    lat_score = group_importance(lead_importance, LATERAL_LEADS)
    glb_score = lead_importance.get('aVR', 0)

    evidence = []

    if glb_score > 0.15:
        name      = 'Global Ischemia / Left Main Disease pattern'
        territory = 'Diffuse subendocardial — all territories'
        artery    = 'Left Main Coronary Artery or proximal multivessel disease'
        key_leads = ['aVR'] + top_n_leads(
            {l: v for l, v in lead_importance.items() if l != 'aVR'}, 2)
        action    = 'URGENT — aVR elevation with diffuse ST changes = Left Main threat. Immediate cardiology.'
        differentials = ['Proximal LAD occlusion', 'Tachyarrhythmia-induced ischemia']
        interp    = (
            f'Prominent aVR attention ({glb_score*100:.1f}%) with diffuse ST/T changes '
            f'raises concern for left main coronary artery disease or severe multivessel '
            f'disease. This is a high-risk pattern requiring immediate evaluation.'
        )
        evidence.append(EvidenceItem(
            lead='aVR', factor=f'High global attention ({glb_score*100:.1f}%)',
            detail='aVR elevation = global subendocardial ischemia / left main disease',
            weight=glb_score, supporting=True
        ))

    elif inf_score > ant_score and inf_score > lat_score:
        name      = 'Inferior ST/T Changes'
        territory = 'Inferior wall'
        artery    = 'Right Coronary Artery (RCA) — possible demand ischemia'
        key_leads = [l for l, _ in top_leads_in_group(lead_importance, INFERIOR_LEADS, 0.05)]
        action    = 'Cardiology review. Serial ECGs and troponin to differentiate ischemia from non-ischemic cause.'
        differentials = ['Early inferior MI', 'Pulmonary embolism (inferior strain)', 'Normal variant']
        interp    = f'Inferior lead ST/T changes ({inf_score*100:.1f}% attention). RCA territory.'
        for l, v in top_leads_in_group(lead_importance, INFERIOR_LEADS, 0.05):
            evidence.append(EvidenceItem(
                lead=l, factor=f'Attention {v*100:.1f}%',
                detail='Inferior ST/T change — inferior wall ischemia',
                weight=v, supporting=True
            ))

    elif ant_score > lat_score:
        name      = 'Anterior ST/T Changes'
        territory = 'Anterior / anteroseptal wall'
        artery    = 'Left Anterior Descending (LAD)'
        key_leads = [l for l, _ in top_leads_in_group(lead_importance, ANTERO_SEP_LEADS, 0.05)]
        action    = 'Cardiology review. Anterior T-wave changes may indicate LAD ischemia or Wellens syndrome.'
        differentials = ["Wellens syndrome (critical LAD stenosis)", 'Early anterior MI', 'LVH repolarisation']
        interp    = f'Anterior/septal ST-T abnormality ({ant_score*100:.1f}% attention). LAD territory.'
        for l, v in top_leads_in_group(lead_importance, ANTERO_SEP_LEADS, 0.05):
            evidence.append(EvidenceItem(
                lead=l, factor=f'Attention {v*100:.1f}%',
                detail='Anterior ST/T — LAD territory ischemia',
                weight=v, supporting=True
            ))

    else:
        name      = 'Lateral ST/T Changes'
        territory = 'Lateral wall'
        artery    = 'Left Circumflex (LCx)'
        key_leads = [l for l, _ in top_leads_in_group(lead_importance, LATERAL_LEADS, 0.05)]
        action    = 'Cardiology review. Lateral changes may be ischemic or secondary.'
        differentials = ['LVH with repolarisation changes', 'Lateral MI', 'Drug effect (digoxin)']
        interp    = f'Lateral ST/T changes ({lat_score*100:.1f}% attention). LCx territory.'
        for l, v in top_leads_in_group(lead_importance, LATERAL_LEADS, 0.05):
            evidence.append(EvidenceItem(
                lead=l, factor=f'Attention {v*100:.1f}%',
                detail='Lateral ST/T — LCx territory',
                weight=v, supporting=True
            ))

    qtc = metrics.get('qtc')
    if qtc and qtc > 450:
        evidence.append(EvidenceItem(
            lead=None, factor=f'Prolonged QTc ({qtc} ms)',
            detail='QTc prolongation can accompany ischemia or indicate drug effect',
            weight=0.5, supporting=True
        ))

    conf_score = model_prob * 0.85
    confidence = 'High' if conf_score > 0.70 else ('Moderate' if conf_score > 0.45 else 'Low')

    return SubDiagnosis(
        superclass=CLASSES[2], name=name, confidence=confidence,
        confidence_score=conf_score, territory=territory, artery=artery,
        key_leads=key_leads, evidence=evidence, interpretation=interp,
        differentials=differentials, clinical_action=action,
    )


def _diagnose_cd(lead_importance: dict,
                 metrics:          dict,
                 model_prob:       float) -> SubDiagnosis:
    """Map CD superclass to specific conduction diagnosis."""

    qrs = metrics.get('qrs_duration')
    pr  = metrics.get('pr_interval')
    v1_imp  = lead_importance.get('V1', 0)
    v6_imp  = lead_importance.get('V6', 0)
    lat_imp = group_importance(lead_importance, LOW_LAT_LEADS)
    sep_imp = group_importance(lead_importance, SEPTAL_LEADS)
    inf_imp = group_importance(lead_importance, INFERIOR_LEADS)

    evidence = []

    if qrs and qrs >= 120:
        if v1_imp > lat_imp:
            name      = 'Right Bundle Branch Block (RBBB)'
            territory = 'Right bundle branch — delayed right ventricular activation'
            artery    = 'Variable — RBBB can be normal variant or indicate RV disease'
            key_leads = ['V1', 'V2', 'V5', 'V6']
            action    = 'Cardiology review if new. Isolated RBBB can be normal. Rule out PE and RV disease.'
            differentials = ['Pulmonary embolism (new RBBB)', 'Brugada syndrome', 'Normal variant']
            interp    = (
                f'Wide QRS ({qrs} ms) with high V1 attention ({v1_imp*100:.1f}%) suggests RBBB. '
                f'Classic rSR\' in V1 and wide S waves in lateral leads expected.'
            )
            evidence += [
                EvidenceItem('V1', f'High attention ({v1_imp*100:.1f}%)',
                             'V1 rSR\' pattern — right bundle delay', v1_imp, True),
                EvidenceItem(None, f'Wide QRS ({qrs} ms)',
                             'Conduction delay ≥120ms — bundle branch block criteria', 0.9, True),
            ]
        else:
            name      = 'Left Bundle Branch Block (LBBB)'
            territory = 'Left bundle branch — delayed left ventricular activation'
            artery    = 'Variable — new LBBB may indicate anterior MI (Sgarbossa criteria)'
            key_leads = ['V1', 'V5', 'V6', 'I']
            action    = 'If new LBBB: treat as STEMI equivalent until proven otherwise. Urgent cardiology.'
            differentials = ['New LBBB = STEMI equivalent', 'Old LBBB (benign if chronic)', 'Paced rhythm']
            interp    = (
                f'Wide QRS ({qrs} ms) with lateral lead dominance suggests LBBB. '
                f'New LBBB should always raise concern for anterior MI — Sgarbossa criteria apply.'
            )
            evidence += [
                EvidenceItem('V5', f'High lateral attention ({lat_imp*100:.1f}%)',
                             'Broad monophasic R in lateral leads — LBBB pattern', lat_imp, True),
                EvidenceItem(None, f'Wide QRS ({qrs} ms)',
                             'Conduction delay ≥120ms', 0.9, True),
            ]

    elif pr and pr > 200:
        name      = 'First-degree AV Block'
        territory = 'AV node — slowed conduction between atria and ventricles'
        artery    = 'AV nodal artery (branch of RCA in 85% of patients)'
        key_leads = ['II', 'V1']
        action    = 'Monitor. Usually benign. Check for Lyme disease, medication effects, inferior MI.'
        differentials = ['High vagal tone (athletes)', 'Drug effect (beta-blockers, digoxin)', 'Inferior MI']
        interp    = (
            f'PR interval {pr} ms (normal 120-200 ms) indicates prolonged AV conduction. '
            f'Usually benign but warrants clinical correlation.'
        )
        evidence.append(EvidenceItem(
            None, f'PR interval {pr} ms',
            'Prolonged AV conduction time — first-degree block criterion >200ms', 0.95, True
        ))

    elif inf_imp > sep_imp and inf_imp > lat_imp:
        name      = 'Left Posterior Fascicular Block (LPFB)'
        territory = 'Left posterior fascicle — inferior-posterior LV conduction'
        artery    = 'Right Coronary Artery (usually)'
        key_leads = ['II', 'III', 'aVF', 'I', 'aVL']
        action    = 'Cardiology review. LPFB is uncommon — rule out inferior MI.'
        differentials = ['Inferior MI', 'Right ventricular hypertrophy', 'Normal variant']
        interp    = (
            f'Inferior lead attention ({inf_imp*100:.1f}%) with conduction pattern suggests '
            f'possible LPFB. Right axis deviation with narrow QRS would confirm.'
        )
        evidence.append(EvidenceItem(
            'III', f'Inferior attention ({inf_imp*100:.1f}%)',
            'Inferior axis shift — posterior fascicular block pattern', inf_imp, True
        ))

    else:
        name      = 'Non-specific Intraventricular Conduction Delay (IVCD)'
        territory = 'Intraventricular conduction — non-specific delay'
        artery    = 'N/A — non-specific pattern'
        key_leads = top_n_leads(lead_importance, 3)
        action    = 'Cardiology review if new. May reflect cardiomyopathy or fibrosis.'
        differentials = ['Incomplete LBBB', 'Incomplete RBBB', 'Cardiomyopathy', 'Electrolyte disturbance']
        interp    = f'Conduction disturbance pattern without clear LBBB/RBBB morphology. Non-specific delay.'

    if not evidence:
        for l in key_leads[:2]:
            v = lead_importance.get(l, 0)
            evidence.append(EvidenceItem(
                l, f'Attention {v*100:.1f}%',
                'Lead relevant to conduction assessment', v, True
            ))

    conf_score = model_prob * 0.85
    if qrs and qrs >= 120:
        conf_score = min(0.95, conf_score * 1.3)
    confidence = 'High' if conf_score > 0.65 else ('Moderate' if conf_score > 0.40 else 'Low')

    return SubDiagnosis(
        superclass=CLASSES[3], name=name, confidence=confidence,
        confidence_score=conf_score, territory=territory, artery=artery,
        key_leads=key_leads, evidence=evidence, interpretation=interp,
        differentials=differentials, clinical_action=action,
    )


def _diagnose_hyp(lead_importance: dict,
                  metrics:          dict,
                  model_prob:       float) -> SubDiagnosis:
    """Map HYP superclass to specific hypertrophy type."""

    lat_imp  = group_importance(lead_importance, LOW_LAT_LEADS)
    sep_imp  = group_importance(lead_importance, SEPTAL_LEADS)
    inf_imp  = group_importance(lead_importance, INFERIOR_LEADS)

    evidence = []

    if lat_imp > sep_imp:
        name      = 'Left Ventricular Hypertrophy (LVH)'
        territory = 'Left ventricle — thickened walls, increased mass'
        artery    = 'N/A — structural diagnosis'
        key_leads = ['V5', 'V6', 'I', 'aVL', 'V1', 'V2']
        action    = 'Echocardiogram to confirm. Assess for underlying hypertension or aortic stenosis.'
        differentials = ['Athletic heart', 'Hypertrophic cardiomyopathy (HCM)', 'Aortic stenosis']
        interp    = (
            f'Lateral lead attention ({lat_imp*100:.1f}%) consistent with LVH voltage pattern. '
            f'Sokolow-Lyon criteria: S(V1) + R(V5 or V6) > 35mm. '
            f'Repolarisation changes (strain pattern) in lateral leads are common with LVH.'
        )
        evidence += [
            EvidenceItem('V5', f'High attention ({lead_importance.get("V5",0)*100:.1f}%)',
                         'Tall R-wave voltage — LVH Sokolow-Lyon criterion', lat_imp, True),
            EvidenceItem('V1', f'Attention ({lead_importance.get("V1",0)*100:.1f}%)',
                         'Deep S-wave — LVH voltage criterion component', sep_imp, True),
        ]

    elif sep_imp > lat_imp:
        name      = 'Right Ventricular Hypertrophy (RVH)'
        territory = 'Right ventricle — increased right-sided pressure or volume'
        artery    = 'N/A — structural diagnosis'
        key_leads = ['V1', 'V2', 'III', 'aVF']
        action    = 'Echocardiogram. Check for pulmonary hypertension, COPD, or congenital heart disease.'
        differentials = ['Pulmonary hypertension', 'RBBB (mimics dominant R in V1)', 'Posterior MI']
        interp    = (
            f'Septal lead attention ({sep_imp*100:.1f}%) with right axis suggests RVH. '
            f'Dominant R in V1 (R>S) is the key criterion. Right axis deviation expected.'
        )
        evidence += [
            EvidenceItem('V1', f'High septal attention ({sep_imp*100:.1f}%)',
                         'Dominant R wave in V1 — RVH criterion', sep_imp, True),
            EvidenceItem('III', f'Attention ({lead_importance.get("III",0)*100:.1f}%)',
                         'Right axis deviation component', inf_imp, True),
        ]

    else:
        name      = 'Biventricular Hypertrophy (BVH)'
        territory = 'Both ventricles — combined hypertrophy'
        artery    = 'N/A — structural diagnosis'
        key_leads = top_n_leads(lead_importance, 4)
        action    = 'Echocardiogram. Associated with congenital heart disease or advanced cardiomyopathy.'
        differentials = ['LVH with RV strain', 'Congenital heart disease']
        interp    = 'Mixed hypertrophy pattern with both lateral and septal lead involvement.'
        evidence.append(EvidenceItem(
            None, f'Mixed territory involvement',
            'Lateral + septal attention suggests biventricular pattern', 0.5, True
        ))

    conf_score = model_prob * 0.75
    confidence = 'High' if conf_score > 0.60 else ('Moderate' if conf_score > 0.35 else 'Low')

    return SubDiagnosis(
        superclass=CLASSES[4], name=name, confidence=confidence,
        confidence_score=conf_score, territory=territory, artery=artery,
        key_leads=key_leads, evidence=evidence, interpretation=interp,
        differentials=differentials, clinical_action=action,
    )


def _diagnose_norm(lead_importance: dict,
                   metrics:          dict,
                   model_prob:       float) -> SubDiagnosis:
    """Map NORM superclass to specific rhythm."""

    hr  = metrics.get('hr')
    pr  = metrics.get('pr_interval')
    qrs = metrics.get('qrs_duration')
    qtc = metrics.get('qtc')

    evidence = []
    flags    = []

    if hr and hr > 100:
        name   = 'Sinus Tachycardia'
        interp = f'Heart rate {hr} bpm — sinus tachycardia. Evaluate for underlying cause.'
        action = 'Identify cause: pain, anxiety, fever, hypovolaemia, PE, drugs.'
        flags.append(f'HR {hr} bpm > 100')
    elif hr and hr < 60:
        name   = 'Sinus Bradycardia'
        interp = f'Heart rate {hr} bpm — sinus bradycardia. Can be normal in athletes.'
        action = 'Evaluate if symptomatic. Check medications (beta-blockers). Rule out heart block.'
        flags.append(f'HR {hr} bpm < 60')
    else:
        name   = 'Normal Sinus Rhythm'
        interp = 'Heart rate and conduction intervals within normal limits.'
        action = 'No immediate action required. Routine follow-up.'

    if pr and pr > 200: flags.append(f'PR {pr} ms (borderline 1st-degree block)')
    if qrs and qrs > 100: flags.append(f'QRS {qrs} ms (borderline wide)')
    if qtc and qtc > 450: flags.append(f'QTc {qtc} ms (prolonged)')

    if flags:
        for f in flags:
            evidence.append(EvidenceItem(
                None, f, 'Metric-based finding', 0.7, True
            ))
    else:
        evidence.append(EvidenceItem(
            None, f'All intervals normal',
            f'HR: {hr} bpm, PR: {pr} ms, QRS: {qrs} ms, QTc: {qtc} ms',
            0.9, True
        ))

    return SubDiagnosis(
        superclass=CLASSES[0],
        name=name,
        confidence='High' if model_prob > 0.75 else 'Moderate',
        confidence_score=model_prob,
        territory='Normal conduction system',
        artery='N/A',
        key_leads=top_n_leads(lead_importance, 2),
        evidence=evidence,
        interpretation=interp,
        differentials=['Early pathology not yet manifest', 'Normal variant'],
        clinical_action=action,
    )


# ── Main engine ────────────────────────────────────────────────────────────────

THRESHOLD = 0.35   # minimum probability to include a class in the report

def run_diagnosis_engine(predictions:     dict,
                          lead_importance: dict,
                          metrics:         dict) -> list[SubDiagnosis]:
    """
    Run the full diagnosis engine.

    Args:
        predictions:     {class_name: probability}  e.g. {'MI': 0.95, ...}
        lead_importance: {lead_name: fraction}       e.g. {'V3': 0.18, ...}
        metrics:         dict from report.extract_clinical_metrics()

    Returns:
        List of SubDiagnosis objects sorted by confidence_score descending.
        Only classes with probability >= THRESHOLD are included.
    """
    _metrics = {k: v for k, v in (metrics or {}).items()}
    results  = []

    dispatchers = {
        'MI':   _diagnose_mi,
        'STTC': _diagnose_sttc,
        'CD':   _diagnose_cd,
        'HYP':  _diagnose_hyp,
        'NORM': _diagnose_norm,
    }

    for cls, prob in predictions.items():
        if prob >= THRESHOLD and cls in dispatchers:
            diag = dispatchers[cls](lead_importance, _metrics, prob)
            results.append(diag)

    return sorted(results, key=lambda x: -x.confidence_score)
