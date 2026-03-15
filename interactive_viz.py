"""
interactive_viz.py — Interactive ECG visualization with anatomy + detailed diagnosis.

Renders a self-contained HTML component via Streamlit embedding:
  - Detailed diagnosis cards: sub-type, territory, artery, evidence chain
  - 12 clickable lead cards with mini ECG + importance bars
  - Full-resolution selected lead viewer with Grad-CAM overlay
  - Lead anatomy panel: territory, supplying artery, what the lead detects
  - Heart cross-section diagram: 4 territories, clickable to highlight leads

Usage (from app.py):
    from interactive_viz import render_interactive_ecg
    render_interactive_ecg(signal_np, heatmap, lead_importance,
                           predictions, top_cls, sub_diagnoses)
"""

import json
import numpy as np
import streamlit.components.v1 as components

from config import LEAD_NAMES, CLASSES, CLASS_NAMES, FS

# ── Lead anatomy knowledge base ───────────────────────────────────────────────

LEAD_ANATOMY = {
    'I':   {'territory':'High Lateral','wall':'High lateral wall of left ventricle',
             'artery':'Left Circumflex (LCx) or 1st Diagonal of LAD','group':'lateral','color':'#7C3AED',
             'detects':['High lateral ST changes and T-wave inversions','Left axis deviation (LAD)',
                        'High lateral MI pattern (often with aVL)','Left ventricular hypertrophy (voltage)']},
    'II':  {'territory':'Inferior','wall':'Inferior wall of left ventricle',
             'artery':'Right Coronary Artery (RCA) — dominant in 85% of people','group':'inferior','color':'#DC2626',
             'detects':['Inferior MI — primary diagnostic lead (with III + aVF)','Sinus rhythm and P-wave morphology',
                        'AV nodal conduction — PR interval measurement','Inferior ST elevation or reciprocal depression']},
    'III': {'territory':'Inferior','wall':'Inferior wall of left ventricle',
             'artery':'Right Coronary Artery (RCA)','group':'inferior','color':'#DC2626',
             'detects':['Inferior MI (confirmatory — with II + aVF)','Right axis deviation',
                        'Reciprocal changes during high lateral MI','RV involvement in inferior MI']},
    'aVR': {'territory':'Global / Endocardial','wall':'Inverted global view — endocardial perspective',
             'artery':'Left Main Coronary Artery or Proximal LAD','group':'global','color':'#6B7280',
             'detects':['Left main coronary artery occlusion (ST elevation in aVR)',
                        'Global subendocardial ischemia','Proximal LAD occlusion pattern',
                        'Antidromic tachycardia in WPW']},
    'aVL': {'territory':'High Lateral','wall':'High lateral wall of left ventricle',
             'artery':'Left Circumflex (LCx) or 1st Diagonal of LAD','group':'lateral','color':'#7C3AED',
             'detects':['High lateral MI (reciprocal to inferior changes)','Left axis deviation and fascicular blocks',
                        'LVH voltage criteria','Early marker of inferior STEMI (reciprocal depression)']},
    'aVF': {'territory':'Inferior','wall':'Inferior wall of left ventricle (diaphragmatic surface)',
             'artery':'Right Coronary Artery (RCA)','group':'inferior','color':'#DC2626',
             'detects':['Inferior MI — key confirmatory lead','Inferior ST elevation / posterior MI reciprocal changes',
                        'Left posterior fascicular block','Inferior wall motion abnormalities']},
    'V1':  {'territory':'Septal / Right Ventricular','wall':'Interventricular septum and anterior right ventricle',
             'artery':'Septal perforating branches of proximal LAD','group':'septal','color':'#D97706',
             'detects':["RBBB — rSR' pattern","LBBB — QS pattern",'Septal MI — loss of septal R-wave',
                        'RVH and strain','Brugada syndrome','WPW delta wave']},
    'V2':  {'territory':'Anteroseptal','wall':'Anterior septum and adjacent anterior LV wall',
             'artery':'Left Anterior Descending (LAD) — proximal','group':'septal','color':'#D97706',
             'detects':['Anteroseptal MI — often first lead to show change','Brugada pattern',
                        'R-wave progression loss','Proximal LAD occlusion']},
    'V3':  {'territory':'Anterior','wall':'Anterior wall of left ventricle (mid level)',
             'artery':'Left Anterior Descending (LAD) — mid','group':'anterior','color':'#2563EB',
             'detects':['Anterior MI — ST elevation V3-V4 = classic anterior STEMI','Mid-LAD occlusion territory',
                        'R/S transition zone (normal between V3-V4)','Anterior subendocardial ischemia']},
    'V4':  {'territory':'Anterior (Apical)','wall':'Anterior-apical wall of left ventricle',
             'artery':'Left Anterior Descending (LAD) — mid to distal','group':'anterior','color':'#2563EB',
             'detects':['Anterior MI — peak ST elevation often here','Apical ischemia',
                        'Left ventricular apical hypertrophy','Hyperkalemia (tall peaked T-waves)']},
    'V5':  {'territory':'Lateral (Low)','wall':'Low lateral wall of left ventricle',
             'artery':'Left Circumflex (LCx) or LAD Diagonal branch','group':'lateral','color':'#059669',
             'detects':['Lateral MI (with V6)','LVH: Sokolow-Lyon S(V1)+R(V5)>35mm',
                        'Lateral ST depression in subendocardial ischemia','LBBB Sgarbossa assessment']},
    'V6':  {'territory':'Lateral (Low)','wall':'Low lateral wall of left ventricle',
             'artery':'Left Circumflex (LCx)','group':'lateral','color':'#059669',
             'detects':['Lateral MI (with V5)','LBBB assessment','Lateral wall ischemia','Low lateral ST changes']},
}

TERRITORY_GROUPS = {
    'inferior': {'name':'Inferior Territory','leads':['II','III','aVF'],
                 'artery':'Right Coronary Artery (RCA)',
                 'condition':'Inferior STEMI / RCA occlusion',
                 'note':'ST elevation in II, III, aVF with reciprocal depression in I, aVL','color':'#DC2626'},
    'anterior': {'name':'Anterior Territory','leads':['V3','V4'],
                 'artery':'Left Anterior Descending (LAD) — mid',
                 'condition':'Anterior STEMI / LAD occlusion',
                 'note':'ST elevation in V1-V4; widest territory, most myocardium at risk','color':'#2563EB'},
    'septal':   {'name':'Septal Territory','leads':['V1','V2'],
                 'artery':'Proximal LAD / Septal perforators',
                 'condition':'Anteroseptal MI / Bundle branch blocks',
                 'note':'Changes here often indicate proximal LAD involvement','color':'#D97706'},
    'lateral':  {'name':'Lateral Territory','leads':['I','aVL','V5','V6'],
                 'artery':'Left Circumflex (LCx) or Diagonal',
                 'condition':'Lateral STEMI / LCx occlusion',
                 'note':'Often missed — LCx is the "silent" artery on ECG','color':'#7C3AED'},
    'global':   {'name':'Global / Right','leads':['aVR'],
                 'artery':'Left Main Coronary Artery',
                 'condition':'Left main disease / Global ischemia',
                 'note':'aVR elevation with diffuse ST depression = left main threat','color':'#6B7280'},
}


# ── Serialise SubDiagnosis objects ─────────────────────────────────────────────

def _serialise_diagnoses(sub_diagnoses) -> list[dict]:
    """Convert SubDiagnosis dataclass list to plain dicts for JSON serialisation."""
    if not sub_diagnoses:
        return []
    result = []
    for d in sub_diagnoses:
        result.append({
            'superclass':      d.superclass,
            'name':            d.name,
            'confidence':      d.confidence,
            'confidenceScore': d.confidence_score,
            'territory':       d.territory,
            'artery':          d.artery,
            'keyLeads':        d.key_leads,
            'interpretation':  d.interpretation,
            'differentials':   d.differentials,
            'clinicalAction':  d.clinical_action,
            'evidence': [
                {
                    'lead':       e.lead,
                    'factor':     e.factor,
                    'detail':     e.detail,
                    'weight':     e.weight,
                    'supporting': e.supporting,
                }
                for e in d.evidence
            ],
        })
    return result


# ── Main render function ───────────────────────────────────────────────────────

def render_interactive_ecg(
    signal_np:       np.ndarray,
    heatmap:         np.ndarray | None,
    lead_importance: dict,
    predictions:     dict,
    top_cls:         str,
    sub_diagnoses:   list | None = None,
    height:          int = 1600,
) -> None:
    """Render the full interactive ECG dashboard as a Streamlit HTML component."""

    step_mini = max(1, signal_np.shape[1] // 500)
    step_main = max(1, signal_np.shape[1] // 1000)
    sig_mini  = signal_np[:, ::step_mini].tolist()
    sig_main  = signal_np[:, ::step_main].tolist()

    if heatmap is not None:
        heat_mini = heatmap[:, ::step_mini].tolist()
        heat_main = heatmap[:, ::step_main].tolist()
    else:
        heat_mini = [[0.0] * len(sig_mini[0])] * 12
        heat_main = [[0.0] * len(sig_main[0])] * 12

    imp_list  = [float(lead_importance.get(l, 1/12)) for l in LEAD_NAMES]
    pred_list = [{'cls': cls, 'name': CLASS_NAMES[cls], 'prob': float(predictions.get(cls, 0))}
                 for cls in CLASSES]

    data = {
        'sigMini':    sig_mini,
        'sigMain':    sig_main,
        'heatMini':   heat_mini,
        'heatMain':   heat_main,
        'importance': imp_list,
        'predictions': pred_list,
        'topCls':     top_cls,
        'topClsName': CLASS_NAMES[top_cls],
        'leadNames':  LEAD_NAMES,
        'anatomy':    LEAD_ANATOMY,
        'territories': TERRITORY_GROUPS,
        'subDiagnoses': _serialise_diagnoses(sub_diagnoses or []),
        'fs':         FS,
    }
    data_json = json.dumps(data)

    conf_colors = {'High': '#10b981', 'Moderate': '#f59e0b', 'Low': '#ef4444'}
    cls_colors  = {'NORM': '#10b981', 'MI': '#ef4444',
                   'STTC': '#f59e0b', 'CD': '#8b5cf6', 'HYP': '#3b82f6'}

    html = f"""
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:transparent;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;color:#e2e8f0}}
#ev{{padding:12px}}

/* ── Diagnosis panel ── */
#ev-dx{{margin-bottom:14px}}
.ev-dx-card{{background:#1e293b;border:1.5px solid #334155;border-radius:10px;
             padding:14px;margin-bottom:10px;transition:border-color .2s}}
.ev-dx-card.high{{border-left:4px solid #10b981}}
.ev-dx-card.moderate{{border-left:4px solid #f59e0b}}
.ev-dx-card.low{{border-left:4px solid #ef4444}}
.ev-dx-header{{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px;flex-wrap:wrap;gap:6px}}
.ev-dx-title{{font-size:16px;font-weight:800}}
.ev-dx-badges{{display:flex;gap:6px;align-items:center;flex-wrap:wrap}}
.ev-badge{{font-size:10px;font-weight:700;padding:3px 8px;border-radius:10px}}
.ev-dx-grid{{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:10px}}
.ev-dx-field{{background:#0f172a;border-radius:6px;padding:8px}}
.ev-dx-label{{font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:#475569;margin-bottom:3px}}
.ev-dx-value{{font-size:11px;color:#cbd5e1;line-height:1.4}}
.ev-dx-value.artery{{color:#fbbf24}}
.ev-dx-interp{{font-size:11px;color:#94a3b8;line-height:1.6;margin-bottom:10px;
               background:#0f172a;border-radius:6px;padding:8px;border-left:2px solid #3b82f6}}

/* Evidence chain */
.ev-chain-title{{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;
                 color:#475569;margin-bottom:6px}}
.ev-chain{{display:flex;flex-direction:column;gap:4px;margin-bottom:10px}}
.ev-ev-item{{display:flex;align-items:flex-start;gap:8px;padding:5px 7px;border-radius:5px;
             background:#0f172a}}
.ev-ev-item.supporting{{border-left:2px solid #10b981}}
.ev-ev-item.opposing{{border-left:2px solid #ef4444}}
.ev-ev-lead{{font-size:10px;font-weight:800;min-width:28px;padding:2px 5px;
             border-radius:4px;text-align:center;flex-shrink:0}}
.ev-ev-body{{flex:1}}
.ev-ev-factor{{font-size:11px;font-weight:600;color:#e2e8f0}}
.ev-ev-detail{{font-size:10px;color:#64748b;line-height:1.3}}
.ev-ev-bar{{height:3px;border-radius:2px;margin-top:3px}}
.ev-key-leads{{display:flex;gap:5px;flex-wrap:wrap;margin-bottom:10px}}
.ev-key-lead-pill{{font-size:10px;font-weight:700;padding:2px 8px;border-radius:10px;
                   background:#1e3a5f;color:#93c5fd;border:1px solid #3b82f666}}
.ev-diff{{margin-bottom:6px}}
.ev-diff-title{{font-size:10px;font-weight:700;color:#475569;text-transform:uppercase;
                letter-spacing:.07em;margin-bottom:4px}}
.ev-diff-list{{display:flex;gap:5px;flex-wrap:wrap}}
.ev-diff-pill{{font-size:10px;padding:2px 8px;border-radius:8px;background:#1e293b;
               color:#94a3b8;border:1px solid #334155}}
.ev-action{{background:#0f1f10;border:1px solid #16a34a44;border-radius:6px;padding:7px 10px;
            font-size:11px;color:#86efac;display:flex;gap:7px;align-items:flex-start}}

/* Diagnosis banner */
#ev-diag{{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:14px;align-items:center}}
.ev-pred{{padding:5px 11px;border-radius:20px;font-size:11px;font-weight:600;
          border:1.5px solid transparent}}
.ev-pred.top{{border-color:#fff;transform:scale(1.06)}}
.ev-primary-label{{font-size:13px;font-weight:700;color:#f1f5f9}}

/* Lead grid */
#ev-grid{{display:grid;grid-template-columns:repeat(6,1fr);gap:6px;margin-bottom:14px}}
.ev-card{{background:#1e293b;border:1.5px solid #334155;border-radius:8px;padding:6px;
          cursor:pointer;transition:all .2s}}
.ev-card:hover{{border-color:#60a5fa;background:#1e3a5f}}
.ev-card.selected{{border-color:#60a5fa;background:#1e3a5f;box-shadow:0 0 0 2px #3b82f680}}
.ev-card.key-lead{{border-color:#10b981 !important}}
.ev-card-name{{font-size:11px;font-weight:700;margin-bottom:3px;
               display:flex;justify-content:space-between;align-items:center}}
.ev-card canvas{{width:100%;height:36px;display:block}}
.ev-imp-bar-wrap{{height:4px;background:#334155;border-radius:2px;margin-top:4px;overflow:hidden}}
.ev-imp-bar{{height:100%;border-radius:2px}}
.ev-heat-badge{{font-size:9px;background:#ef444422;color:#fca5a5;border-radius:3px;padding:1px 4px}}

/* Main ECG + info */
#ev-body{{display:flex;gap:12px;margin-bottom:14px}}
#ev-ecg-wrap{{flex:0 0 58%;background:#1e293b;border:1px solid #334155;border-radius:10px;padding:10px}}
#ev-ecg-header{{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px}}
#ev-ecg-title{{font-size:14px;font-weight:700;color:#93c5fd}}
#ev-ecg-controls{{display:flex;gap:10px;font-size:12px;color:#94a3b8}}
#ev-ecg-controls label{{display:flex;align-items:center;gap:4px;cursor:pointer}}
#ev-main-canvas{{width:100%;height:180px;display:block;border-radius:6px}}
#ev-heat-bar{{height:6px;border-radius:3px;background:linear-gradient(to right,#1e293b,#ef4444);margin-top:4px;opacity:0.7}}
#ev-heat-label{{font-size:10px;color:#94a3b8;display:flex;justify-content:space-between;margin-top:2px}}

#ev-info{{flex:1;background:#1e293b;border:1px solid #334155;border-radius:10px;
          padding:12px;overflow-y:auto;max-height:280px}}
#ev-info-lead{{font-size:18px;font-weight:800;margin-bottom:4px}}
#ev-info-territory{{display:inline-block;font-size:11px;font-weight:600;
                    padding:3px 10px;border-radius:12px;margin-bottom:10px}}
.ev-info-row{{margin-bottom:10px}}
.ev-info-label{{font-size:10px;font-weight:700;text-transform:uppercase;
                letter-spacing:.05em;color:#64748b;margin-bottom:3px}}
.ev-info-value{{font-size:12px;color:#cbd5e1;line-height:1.5}}
.ev-detects{{list-style:none;padding:0}}
.ev-detects li{{font-size:11px;color:#94a3b8;padding:2px 0 2px 14px;position:relative;line-height:1.4}}
.ev-detects li::before{{content:"•";position:absolute;left:4px;color:#60a5fa}}
.ev-group-badge{{display:inline-flex;align-items:center;gap:5px;background:#0f172a;
                 border-radius:6px;padding:5px 8px;margin-top:6px;font-size:11px}}

/* Heart diagram */
#ev-heart{{background:#1e293b;border:1px solid #334155;border-radius:10px;padding:12px}}
#ev-heart-title{{font-size:13px;font-weight:700;color:#94a3b8;margin-bottom:10px;text-align:center}}
#ev-heart-wrap{{display:flex;gap:16px;align-items:center;justify-content:center;flex-wrap:wrap}}
#ev-heart-svg-wrap{{flex:0 0 340px}}
#ev-heart-legend{{display:grid;grid-template-columns:1fr 1fr;gap:8px;flex:1;min-width:200px}}
.ev-territory-card{{background:#0f172a;border:1.5px solid #334155;border-radius:8px;
                    padding:8px;cursor:pointer;transition:all .2s}}
.ev-territory-card:hover,.ev-territory-card.active{{background:#1e3a5f;border-color:#60a5fa}}
.ev-tc-header{{display:flex;align-items:center;gap:6px;margin-bottom:4px}}
.ev-tc-dot{{width:10px;height:10px;border-radius:50%;flex-shrink:0}}
.ev-tc-name{{font-size:11px;font-weight:700}}
.ev-tc-leads{{font-size:10px;color:#94a3b8;margin-bottom:2px}}
.ev-tc-artery{{font-size:10px;color:#64748b;font-style:italic}}
</style>

<div id="ev">

  <!-- ── Diagnosis panel ── -->
  <div id="ev-dx"></div>

  <!-- ── Prediction badges ── -->
  <div id="ev-diag">
    <span class="ev-primary-label" id="ev-primary-label"></span>
  </div>

  <!-- ── Lead grid ── -->
  <div id="ev-grid"></div>

  <!-- ── Main ECG + lead info ── -->
  <div id="ev-body">
    <div id="ev-ecg-wrap">
      <div id="ev-ecg-header">
        <span id="ev-ecg-title">Lead I</span>
        <div id="ev-ecg-controls">
          <label><input type="checkbox" id="ev-grad-toggle" checked> Grad-CAM</label>
        </div>
      </div>
      <canvas id="ev-main-canvas"></canvas>
      <div id="ev-heat-bar"></div>
      <div id="ev-heat-label"><span>low attention</span><span>high attention</span></div>
    </div>
    <div id="ev-info">
      <div id="ev-info-lead"></div>
      <div id="ev-info-territory"></div>
      <div id="ev-info-wall"  class="ev-info-row"></div>
      <div id="ev-info-artery" class="ev-info-row"></div>
      <div id="ev-info-detects" class="ev-info-row"></div>
      <div id="ev-info-group" class="ev-info-row"></div>
    </div>
  </div>

  <!-- ── Heart territory diagram ── -->
  <div id="ev-heart">
    <div id="ev-heart-title">Cardiac Territory Map — click a region to highlight its leads</div>
    <div id="ev-heart-wrap">
      <div id="ev-heart-svg-wrap"></div>
      <div id="ev-heart-legend"></div>
    </div>
  </div>

</div>

<script>
const D = {data_json};
let selectedLead = 0;
let showHeat = true;

const CLS_COLORS = {{NORM:'#10b981',MI:'#ef4444',STTC:'#f59e0b',CD:'#8b5cf6',HYP:'#3b82f6'}};
const CONF_COLORS = {{High:'#10b981',Moderate:'#f59e0b',Low:'#ef4444'}};

function clamp(v,a,b){{return Math.max(a,Math.min(b,v));}}

/* ── ECG canvas ── */
function drawECG(canvas, signal, heat, showH){{
  if(!canvas) return;
  const dpr=window.devicePixelRatio||1;
  if(canvas.dataset.sized!=='1'){{
    canvas.width=canvas.offsetWidth*dpr;
    canvas.height=canvas.offsetHeight*dpr;
    canvas.dataset.sized='1';
  }}
  const ctx=canvas.getContext('2d');
  const w=canvas.offsetWidth, h=canvas.offsetHeight;
  ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.clearRect(0,0,w,h);
  ctx.fillStyle='#0f172a'; ctx.fillRect(0,0,w,h);
  const n=signal.length; if(n<2) return;
  if(showH&&heat){{
    for(let i=0;i<n;i++){{
      const x=(i/(n-1))*w, nx=((i+1)/(n-1))*w;
      const a=clamp(heat[i],0,1)*0.65;
      if(a>0.02){{ctx.fillStyle=`rgba(239,68,68,${{a}})`;ctx.fillRect(x,0,nx-x+1,h);}}
    }}
  }}
  ctx.strokeStyle='#1e3a5f';ctx.lineWidth=0.5;
  for(let gx=0;gx<=w;gx+=w/10){{ctx.beginPath();ctx.moveTo(gx,0);ctx.lineTo(gx,h);ctx.stroke();}}
  for(let gy=0;gy<=h;gy+=h/4){{ctx.beginPath();ctx.moveTo(0,gy);ctx.lineTo(w,gy);ctx.stroke();}}
  const mn=Math.min(...signal),mx=Math.max(...signal),rng=mx-mn||1,pad=8;
  ctx.beginPath();ctx.strokeStyle='#60a5fa';ctx.lineWidth=1.4;ctx.lineJoin='round';
  for(let i=0;i<n;i++){{
    const x=(i/(n-1))*w, y=h-pad-((signal[i]-mn)/rng)*(h-pad*2);
    i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
  }}
  ctx.stroke();
}}

/* ── Diagnosis panel ── */
function buildDiagPanel(){{
  const wrap=document.getElementById('ev-dx');
  if(!D.subDiagnoses||D.subDiagnoses.length===0) return;

  D.subDiagnoses.forEach(diag=>{{
    const col=CLS_COLORS[diag.superclass]||'#6b7280';
    const cCol=CONF_COLORS[diag.confidence]||'#6b7280';
    const card=document.createElement('div');
    card.className=`ev-dx-card ${{diag.confidence.toLowerCase()}}`;

    /* Key leads pills */
    const keyPills=(diag.keyLeads||[]).map(l=>
      `<span class="ev-key-lead-pill" onclick="selectLeadByName('${{l}}')" style="cursor:pointer">${{l}}</span>`
    ).join('');

    /* Evidence items */
    const evItems=(diag.evidence||[]).map(e=>{{
      const leadBg=e.lead ? (D.anatomy[e.lead]?.color||'#6b7280')+'33' : '#33415533';
      const leadCol=e.lead ? (D.anatomy[e.lead]?.color||'#6b7280') : '#6b7280';
      const supCol=e.supporting?'#10b981':'#ef4444';
      const barW=Math.round(e.weight*100);
      return `<div class="ev-ev-item ${{e.supporting?'supporting':'opposing'}}"
                   onclick="${{e.lead?`selectLeadByName('${{e.lead}}')`:''}}"
                   style="${{e.lead?'cursor:pointer':''}}">
        <span class="ev-ev-lead" style="background:${{leadBg}};color:${{leadCol}}">
          ${{e.lead||'—'}}
        </span>
        <div class="ev-ev-body">
          <div class="ev-ev-factor">${{e.factor}}</div>
          <div class="ev-ev-detail">${{e.detail}}</div>
          <div class="ev-ev-bar" style="width:${{barW}}%;background:${{supCol}}44;
               border:0.5px solid ${{supCol}}66"></div>
        </div>
      </div>`;
    }}).join('');

    /* Differentials */
    const diffPills=(diag.differentials||[]).map(d=>
      `<span class="ev-diff-pill">${{d}}</span>`
    ).join('');

    card.innerHTML=`
      <div class="ev-dx-header">
        <div class="ev-dx-title" style="color:${{col}}">${{diag.name}}</div>
        <div class="ev-dx-badges">
          <span class="ev-badge" style="background:${{cCol}}22;color:${{cCol}};border:1px solid ${{cCol}}44">
            ${{diag.confidence}} confidence
          </span>
          <span class="ev-badge" style="background:${{col}}22;color:${{col}};border:1px solid ${{col}}44">
            ${{diag.superclass}}
          </span>
          <span class="ev-badge" style="background:#1e293b;color:#94a3b8;border:1px solid #334155">
            ${{(D.predictions.find(p=>p.cls===diag.superclass)?.prob*100||0).toFixed(1)}}%
          </span>
        </div>
      </div>

      <div class="ev-dx-grid">
        <div class="ev-dx-field">
          <div class="ev-dx-label">Territory</div>
          <div class="ev-dx-value">${{diag.territory}}</div>
        </div>
        <div class="ev-dx-field">
          <div class="ev-dx-label">Implicated Artery</div>
          <div class="ev-dx-value artery">${{diag.artery}}</div>
        </div>
      </div>

      <div class="ev-dx-interp">${{diag.interpretation}}</div>

      <div style="margin-bottom:8px">
        <div class="ev-chain-title">Key Leads</div>
        <div class="ev-key-leads">${{keyPills}}</div>
      </div>

      <div class="ev-chain-title">Evidence Chain
        <span style="font-weight:400;color:#64748b;font-size:9px;margin-left:6px">
          green = supporting · red = opposing · click lead to inspect
        </span>
      </div>
      <div class="ev-chain">${{evItems}}</div>

      ${{diffPills?`
      <div class="ev-diff">
        <div class="ev-diff-title">Differential Diagnoses</div>
        <div class="ev-diff-list">${{diffPills}}</div>
      </div>`:''}}`

      if(diag.clinicalAction){{
        card.innerHTML+=`
        <div class="ev-action">
          <span style="color:#4ade80;font-size:13px">⚕</span>
          <span>${{diag.clinicalAction}}</span>
        </div>`;
      }}

    wrap.appendChild(card);
  }});
}}

/* ── Prediction badges ── */
function buildDiagBanner(){{
  const wrap=document.getElementById('ev-diag');
  document.getElementById('ev-primary-label').textContent=`Primary: ${{D.topClsName}}`;
  const sorted=[...D.predictions].sort((a,b)=>b.prob-a.prob);
  sorted.forEach(p=>{{
    const col=CLS_COLORS[p.cls]||'#6b7280';
    const b=document.createElement('div');
    b.className='ev-pred'+(p.cls===D.topCls?' top':'');
    b.style.background=col+'22'; b.style.color=col;
    if(p.cls===D.topCls) b.style.borderColor=col;
    b.textContent=`${{p.name}}: ${{(p.prob*100).toFixed(1)}}%`;
    wrap.appendChild(b);
  }});
}}

/* ── Lead grid ── */
function buildLeadGrid(){{
  const grid=document.getElementById('ev-grid');
  const keyLeads=new Set((D.subDiagnoses||[]).flatMap(d=>d.keyLeads||[]));
  D.leadNames.forEach((name,i)=>{{
    const anat=D.anatomy[name];
    const imp=D.importance[i];
    const isKey=keyLeads.has(name);
    const col=anat.color;
    const card=document.createElement('div');
    card.className='ev-card'+(i===0?' selected':'')+(isKey?' key-lead':'');
    card.id=`ev-card-${{i}}`;
    card.innerHTML=`
      <div class="ev-card-name" style="color:${{col}}">
        ${{name}}${{isKey?' <span style="color:#10b981;font-size:9px">★</span>':''}}
        <span class="ev-heat-badge">${{(imp*100).toFixed(0)}}%</span>
      </div>
      <canvas id="ev-mini-${{i}}" style="height:36px"></canvas>
      <div class="ev-imp-bar-wrap">
        <div class="ev-imp-bar" style="width:${{imp*100}}%;background:${{col}}"></div>
      </div>`;
    card.addEventListener('click',()=>selectLead(i));
    grid.appendChild(card);
    requestAnimationFrame(()=>{{
      const c=document.getElementById(`ev-mini-${{i}}`);
      if(c){{ c.style.width='100%'; c.style.height='36px';
               drawECG(c, D.sigMini[i], D.heatMini[i], showHeat); }}
    }});
  }});
}}

function selectLeadByName(name){{
  const idx=D.leadNames.indexOf(name);
  if(idx>=0) selectLead(idx);
}}

function selectLead(idx){{
  document.querySelectorAll('.ev-card').forEach((c,i)=>c.classList.toggle('selected',i===idx));
  selectedLead=idx;
  renderMainECG();
  renderLeadInfo(idx);
}}

function renderMainECG(){{
  const c=document.getElementById('ev-main-canvas');
  c.style.width='100%'; c.style.height='180px'; c.dataset.sized='';
  drawECG(c, D.sigMain[selectedLead], D.heatMain[selectedLead], showHeat);
  document.getElementById('ev-ecg-title').textContent=
    `Lead ${{D.leadNames[selectedLead]}} — ${{D.anatomy[D.leadNames[selectedLead]].territory}}`;
}}

function renderLeadInfo(idx){{
  const name=D.leadNames[idx];
  const anat=D.anatomy[name];
  const imp=D.importance[idx];
  const col=anat.color;
  const grp=D.territories[anat.group]||{{}};

  document.getElementById('ev-info-lead').innerHTML=
    `<span style="color:${{col}}">Lead ${{name}}</span>
     <span style="font-size:12px;color:#64748b;font-weight:400;margin-left:8px">
       Importance: ${{(imp*100).toFixed(1)}}%
     </span>`;
  const terr=document.getElementById('ev-info-territory');
  terr.textContent=anat.territory;
  terr.style.background=col+'22'; terr.style.color=col;

  document.getElementById('ev-info-wall').innerHTML=
    `<div class="ev-info-label">Heart Wall</div><div class="ev-info-value">${{anat.wall}}</div>`;
  document.getElementById('ev-info-artery').innerHTML=
    `<div class="ev-info-label">Supplying Artery</div>
     <div class="ev-info-value" style="color:#fbbf24">${{anat.artery}}</div>`;
  document.getElementById('ev-info-detects').innerHTML=
    `<div class="ev-info-label">This lead detects</div>
     <ul class="ev-detects">${{anat.detects.map(d=>`<li>${{d}}</li>`).join('')}}</ul>`;

  if(grp.name){{
    document.getElementById('ev-info-group').innerHTML=
      `<div class="ev-group-badge" style="border:1px solid ${{grp.color||col}}40">
         <span style="width:8px;height:8px;border-radius:50%;background:${{grp.color||col}};display:inline-block"></span>
         <span style="color:${{grp.color||col}};font-weight:700">${{grp.name}}</span>
         <span style="color:#64748b">leads: ${{(grp.leads||[]).join(', ')}}</span>
       </div>
       ${{grp.note?`<div style="font-size:10px;color:#64748b;margin-top:4px;padding-left:2px">${{grp.note}}</div>`:''}}`;
  }}
}}

/* ── Heart diagram ── */
function buildHeartDiagram(){{
  const svgWrap=document.getElementById('ev-heart-svg-wrap');
  const legend=document.getElementById('ev-heart-legend');
  const cx=170,cy=130,outerR=100,innerR=38;
  function pxy(r,deg){{const a=(deg-90)*Math.PI/180;return[cx+r*Math.cos(a),cy+r*Math.sin(a)];}}
  function seg(a1,a2){{
    const[x1,y1]=pxy(outerR,a1),[x2,y2]=pxy(outerR,a2);
    const[x3,y3]=pxy(innerR,a2),[x4,y4]=pxy(innerR,a1);
    const la=(a2-a1)>180?1:0;
    return `M${{x1}},${{y1}} A${{outerR}},${{outerR}} 0 ${{la}},1 ${{x2}},${{y2}} L${{x3}},${{y3}} A${{innerR}},${{innerR}} 0 ${{la}},0 ${{x4}},${{y4}} Z`;
  }}
  const segs=[
    {{id:'anterior',a1:310,a2:50, color:'#2563EB',label:'Anterior', sub:'V3, V4'}},
    {{id:'septal',  a1:50, a2:130,color:'#D97706',label:'Septal',   sub:'V1, V2'}},
    {{id:'inferior',a1:130,a2:230,color:'#DC2626',label:'Inferior', sub:'II, III, aVF'}},
    {{id:'lateral', a1:230,a2:310,color:'#7C3AED',label:'Lateral',  sub:'I, aVL, V5, V6'}},
  ];
  const paths=segs.map(s=>{{
    const[lx,ly]=pxy((outerR+innerR)/2,(s.a1+s.a2)/2);
    return `<path id="seg-${{s.id}}" d="${{seg(s.a1,s.a2)}}" fill="${{s.color}}33"
                  stroke="${{s.color}}" stroke-width="1.5" style="cursor:pointer;transition:fill .2s"
                  onmouseenter="this.style.fill='${{s.color}}66'"
                  onmouseleave="this.style.fill='${{s.color}}33'"
                  onclick="clickTerritory('${{s.id}}')"/>
            <text x="${{lx}}" y="${{ly-6}}" text-anchor="middle" fill="${{s.color}}"
                  font-size="9" font-weight="700" pointer-events="none">${{s.label}}</text>
            <text x="${{lx}}" y="${{ly+6}}" text-anchor="middle" fill="${{s.color}}cc"
                  font-size="8" pointer-events="none">${{s.sub}}</text>`;
  }}).join('');
  svgWrap.innerHTML=`
    <svg width="340" height="260" viewBox="0 0 340 260" xmlns="http://www.w3.org/2000/svg">
      <circle cx="${{cx}}" cy="${{cy}}" r="${{outerR+8}}" fill="#0f172a" stroke="#334155" stroke-width="1"/>
      ${{paths}}
      <circle cx="${{cx}}" cy="${{cy}}" r="${{innerR}}" fill="#0f172a" stroke="#475569" stroke-width="1.5"/>
      <text x="${{cx}}" y="${{cy-6}}" text-anchor="middle" fill="#94a3b8" font-size="10" font-weight="600">LV</text>
      <text x="${{cx}}" y="${{cy+8}}" text-anchor="middle" fill="#64748b" font-size="8">cavity</text>
      <text x="300" y="55" text-anchor="middle" fill="#6B7280" font-size="9" font-weight="700">aVR</text>
      <text x="300" y="67" text-anchor="middle" fill="#6B7280" font-size="8">Global</text>
      <line x1="280" y1="62" x2="${{cx+outerR+8}}" y2="${{cy-40}}"
            stroke="#6B728066" stroke-width="1" stroke-dasharray="3,3"/>
      <text x="${{cx}}" y="240" text-anchor="middle" fill="#475569" font-size="9">
        Short-axis cross-section (anterior view)
      </text>
    </svg>`;
  Object.entries(D.territories).forEach(([key,grp])=>{{
    if(key==='global') return;
    const card=document.createElement('div');
    card.className='ev-territory-card'; card.id=`tc-${{key}}`;
    card.innerHTML=`
      <div class="ev-tc-header">
        <div class="ev-tc-dot" style="background:${{grp.color}}"></div>
        <div class="ev-tc-name" style="color:${{grp.color}}">${{grp.name}}</div>
      </div>
      <div class="ev-tc-leads">Leads: ${{grp.leads.join(', ')}}</div>
      <div class="ev-tc-artery">${{grp.artery}}</div>`;
    card.addEventListener('click',()=>clickTerritory(key));
    legend.appendChild(card);
  }});
}}

function clickTerritory(key){{
  const grp=D.territories[key]; if(!grp) return;
  const firstIdx=D.leadNames.indexOf(grp.leads[0]);
  if(firstIdx>=0) selectLead(firstIdx);
  document.querySelectorAll('.ev-territory-card').forEach(c=>c.classList.remove('active'));
  const tc=document.getElementById(`tc-${{key}}`);
  if(tc) tc.classList.add('active');
  grp.leads.forEach(l=>{{
    const idx=D.leadNames.indexOf(l);
    const card=document.getElementById(`ev-card-${{idx}}`);
    if(card){{
      const orig=card.style.borderColor;
      card.style.borderColor=grp.color;
      setTimeout(()=>{{card.style.borderColor=orig;}},1500);
    }}
  }});
}}

/* ── Boot ── */
document.addEventListener('DOMContentLoaded',()=>{{
  buildDiagPanel();
  buildDiagBanner();
  buildLeadGrid();
  buildHeartDiagram();
  requestAnimationFrame(()=>{{ renderMainECG(); renderLeadInfo(0); }});

  document.getElementById('ev-grad-toggle').addEventListener('change',function(){{
    showHeat=this.checked;
    renderMainECG();
    D.leadNames.forEach((_,i)=>{{
      const c=document.getElementById(`ev-mini-${{i}}`);
      if(c) drawECG(c, D.sigMini[i], D.heatMini[i], showHeat);
    }});
  }});
}});
</script>
"""

    components.html(html, height=height, scrolling=True)