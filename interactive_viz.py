"""
interactive_viz.py — Interactive ECG visualization with cardiac anatomy context.

Renders a self-contained HTML component via Streamlit embedding containing:
  - 12 clickable lead cards with mini ECG traces + importance bars
  - Full-resolution selected lead viewer with Grad-CAM overlay
  - Lead anatomy panel: cardiac territory, supplying artery, what the lead detects
  - Heart cross-section diagram: 4 coloured territories, clickable to select lead groups
  - Diagnosis confidence breakdown at the top

Usage (from app.py):
    from interactive_viz import render_interactive_ecg
    render_interactive_ecg(signal_np, heatmap, lead_importance, predictions, top_cls)
"""

import json
import numpy as np
import streamlit.components.v1 as components

from config import LEAD_NAMES, CLASSES, CLASS_NAMES, FS

# ── Lead anatomy knowledge base ───────────────────────────────────────────────
# Each entry describes what that electrode "sees" and what it is sensitive to.
# Source: standard 12-lead ECG interpretation (Dubin, Goldberger, Wagner).

LEAD_ANATOMY = {
    'I': {
        'territory': 'High Lateral',
        'wall':      'High lateral wall of left ventricle',
        'artery':    'Left Circumflex (LCx) or 1st Diagonal of LAD',
        'group':     'lateral',
        'color':     '#7C3AED',
        'detects': [
            'High lateral ST changes and T-wave inversions',
            'Left axis deviation (LAD)',
            'High lateral MI pattern (often with aVL)',
            'Left ventricular hypertrophy (voltage)',
        ],
    },
    'II': {
        'territory': 'Inferior',
        'wall':      'Inferior wall of left ventricle',
        'artery':    'Right Coronary Artery (RCA) — dominant in 85% of people',
        'group':     'inferior',
        'color':     '#DC2626',
        'detects': [
            'Inferior MI — primary diagnostic lead (with III + aVF)',
            'Sinus rhythm assessment and P-wave morphology',
            'AV nodal conduction — PR interval measurement',
            'Inferior ST elevation or reciprocal depression',
        ],
    },
    'III': {
        'territory': 'Inferior',
        'wall':      'Inferior wall of left ventricle',
        'artery':    'Right Coronary Artery (RCA)',
        'group':     'inferior',
        'color':     '#DC2626',
        'detects': [
            'Inferior MI (confirmatory — with II + aVF)',
            'Right axis deviation',
            'Reciprocal changes during high lateral MI',
            'Right ventricular involvement in inferior MI',
        ],
    },
    'aVR': {
        'territory': 'Global / Endocardial',
        'wall':      'Inverted global view — endocardial perspective',
        'artery':    'Left Main Coronary Artery or Proximal LAD',
        'group':     'global',
        'color':     '#6B7280',
        'detects': [
            'Left main coronary artery occlusion (ST elevation in aVR)',
            'Global subendocardial ischemia (diffuse ST depression elsewhere)',
            'Proximal LAD occlusion pattern',
            'Antidromic tachycardia in Wolff-Parkinson-White (WPW)',
        ],
    },
    'aVL': {
        'territory': 'High Lateral',
        'wall':      'High lateral wall of left ventricle',
        'artery':    'Left Circumflex (LCx) or 1st Diagonal of LAD',
        'group':     'lateral',
        'color':     '#7C3AED',
        'detects': [
            'High lateral MI (reciprocal to inferior changes)',
            'Left axis deviation and fascicular blocks',
            'Left ventricular hypertrophy (voltage criteria)',
            'Early marker of inferior STEMI (reciprocal depression)',
        ],
    },
    'aVF': {
        'territory': 'Inferior',
        'wall':      'Inferior wall of left ventricle (diaphragmatic surface)',
        'artery':    'Right Coronary Artery (RCA)',
        'group':     'inferior',
        'color':     '#DC2626',
        'detects': [
            'Inferior MI — key confirmatory lead',
            'Inferior ST elevation / posterior MI reciprocal changes',
            'Left posterior fascicular block',
            'Inferior wall motion abnormalities',
        ],
    },
    'V1': {
        'territory': 'Septal / Right Ventricular',
        'wall':      'Interventricular septum and anterior right ventricle',
        'artery':    'Septal perforating branches of proximal LAD',
        'group':     'septal',
        'color':     '#D97706',
        'detects': [
            'Right bundle branch block (RBBB) — rSR\' pattern',
            'Left bundle branch block (LBBB) — QS pattern',
            'Septal MI — loss of septal R-wave',
            'Right ventricular hypertrophy and strain (dominant R)',
            'Brugada syndrome (coved ST elevation V1-V2)',
            'Wolff-Parkinson-White delta wave',
        ],
    },
    'V2': {
        'territory': 'Anteroseptal',
        'wall':      'Anterior septum and adjacent anterior LV wall',
        'artery':    'Left Anterior Descending (LAD) — proximal',
        'group':     'septal',
        'color':     '#D97706',
        'detects': [
            'Anteroseptal MI — often first lead to show change',
            'Brugada pattern (ST elevation V1-V2 with RBBB morphology)',
            'R-wave progression loss (poor R-wave progression = anterior scar)',
            'Proximal LAD occlusion pattern',
        ],
    },
    'V3': {
        'territory': 'Anterior',
        'wall':      'Anterior wall of left ventricle (mid level)',
        'artery':    'Left Anterior Descending (LAD) — mid',
        'group':     'anterior',
        'color':     '#2563EB',
        'detects': [
            'Anterior MI — ST elevation V3-V4 = classic anterior STEMI',
            'Mid-LAD occlusion territory',
            'R/S transition zone (normal between V3-V4)',
            'Anterior subendocardial ischemia',
        ],
    },
    'V4': {
        'territory': 'Anterior (Apical)',
        'wall':      'Anterior-apical wall of left ventricle',
        'artery':    'Left Anterior Descending (LAD) — mid to distal',
        'group':     'anterior',
        'color':     '#2563EB',
        'detects': [
            'Anterior MI — peak ST elevation often here in anterior STEMI',
            'Apical ischemia and wall motion abnormality',
            'Left ventricular apical hypertrophy',
            'Hyperkalemia (tall peaked T-waves)',
        ],
    },
    'V5': {
        'territory': 'Lateral (Low)',
        'wall':      'Low lateral wall of left ventricle',
        'artery':    'Left Circumflex (LCx) or LAD Diagonal branch',
        'group':     'lateral',
        'color':     '#059669',
        'detects': [
            'Lateral MI (with V6)',
            'LVH by Sokolow-Lyon voltage: S(V1) + R(V5) > 35mm',
            'Lateral ST depression in subendocardial ischemia',
            'LBBB concordant/discordant ST assessment (Smith-modified Sgarbossa)',
        ],
    },
    'V6': {
        'territory': 'Lateral (Low)',
        'wall':      'Low lateral wall of left ventricle',
        'artery':    'Left Circumflex (LCx)',
        'group':     'lateral',
        'color':     '#059669',
        'detects': [
            'Lateral MI (with V5)',
            'Left bundle branch block assessment',
            'Lateral wall ischemia extending from anterolateral MI',
            'Low lateral ST changes',
        ],
    },
}

# Territory groups — what happens when multiple leads in a group change together
TERRITORY_GROUPS = {
    'inferior': {
        'name':      'Inferior Territory',
        'leads':     ['II', 'III', 'aVF'],
        'artery':    'Right Coronary Artery (RCA)',
        'condition': 'Inferior STEMI / RCA occlusion',
        'note':      'ST elevation in II, III, aVF with reciprocal depression in I, aVL',
        'color':     '#DC2626',
    },
    'anterior': {
        'name':      'Anterior Territory',
        'leads':     ['V3', 'V4'],
        'artery':    'Left Anterior Descending (LAD) — mid',
        'condition': 'Anterior STEMI / LAD occlusion',
        'note':      'ST elevation in V1-V4; widest territory, most myocardium at risk',
        'color':     '#2563EB',
    },
    'septal': {
        'name':      'Septal Territory',
        'leads':     ['V1', 'V2'],
        'artery':    'Proximal LAD / Septal perforators',
        'condition': 'Anteroseptal MI / Bundle branch blocks',
        'note':      'Changes here often indicate proximal LAD involvement',
        'color':     '#D97706',
    },
    'lateral': {
        'name':      'Lateral Territory',
        'leads':     ['I', 'aVL', 'V5', 'V6'],
        'artery':    'Left Circumflex (LCx) or Diagonal',
        'condition': 'Lateral STEMI / LCx occlusion',
        'note':      'Often missed — LCx is the "silent" artery on ECG',
        'color':     '#7C3AED',
    },
    'global': {
        'name':      'Global / Right',
        'leads':     ['aVR'],
        'artery':    'Left Main Coronary Artery',
        'condition': 'Left main disease / Global ischemia',
        'note':      'aVR elevation with diffuse ST depression = left main threat',
        'color':     '#6B7280',
    },
}


# ── HTML generation ───────────────────────────────────────────────────────────

def render_interactive_ecg(
    signal_np:       np.ndarray,
    heatmap:         np.ndarray | None,
    lead_importance: dict,
    predictions:     dict,
    top_cls:         str,
    height:          int = 1180,
) -> None:
    """
    Render the interactive ECG dashboard as a Streamlit HTML component.

    Args:
        signal_np:       (12, 5000) normalised ECG array.
        heatmap:         (12, 5000) Grad-CAM values in [0,1], or None.
        lead_importance: {lead_name: fraction} from attention rollout.
        predictions:     {class_name: probability}.
        top_cls:         Primary predicted class key (e.g. 'MI').
        height:          Iframe height in pixels.
    """
    # ── Prepare data for JSON ─────────────────────────────────────────────────
    # Downsample to 500 pts for mini cards (every 10th), 1000 pts for main view
    step_mini = max(1, signal_np.shape[1] // 500)
    step_main = max(1, signal_np.shape[1] // 1000)

    sig_mini  = signal_np[:, ::step_mini].tolist()
    sig_main  = signal_np[:, ::step_main].tolist()

    if heatmap is not None:
        heat_mini = heatmap[:, ::step_mini].tolist()
        heat_main = heatmap[:, ::step_main].tolist()
    else:
        n_mini = len(sig_mini[0])
        n_main = len(sig_main[0])
        heat_mini = [[0.0] * n_mini] * 12
        heat_main = [[0.0] * n_main] * 12

    imp_list  = [float(lead_importance.get(l, 1/12)) for l in LEAD_NAMES]
    pred_list = [
        {'cls': cls, 'name': CLASS_NAMES[cls], 'prob': float(predictions.get(cls, 0))}
        for cls in CLASSES
    ]
    anatomy_json  = json.dumps(LEAD_ANATOMY)
    territory_json = json.dumps(TERRITORY_GROUPS)

    data = {
        'sigMini':   sig_mini,
        'sigMain':   sig_main,
        'heatMini':  heat_mini,
        'heatMain':  heat_main,
        'importance': imp_list,
        'predictions': pred_list,
        'topCls':    top_cls,
        'topClsName': CLASS_NAMES[top_cls],
        'leadNames': LEAD_NAMES,
        'anatomy':   LEAD_ANATOMY,
        'territories': TERRITORY_GROUPS,
        'fs':        FS,
    }
    data_json = json.dumps(data)

    html = f"""
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:transparent;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;color:#e2e8f0}}
#ev{{padding:12px;max-width:100%;overflow-x:hidden}}

/* ── Diagnosis banner ── */
#ev-diag{{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:14px;align-items:center}}
.ev-pred{{padding:6px 12px;border-radius:20px;font-size:12px;font-weight:600;border:1.5px solid transparent;white-space:nowrap}}
.ev-pred.top{{border-color:#fff;transform:scale(1.06)}}
.ev-primary-label{{font-size:13px;font-weight:700;margin-right:4px;color:#f1f5f9}}

/* ── Lead grid ── */
#ev-grid{{display:grid;grid-template-columns:repeat(6,1fr);gap:6px;margin-bottom:14px}}
.ev-card{{background:#1e293b;border:1.5px solid #334155;border-radius:8px;padding:6px;cursor:pointer;transition:all .2s;position:relative;overflow:hidden}}
.ev-card:hover{{border-color:#60a5fa;background:#1e3a5f}}
.ev-card.selected{{border-color:#60a5fa;background:#1e3a5f;box-shadow:0 0 0 2px #3b82f680}}
.ev-card-name{{font-size:11px;font-weight:700;margin-bottom:3px;display:flex;justify-content:space-between;align-items:center}}
.ev-card canvas{{width:100%;height:36px;display:block}}
.ev-imp-bar-wrap{{height:4px;background:#334155;border-radius:2px;margin-top:4px;overflow:hidden}}
.ev-imp-bar{{height:100%;border-radius:2px;transition:width .3s}}
.ev-heat-badge{{font-size:9px;background:#ef444422;color:#fca5a5;border-radius:3px;padding:1px 4px}}

/* ── Body: main canvas + info panel ── */
#ev-body{{display:flex;gap:12px;margin-bottom:14px}}
#ev-ecg-wrap{{flex:0 0 58%;background:#1e293b;border:1px solid #334155;border-radius:10px;padding:10px}}
#ev-ecg-header{{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px}}
#ev-ecg-title{{font-size:14px;font-weight:700;color:#93c5fd}}
#ev-ecg-controls{{display:flex;gap:10px;align-items:center;font-size:12px;color:#94a3b8}}
#ev-ecg-controls label{{display:flex;align-items:center;gap:4px;cursor:pointer}}
#ev-main-canvas{{width:100%;height:180px;display:block;border-radius:6px}}
#ev-heat-bar{{height:6px;border-radius:3px;background:linear-gradient(to right,#1e293b,#ef4444);margin-top:4px;opacity:0.7}}
#ev-heat-label{{font-size:10px;color:#94a3b8;display:flex;justify-content:space-between;margin-top:2px}}

/* ── Lead info panel ── */
#ev-info{{flex:1;background:#1e293b;border:1px solid #334155;border-radius:10px;padding:12px;overflow-y:auto;max-height:280px}}
#ev-info-lead{{font-size:18px;font-weight:800;margin-bottom:4px}}
#ev-info-territory{{display:inline-block;font-size:11px;font-weight:600;padding:3px 10px;border-radius:12px;margin-bottom:10px}}
.ev-info-row{{margin-bottom:10px}}
.ev-info-label{{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.05em;color:#64748b;margin-bottom:3px}}
.ev-info-value{{font-size:12px;color:#cbd5e1;line-height:1.5}}
.ev-detects{{list-style:none;padding:0}}
.ev-detects li{{font-size:11px;color:#94a3b8;padding:2px 0 2px 14px;position:relative;line-height:1.4}}
.ev-detects li::before{{content:"•";position:absolute;left:4px;color:#60a5fa}}
.ev-group-badge{{display:inline-flex;align-items:center;gap:5px;background:#0f172a;border-radius:6px;padding:5px 8px;margin-top:6px;font-size:11px}}

/* ── Heart diagram ── */
#ev-heart{{background:#1e293b;border:1px solid #334155;border-radius:10px;padding:12px}}
#ev-heart-title{{font-size:13px;font-weight:700;color:#94a3b8;margin-bottom:10px;text-align:center}}
#ev-heart-wrap{{display:flex;gap:16px;align-items:center;justify-content:center;flex-wrap:wrap}}
#ev-heart-svg-wrap{{flex:0 0 340px}}
#ev-heart-legend{{display:grid;grid-template-columns:1fr 1fr;gap:8px;flex:1;min-width:200px}}
.ev-territory-card{{background:#0f172a;border:1.5px solid #334155;border-radius:8px;padding:8px;cursor:pointer;transition:all .2s}}
.ev-territory-card:hover,.ev-territory-card.active{{background:#1e3a5f;border-color:#60a5fa}}
.ev-tc-header{{display:flex;align-items:center;gap:6px;margin-bottom:4px}}
.ev-tc-dot{{width:10px;height:10px;border-radius:50%;flex-shrink:0}}
.ev-tc-name{{font-size:11px;font-weight:700}}
.ev-tc-leads{{font-size:10px;color:#94a3b8;margin-bottom:2px}}
.ev-tc-artery{{font-size:10px;color:#64748b;font-style:italic}}
</style>

<div id="ev">
  <div id="ev-diag">
    <span id="ev-primary-label" class="ev-primary-label"></span>
  </div>

  <div id="ev-grid"></div>

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

/* ── Utility ── */
function clamp(v,a,b){{return Math.max(a,Math.min(b,v));}}
function hexToRgb(hex){{
  const r=parseInt(hex.slice(1,3),16),g=parseInt(hex.slice(3,5),16),b=parseInt(hex.slice(5,7),16);
  return [r,g,b];
}}

/* ── ECG rendering ── */
function drawECG(canvas, signal, heat, showH, bgColor){{
  const W = canvas.width, H = canvas.height;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio||1;
  if(canvas.dataset.sized!=='1'){{
    canvas.width  = canvas.offsetWidth  * dpr;
    canvas.height = canvas.offsetHeight * dpr;
    canvas.dataset.sized='1';
    ctx.scale(dpr,dpr);
  }}
  const w = canvas.offsetWidth, h = canvas.offsetHeight;

  ctx.clearRect(0,0,w,h);
  ctx.fillStyle = bgColor||'#0f172a';
  ctx.fillRect(0,0,w,h);

  const n = signal.length;
  if(n<2) return;

  /* Grad-CAM bands */
  if(showH && heat){{
    for(let i=0;i<n;i++){{
      const x = (i/(n-1))*w;
      const nxt = ((i+1)/(n-1))*w;
      const alpha = clamp(heat[i],0,1)*0.65;
      if(alpha>0.02){{
        ctx.fillStyle=`rgba(239,68,68,${{alpha}})`;
        ctx.fillRect(x,0,nxt-x+1,h);
      }}
    }}
  }}

  /* Light grid lines */
  ctx.strokeStyle='#1e3a5f';
  ctx.lineWidth=0.5;
  for(let gx=0;gx<=w;gx+=w/10){{ctx.beginPath();ctx.moveTo(gx,0);ctx.lineTo(gx,h);ctx.stroke();}}
  for(let gy=0;gy<=h;gy+=h/4){{ctx.beginPath();ctx.moveTo(0,gy);ctx.lineTo(w,gy);ctx.stroke();}}

  /* Signal */
  const mn=Math.min(...signal), mx=Math.max(...signal);
  const rng = mx-mn||1;
  const pad=8;
  ctx.beginPath();
  ctx.strokeStyle='#60a5fa';
  ctx.lineWidth=1.4;
  ctx.lineJoin='round';
  for(let i=0;i<n;i++){{
    const x=(i/(n-1))*w;
    const y=h-pad-((signal[i]-mn)/rng)*(h-pad*2);
    i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
  }}
  ctx.stroke();
}}

/* ── Diagnosis banner ── */
function buildDiagBanner(){{
  const wrap=document.getElementById('ev-diag');
  const sorted=[...D.predictions].sort((a,b)=>b.prob-a.prob);
  const primaryLabel=document.getElementById('ev-primary-label');
  primaryLabel.textContent=`Primary: ${{D.topClsName}}`;
  sorted.forEach(p=>{{
    const pct=(p.prob*100).toFixed(1);
    const isTop=p.cls===D.topCls;
    const anat=Object.values(D.anatomy).find(a=>true); /* just get any color */
    /* assign color per class */
    const clsColors={{NORM:'#10b981',MI:'#ef4444',STTC:'#f59e0b',CD:'#8b5cf6',HYP:'#3b82f6'}};
    const col=clsColors[p.cls]||'#6b7280';
    const badge=document.createElement('div');
    badge.className='ev-pred'+(isTop?' top':'');
    badge.style.background=col+'22';
    badge.style.color=col;
    badge.style.borderColor=isTop?col:'transparent';
    badge.textContent=`${{p.name}}: ${{pct}}%`;
    wrap.appendChild(badge);
  }});
}}

/* ── Lead grid ── */
function buildLeadGrid(){{
  const grid=document.getElementById('ev-grid');
  D.leadNames.forEach((name,i)=>{{
    const anat=D.anatomy[name];
    const imp=D.importance[i];
    const maxHeat=D.heatMini[i]?Math.max(...D.heatMini[i]):0;
    const col=anat.color;

    const card=document.createElement('div');
    card.className='ev-card'+(i===0?' selected':'');
    card.id=`ev-card-${{i}}`;
    card.innerHTML=`
      <div class="ev-card-name" style="color:${{col}}">
        ${{name}}
        <span class="ev-heat-badge">${{(imp*100).toFixed(0)}}%</span>
      </div>
      <canvas id="ev-mini-${{i}}" style="height:36px"></canvas>
      <div class="ev-imp-bar-wrap">
        <div class="ev-imp-bar" style="width:${{imp*100}}%;background:${{col}}"></div>
      </div>`;
    card.addEventListener('click',()=>selectLead(i));
    grid.appendChild(card);

    /* Render mini ECG after DOM insertion */
    requestAnimationFrame(()=>{{
      const c=document.getElementById(`ev-mini-${{i}}`);
      if(c){{
        c.style.width='100%';
        c.style.height='36px';
        drawECG(c, D.sigMini[i], D.heatMini[i], showHeat, '#0f172a');
      }}
    }});
  }});
}}

/* ── Select a lead ── */
function selectLead(idx){{
  document.querySelectorAll('.ev-card').forEach((c,i)=>c.classList.toggle('selected',i===idx));
  selectedLead=idx;
  renderMainECG();
  renderLeadInfo(idx);
}}

/* ── Main ECG canvas ── */
function renderMainECG(){{
  const c=document.getElementById('ev-main-canvas');
  c.style.width='100%';
  c.style.height='180px';
  c.dataset.sized='';   /* force resize */
  drawECG(c, D.sigMain[selectedLead], D.heatMain[selectedLead], showHeat, '#0f172a');
  document.getElementById('ev-ecg-title').textContent=
    `Lead ${{D.leadNames[selectedLead]}} — ${{D.anatomy[D.leadNames[selectedLead]].territory}}`;
}}

/* ── Lead info panel ── */
function renderLeadInfo(idx){{
  const name=D.leadNames[idx];
  const anat=D.anatomy[name];
  const imp=D.importance[idx];
  const col=anat.color;
  const grpKey=anat.group;
  const grp=D.territories[grpKey]||{{}};

  document.getElementById('ev-info-lead').innerHTML=
    `<span style="color:${{col}}">Lead ${{name}}</span>
     <span style="font-size:12px;color:#64748b;font-weight:400;margin-left:8px">
       Importance: ${{(imp*100).toFixed(1)}}%
     </span>`;

  document.getElementById('ev-info-territory').textContent=anat.territory;
  document.getElementById('ev-info-territory').style.background=col+'22';
  document.getElementById('ev-info-territory').style.color=col;

  document.getElementById('ev-info-wall').innerHTML=
    `<div class="ev-info-label">Heart Wall</div>
     <div class="ev-info-value">${{anat.wall}}</div>`;

  document.getElementById('ev-info-artery').innerHTML=
    `<div class="ev-info-label">Supplying Artery</div>
     <div class="ev-info-value" style="color:#fbbf24">${{anat.artery}}</div>`;

  const detectItems=anat.detects.map(d=>`<li>${{d}}</li>`).join('');
  document.getElementById('ev-info-detects').innerHTML=
    `<div class="ev-info-label">This lead detects</div>
     <ul class="ev-detects">${{detectItems}}</ul>`;

  if(grp.name){{
    const groupLeads=(grp.leads||[]).join(', ');
    document.getElementById('ev-info-group').innerHTML=
      `<div class="ev-group-badge" style="border:1px solid ${{grp.color||col}}40">
        <span style="width:8px;height:8px;border-radius:50%;background:${{grp.color||col}};display:inline-block"></span>
        <span style="color:${{grp.color||col}};font-weight:700">${{grp.name}}</span>
        <span style="color:#64748b">leads: ${{groupLeads}}</span>
       </div>
       ${{grp.note?`<div style="font-size:10px;color:#64748b;margin-top:4px;padding-left:2px">${{grp.note}}</div>`:''}}`;
  }}
}}

/* ── Heart diagram ── */
function buildHeartDiagram(){{
  const svgWrap=document.getElementById('ev-heart-svg-wrap');
  const legend=document.getElementById('ev-heart-legend');

  /* SVG cross-section */
  const cx=170,cy=130,outerR=100,innerR=38;

  function polarXY(r,deg){{
    const rad=(deg-90)*Math.PI/180;
    return [cx+r*Math.cos(rad), cy+r*Math.sin(rad)];
  }}
  function segPath(a1,a2){{
    const [x1,y1]=polarXY(outerR,a1);
    const [x2,y2]=polarXY(outerR,a2);
    const [x3,y3]=polarXY(innerR,a2);
    const [x4,y4]=polarXY(innerR,a1);
    const la=(a2-a1)>180?1:0;
    return `M${{x1}},${{y1}} A${{outerR}},${{outerR}} 0 ${{la}},1 ${{x2}},${{y2}} L${{x3}},${{y3}} A${{innerR}},${{innerR}} 0 ${{la}},0 ${{x4}},${{y4}} Z`;
  }}

  /* Segments: Anterior(top), Lateral(left), Inferior(bottom), Septal(right) */
  const segments=[
    {{id:'anterior', a1:310, a2:50,  color:'#2563EB', label:'Anterior',  sub:'V3, V4',  note:'LAD'}},
    {{id:'septal',   a1:50,  a2:130, color:'#D97706', label:'Septal',    sub:'V1, V2',  note:'Prox. LAD'}},
    {{id:'inferior', a1:130, a2:230, color:'#DC2626', label:'Inferior',  sub:'II, III, aVF', note:'RCA'}},
    {{id:'lateral',  a1:230, a2:310, color:'#7C3AED', label:'Lateral',   sub:'I, aVL, V5, V6', note:'LCx'}},
  ];

  const paths=segments.map(s=>{{
    const [lx,ly]=polarXY((outerR+innerR)/2, (s.a1+s.a2)/2);
    return `<path id="seg-${{s.id}}" d="${{segPath(s.a1,s.a2)}}"
              fill="${{s.color}}33" stroke="${{s.color}}" stroke-width="1.5"
              style="cursor:pointer;transition:fill .2s"
              onmouseenter="this.style.fill='${{s.color}}66'"
              onmouseleave="highlightTerritory('${{s.id}}')"
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
      <!-- aVR label outside -->
      <text x="300" y="55" text-anchor="middle" fill="#6B7280" font-size="9" font-weight="700">aVR</text>
      <text x="300" y="67" text-anchor="middle" fill="#6B7280" font-size="8">Global</text>
      <line x1="280" y1="62" x2="${{cx+outerR+8}}" y2="${{cy-40}}" stroke="#6B728066" stroke-width="1" stroke-dasharray="3,3"/>
      <text x="${{cx}}" y="240" text-anchor="middle" fill="#475569" font-size="9">Short-axis cross-section (anterior view)</text>
    </svg>`;

  /* Territory legend cards */
  Object.entries(D.territories).forEach(([key,grp])=>{{
    if(key==='global') return;
    const card=document.createElement('div');
    card.className='ev-territory-card';
    card.id=`tc-${{key}}`;
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

function highlightTerritory(id){{
  /* reset fill on segment to match selected state */
  const D_terr={{anterior:'#2563EB',septal:'#D97706',inferior:'#DC2626',lateral:'#7C3AED'}};
  const el=document.getElementById(`seg-${{id}}`);
  if(el) el.style.fill=D_terr[id]+'33';
}}

function clickTerritory(key){{
  /* Highlight all lead cards in this territory */
  const grp=D.territories[key];
  if(!grp) return;
  const firstIdx=D.leadNames.indexOf(grp.leads[0]);
  if(firstIdx>=0) selectLead(firstIdx);
  /* Highlight territory cards */
  document.querySelectorAll('.ev-territory-card').forEach(c=>c.classList.remove('active'));
  const tc=document.getElementById(`tc-${{key}}`);
  if(tc) tc.classList.add('active');
  /* Flash all lead cards in the group */
  grp.leads.forEach(l=>{{
    const idx=D.leadNames.indexOf(l);
    const card=document.getElementById(`ev-card-${{idx}}`);
    if(card){{
      card.style.borderColor=grp.color;
      setTimeout(()=>{{ if(card) card.style.borderColor=''; }},1500);
    }}
  }});
}}

/* ── Heatmap toggle ── */
document.addEventListener('DOMContentLoaded',()=>{{
  buildDiagBanner();
  buildLeadGrid();
  buildHeartDiagram();

  requestAnimationFrame(()=>{{
    renderMainECG();
    renderLeadInfo(0);
  }});

  document.getElementById('ev-grad-toggle').addEventListener('change',function(){{
    showHeat=this.checked;
    renderMainECG();
    /* Redraw all mini cards */
    D.leadNames.forEach((_,i)=>{{
      const c=document.getElementById(`ev-mini-${{i}}`);
      if(c) drawECG(c, D.sigMini[i], D.heatMini[i], showHeat, '#0f172a');
    }});
  }});
}});
</script>
"""

    components.html(html, height=height, scrolling=True)
