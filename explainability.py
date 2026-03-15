"""
explainability.py — Grad-CAM temporal heatmaps + Attention Rollout lead scores.

All constants (LEAD_NAMES, SIGNAL_LEN, TOP_K_LEADS) imported from config.py.
"""

import numpy as np
import torch
import torch.nn.functional as F

from config import LEAD_NAMES, SIGNAL_LEN, TOP_K_LEADS
from model  import ECGAttentionNet


# ── Grad-CAM ──────────────────────────────────────────────────────────────────

class GradCAM1D:
    """Gradient-weighted Class Activation Maps for 1-D ECG signals.

    Hooks backbone.layer3 to produce a (12, SIGNAL_LEN) temporal heatmap
    showing *when* in the recording each lead was most informative.

    Usage:
        explainer = GradCAM1D(model)
        heatmap   = explainer.generate(signal_tensor, class_idx=1)   # 1 = MI
        # heatmap.shape == (12, 5000), values in [0, 1]
    """

    def __init__(self, model: ECGAttentionNet):
        self.model        = model
        self._acts: dict  = {}
        self._grads: dict = {}
        self._register_hooks()

    def _register_hooks(self):
        target = self.model.backbone.layer3

        def fwd_hook(module, inp, out):
            self._acts['layer3'] = out

        def bwd_hook(module, grad_in, grad_out):
            self._grads['layer3'] = grad_out[0]

        target.register_forward_hook(fwd_hook)
        target.register_full_backward_hook(bwd_hook)

    def generate(self,
                 signal:    torch.Tensor,
                 class_idx: int,
                 device:    torch.device | None = None) -> np.ndarray:
        """Compute per-lead Grad-CAM heatmap.

        Args:
            signal:    (1, 12, 5000) tensor — single sample.
            class_idx: Target class index (0=NORM,1=MI,2=STTC,3=CD,4=HYP).
        Returns:
            heatmap: (12, SIGNAL_LEN) float32 array in [0, 1].
        """
        if device is None:
            device = next(self.model.parameters()).device

        self.model.eval()
        signal = signal.to(device).detach().requires_grad_(True)
        _, L, T = signal.shape

        logits = self.model(signal)
        self.model.zero_grad()
        logits[0, class_idx].backward()

        acts    = self._acts['layer3']               # (12, d, T')
        grads   = self._grads['layer3']              # (12, d, T')
        weights = grads.mean(dim=-1, keepdim=True)   # (12, d, 1)
        cam     = F.relu((weights * acts).sum(dim=1))# (12, T')

        # Upsample to original signal length (from config.SIGNAL_LEN)
        cam = F.interpolate(cam.unsqueeze(0), size=SIGNAL_LEN,
                            mode='linear', align_corners=False).squeeze(0)
        cam = cam.detach().cpu().numpy()             # (12, SIGNAL_LEN)

        for i in range(L):
            lo, hi = cam[i].min(), cam[i].max()
            cam[i] = (cam[i] - lo) / (hi - lo) if hi > lo else np.zeros_like(cam[i])

        return cam


# ── Attention Rollout ─────────────────────────────────────────────────────────

def compute_lead_importance(model:     ECGAttentionNet,
                             signal:   torch.Tensor,
                             class_idx: int | None = None) -> dict:
    """Per-lead importance from the stored attention weight matrix (B, 12, 12).

    Column-aggregates the attention matrix: how much was each lead
    attended to across all query leads.

    Returns:
        dict mapping lead name → importance fraction (sums to 1.0).
    """
    model.eval()
    with torch.no_grad():
        model(signal)

    attn = model.get_attention_weights()

    if attn is None:
        uniform = 1.0 / len(LEAD_NAMES)
        return {name: uniform for name in LEAD_NAMES}

    importance = attn[0].numpy().mean(axis=0)   # (12,)
    importance = importance / importance.sum()

    return {LEAD_NAMES[i]: float(importance[i]) for i in range(len(LEAD_NAMES))}


def top_k_leads(lead_importance: dict,
                k: int = TOP_K_LEADS) -> list[tuple[str, float]]:
    """Return the k leads with highest importance (k defaults to config.TOP_K_LEADS)."""
    return sorted(lead_importance.items(), key=lambda x: -x[1])[:k]
