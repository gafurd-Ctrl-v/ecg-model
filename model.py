"""
model.py — ECGAttentionNet architecture.

All hyperparameters (BASE_CH, N_HEADS, DROPOUT, NUM_CLASSES) are
imported from config.py. Nothing is hardcoded here.

Architecture:
  Input (B, 12, 5000)
  → Shared ResNet-1D backbone (3 residual blocks, BASE_CH=32 → 256)
  → Global avg pool per lead
  → Learnable lead positional encoding
  → Multi-Head Self-Attention (N_HEADS=4) over 12 leads
  → LayerNorm + Feed-Forward + LayerNorm
  → Mean pool over leads → Dense(5) classifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import NUM_CLASSES, BASE_CH, N_HEADS, DROPOUT


# ── Building blocks ────────────────────────────────────────────────────────────

class ResidualBlock1D(nn.Module):
    """1-D residual block with optional downsampling skip connection."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dropout: float = DROPOUT):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch,  out_ch, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=7, stride=1,      padding=3, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.drop  = nn.Dropout(dropout)
        self.skip  = (
            nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch)
            )
            if stride != 1 or in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.skip(x))


class ECGResNetBackbone(nn.Module):
    """Shared 1-D ResNet processing a single ECG lead.

    Input:  (N, 1, 5000)   where N = B × 12
    Output: (N, BASE_CH*8, T')  — feature maps (Grad-CAM hooks layer3).

    Channel progression with BASE_CH=32:
      stem   → (N, 32, 1250)
      layer1 → (N,  64, 625)
      layer2 → (N, 128, 313)
      layer3 → (N, 256, 157)   ← Grad-CAM target
    """

    def __init__(self, base_ch: int = BASE_CH):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, base_ch, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(base_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = ResidualBlock1D(base_ch,     base_ch * 2, stride=2)
        self.layer2 = ResidualBlock1D(base_ch * 2, base_ch * 4, stride=2)
        self.layer3 = ResidualBlock1D(base_ch * 4, base_ch * 8, stride=2)
        self.out_channels = base_ch * 8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


# ── Main model ─────────────────────────────────────────────────────────────────

class ECGAttentionNet(nn.Module):
    """
    Full ECG classification model.

    All architecture hyperparameters default to values in config.py.
    You can override them at instantiation, but the recommended workflow
    is to change config.py and leave this call unchanged.
    """

    def __init__(self,
                 num_classes: int   = NUM_CLASSES,
                 base_ch:     int   = BASE_CH,
                 nhead:       int   = N_HEADS,
                 dropout:     float = DROPOUT):
        super().__init__()
        self.num_leads = 12
        self.backbone  = ECGResNetBackbone(base_ch)
        d_model        = self.backbone.out_channels   # BASE_CH * 8

        # Learnable positional encoding — one vector per lead
        self.lead_pos_enc = nn.Parameter(torch.randn(1, 12, d_model) * 0.02)

        # Multi-head self-attention over the 12 leads
        self.attn = nn.MultiheadAttention(
            embed_dim   = d_model,
            num_heads   = nhead,
            dropout     = dropout,
            batch_first = True
        )
        self.norm1 = nn.LayerNorm(d_model)

        # Position-wise feed-forward (2× expansion)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

        # Attention weights cached after each forward pass (for explainability)
        self._attn_weights: torch.Tensor | None = None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 12, 5000)
        Returns:
            logits: (B, num_classes) — raw logits; apply sigmoid for probabilities.
        """
        B, L, T = x.shape

        # CNN per lead (shared weights)
        x    = x.reshape(B * L, 1, T)
        feat = self.backbone(x).mean(dim=-1)   # (B×12, d_model)
        feat = feat.reshape(B, L, -1)           # (B, 12, d_model)

        # Lead positional encoding
        feat = feat + self.lead_pos_enc

        # Multi-head attention
        attn_out, attn_w = self.attn(feat, feat, feat,
                                     need_weights=True,
                                     average_attn_weights=True)
        self._attn_weights = attn_w.detach().cpu()   # (B, 12, 12)

        feat = self.norm1(feat + attn_out)
        feat = self.norm2(feat + self.ff(feat))
        feat = feat.mean(dim=1)   # (B, d_model)
        return self.head(feat)

    def get_attention_weights(self) -> torch.Tensor | None:
        """Returns (B, 12, 12) attention matrix from the last forward pass."""
        return self._attn_weights
