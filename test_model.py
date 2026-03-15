import torch
from model import ECGAttentionNet

model = ECGAttentionNet()
dummy = torch.randn(2, 12, 5000)
out   = model(dummy)
attn  = model.get_attention_weights()
n     = sum(p.numel() for p in model.parameters() if p.requires_grad)

print('Output shape:   ', out.shape)
print('Attention shape:', attn.shape)
print('Parameters:     ', f'{n:,}')
