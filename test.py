import torch
from PoPE_pytorch import PoPE, flash_attn_with_pope

# pope

pope = PoPE(dim = 32, heads = 8).cuda()

# queries, keys, values for attention

q = torch.randn(1, 8, 1024, 64).cuda()
k = torch.randn(1, 8, 1024, 64).cuda()
v = torch.randn(1, 8, 1024, 64).cuda()

pope_emb = pope(1024)

out = flash_attn_with_pope(q, k, v, pope = pope_emb, causal = True)

assert out.shape == (1, 8, 1024, 64)