import pytest
import torch
from torch import allclose

from PoPE_pytorch.pope import PoPE
from PoPE_pytorch.attention import flash_attn_with_pope

param = pytest.mark.parametrize

@pytest.fixture(scope="module")
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
@param('latent_strategy', ('start', 'end', 'random', 'none', 'all'))
@param('is_causal', (True, False))
@param('gqa', (True, False))
def test_mixed_positions_edge_cases(
    device,
    latent_strategy,
    is_causal,
    gqa
):
    seq_len = 128
    q_heads = 8
    kv_heads = 2 if gqa else 8
    dim = 64
    rotate_dim = 32

    pope = PoPE(dim=rotate_dim, heads=q_heads).to(device)

    num_latents = {
        'none': 0,
        'all': seq_len,
        'start': 4,
        'end': 4,
        'random': 10
    }[latent_strategy]

    num_pope_tokens = seq_len - num_latents
    pos_emb = pope(num_pope_tokens) if num_pope_tokens > 0 else pope(1)

    if latent_strategy == 'none':
        pos_indices = torch.arange(seq_len, device = device)
    elif latent_strategy == 'all':
        pos_indices = None
    elif latent_strategy == 'start':
        pos_indices = torch.arange(num_latents, seq_len, device = device)
    elif latent_strategy == 'end':
        pos_indices = torch.arange(seq_len - num_latents, device = device)
    elif latent_strategy == 'random':
        pos_indices = torch.randperm(seq_len, device = device)[:num_pope_tokens].sort()[0]

    q = torch.randn(2, q_heads, seq_len, dim, device=device)
    k = torch.randn(2, kv_heads, seq_len, dim, device=device)
    v = torch.randn(2, kv_heads, seq_len, dim, device=device)

    # triton pass
    out_triton = flash_attn_with_pope(
        q, k, v,
        pos_emb=pos_emb,
        pope_pos_emb_indices=pos_indices,
        causal=is_causal,
        fused=True,
        head_dimension_at_first=True
    )

    # reference pytorch pass
    out_ref = flash_attn_with_pope(
        q, k, v,
        pos_emb=pos_emb,
        pope_pos_emb_indices=pos_indices,
        causal=is_causal,
        fused=False,
        head_dimension_at_first=True
    )

    # assert equality
    assert allclose(out_triton, out_ref, atol=5e-3)
