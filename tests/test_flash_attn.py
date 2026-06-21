import pytest
param = pytest.mark.parametrize

import torch
from torch import allclose

from PoPE_pytorch.pope import PoPE
from PoPE_pytorch.attention import flash_attn_with_pope

# helper

def exists(v):
    return v is not None

# test config

@pytest.fixture(scope="module")
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# tests

@pytest.mark.skipif(not torch.cuda.is_available(), reason = 'CUDA not available')
@param('seq_len', (128, 256))
@param('is_causal', (True, False))
@param('gqa', (True, False))
@param('mixed_pos', (True, False))
def test_flash_attn_with_pope(
    device,
    seq_len,
    is_causal,
    gqa,
    mixed_pos
):
    q_heads = 8
    kv_heads = 2 if gqa else 8
    dim = 64
    rotate_dim = 32

    pope = PoPE(dim = rotate_dim, heads = q_heads).to(device)

    # mixed positions
    # simulate some unrotated tokens

    num_pope_tokens = (seq_len // 2) if mixed_pos else seq_len
    pos_emb = pope(num_pope_tokens)

    # random pos indices for mixed pos
    pos_indices = None
    if mixed_pos:
        pos_indices = torch.randperm(seq_len, device = device)[:num_pope_tokens].sort()[0]

    # q, k, v
    q = torch.randn(1, q_heads, seq_len, dim, device = device)
    k = torch.randn(1, kv_heads, seq_len, dim, device = device)
    v = torch.randn(1, kv_heads, seq_len, dim, device = device)

    # triton pass

    out_triton = flash_attn_with_pope(
        q, k, v,
        pos_emb = pos_emb,
        pope_pos_emb_indices = pos_indices,
        causal = is_causal,
        fused = True,
        head_dimension_at_first = True
    )

    # reference pytorch pass

    out_ref = flash_attn_with_pope(
        q, k, v,
        pos_emb = pos_emb,
        pope_pos_emb_indices = pos_indices,
        causal = is_causal,
        fused = False,
        head_dimension_at_first = True
    )

    # assert equality
    
    assert allclose(out_triton, out_ref, atol = 5e-3)

@pytest.mark.skipif(not torch.cuda.is_available(), reason = 'CUDA not available')
def test_flash_attn_mask():
    device = torch.device('cuda')

    q = torch.randn(1, 4, 128, 64, device = device)
    k = torch.randn(1, 4, 128, 64, device = device)
    v = torch.randn(1, 4, 128, 64, device = device)

    mask = torch.randint(0, 2, (1, 128, 128), device = device, dtype = torch.bool)

    pope = PoPE(dim = 32, heads = 4).to(device)
    pos_emb = pope(128)

    out_ref = flash_attn_with_pope(
        q, k, v, pos_emb = pos_emb, mask = mask, fused = False
    )

    out_tri = flash_attn_with_pope(
        q, k, v, pos_emb = pos_emb, mask = mask, fused = True
    )

    assert allclose(out_ref, out_tri, atol = 5e-3)
