import torch
import torch.nn.functional as F
from torch import is_tensor, einsum
from PoPE_pytorch.pope import apply_pope_to_qk

from torch_einops_utils import and_masks

import einx
from einops import rearrange, repeat

# triton available

try:
    from .triton_pope import triton_compute_qk_similarity
    from .triton_pope_flash_attn import flash_attn
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

def pad_freqs_for_mixed_positions(freqs, indices, seq_len, device):
    if not exists(freqs):
        return freqs

    padded_freqs = freqs.new_zeros((seq_len, freqs.shape[-1]))
    padded_freqs[indices] = freqs
    return padded_freqs

# functions

def compute_attn_similarity_non_fused(
    q,
    k,
    pope,
    pope_pos_emb_indices = None,
    head_dimension_at_first = True
):
    if not head_dimension_at_first:
        q = rearrange(q, 'b n h d -> b h n d')
        k = rearrange(k, 'b n h d -> b h n d')

    q_heads, k_heads, q_len, device = q.shape[1], k.shape[1], q.shape[2], q.device

    groups = q_heads // k_heads
    k = repeat(k, 'b h ... -> b (g h) ...', g = groups)

    freqs, bias = pope

    if exists(pope_pos_emb_indices):
        freqs = pad_freqs_for_mixed_positions(freqs, pope_pos_emb_indices, q_len, device)

    pope = (freqs, bias)
    q_pope, k_pope = apply_pope_to_qk(pope, q, k, to_magnitude = F.softplus)

    sim_rot = einsum('b h i d, b h j d -> b h i j', q_pope, k_pope)

    if exists(pope_pos_emb_indices):
        # handle mixed positions
        sim_unrot = einsum('b h i d, b h j d -> b h i j', q, k)
        pos_mask = q.new_zeros(q_len, dtype = torch.bool)
        pos_mask[pope_pos_emb_indices] = True
        is_pos_2d = einx.logical_and('i, j -> i j', pos_mask, pos_mask)
        return torch.where(is_pos_2d, sim_rot, sim_unrot)

    return sim_rot

def compute_attn_similarity(
    q,
    k,
    pope,
    pope_pos_emb_indices = None,
    allow_tf32 = True,
    head_dimension_at_first = True
):
    head_idx = 1 if head_dimension_at_first else 2
    q_heads, k_heads = q.shape[head_idx], k.shape[head_idx]
    assert divisible_by(q_heads, k_heads)

    freqs, bias = pope
    head_dim = q.shape[-1]

    assert head_dim in {32, 48, 64, 128, 256}, f"head_dim {head_dim} not in common sizes"

    is_cuda = q.is_cuda and k.is_cuda and freqs.is_cuda and bias.is_cuda

    if TRITON_AVAILABLE and is_cuda:
        if not head_dimension_at_first:
            q = rearrange(q, 'b n h d -> b h n d')
            k = rearrange(k, 'b n h d -> b h n d')

        rotate_dim = freqs.shape[-1]
        
        # fallthrough to non-fused for mixed pos
        if not exists(pope_pos_emb_indices):
            return triton_compute_qk_similarity(q, k, freqs, bias, rotate_dim, allow_tf32 = allow_tf32)

    return compute_attn_similarity_non_fused(q, k, pope, pope_pos_emb_indices = pope_pos_emb_indices, head_dimension_at_first = head_dimension_at_first)

def flash_attn_with_pope(
    q,
    k,
    v,
    pos_emb = None,
    pope_pos_emb_indices = None,
    mask = None,
    causal = False,
    softmax_scale = None,
    fused = None,
    head_dimension_at_first = True,
    dropout = 0.
):
    seq_dim = 2 if head_dimension_at_first else 1
    head_idx = 1 if head_dimension_at_first else 2
    q_len, kv_len, device = q.shape[seq_dim], k.shape[seq_dim], q.device
    q_heads, k_heads = q.shape[head_idx], k.shape[head_idx]

    fused = default(fused, TRITON_AVAILABLE and q.is_cuda)

    softmax_scale = default(softmax_scale, q.shape[-1] ** -0.5)

    groups = q_heads // k_heads
    k = repeat(k, 'b h ... -> b (g h) ...', g = groups) if groups > 1 else k
    v = repeat(v, 'b h ... -> b (g h) ...', g = groups) if groups > 1 else v

    attn_mask = mask
    if exists(attn_mask):
        if attn_mask.ndim == 2 and attn_mask.shape[0] == q_len and attn_mask.shape[1] == kv_len:
            attn_mask = rearrange(attn_mask, 'i j -> 1 1 i j')
        elif attn_mask.ndim == 2:
            attn_mask = rearrange(attn_mask, 'b j -> b 1 1 j')
        elif attn_mask.ndim == 3:
            attn_mask = rearrange(attn_mask, 'b i j -> b 1 i j')

    pos_mask = None

    if exists(pos_emb):
        freqs, bias = pos_emb
        
        if exists(pope_pos_emb_indices):
            # handle mixed positions
            freqs = pad_freqs_for_mixed_positions(freqs, pope_pos_emb_indices, q_len, device)
            pos_mask = q.new_zeros(q_len, dtype = torch.bool)
            pos_mask[pope_pos_emb_indices] = True

        pos_emb = (freqs, bias)

    if fused:
        freqs, bias = pos_emb
        if head_dimension_at_first:
            q = rearrange(q, 'b h n d -> b n h d')
            k = rearrange(k, 'b h n d -> b n h d')
            v = rearrange(v, 'b h n d -> b n h d')
        out = flash_attn(q, k, v, freqs = freqs, pope_bias = bias, mask = attn_mask, causal = causal, softmax_scale = softmax_scale, dropout = dropout, pos_mask = pos_mask)
        if head_dimension_at_first:
            out = rearrange(out, 'b n h d -> b h n d')

        return out

    # non-fused manual path
    # standardize to (batch, heads, seq, dim)

    is_decode = q_len == 1

    if causal and is_decode:
        causal = False

    if not head_dimension_at_first:
        q = rearrange(q, 'b n h d -> b h n d')
        k = rearrange(k, 'b n h d -> b h n d')
        v = rearrange(v, 'b n h d -> b h n d')

    q_pope, k_pope = apply_pope_to_qk(pos_emb, q, k, to_magnitude = F.softplus)

    # manual attention path using SDPA
    # ensure dtypes match for SDPA (apply_pope_to_qk might have upcasted to float32)

    v_dtype = v.dtype
    v_dim = v.shape[-1]

    if q.dtype != v.dtype:
        v = v.to(q.dtype)

    if exists(pos_mask):
        # handle mixed positions manually

        sim_unrot = einsum('b h i d, b h j d -> b h i j', q, k) * softmax_scale
        sim_rot = einsum('b h i d, b h j d -> b h i j', q_pope, k_pope) * softmax_scale
        is_pos_2d = einx.logical_and('i, j -> i j', pos_mask, pos_mask)
        sim = torch.where(is_pos_2d, sim_rot, sim_unrot)

        if exists(attn_mask):
            sim = torch.where(attn_mask, sim, float('-inf'))

        if causal:
            causal_mask = torch.ones((q_len, kv_len), dtype = torch.bool, device = device).tril(diagonal = kv_len - q_len)
            sim = torch.where(causal_mask, sim, float('-inf'))

        attn = sim.softmax(dim = -1)
        attn = F.dropout(attn, p = dropout, training = v.requires_grad)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
    else:
        out = F.scaled_dot_product_attention(
            q_pope, k_pope, v,
            attn_mask = attn_mask,
            is_causal = causal,
            scale = softmax_scale,
            dropout_p = dropout
        )

    # mps sdpa bug (pytorch 2.9.1) - output takes q/k dim instead of v dim
    # first v_dim elements are correct, so slicing suffices
    # only triggers in no_grad (inference). todo - remove once fixed upstream

    if out.shape[-1] != v_dim:
        out = out[..., :v_dim]

    out = out.to(v_dtype)

    if not head_dimension_at_first:
        out = rearrange(out, 'b h n d -> b n h d')

    return out
