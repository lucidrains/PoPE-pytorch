import torch
import torch.nn.functional as F

from PoPE_pytorch.pope import apply_pope_to_qk

from einops import rearrange, repeat

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

# triton available

try:
    from .triton_pope import triton_compute_qk_similarity
    from .triton_pope_flash_attn import flash_attn
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

# functions

def compute_attn_similarity_non_fused(
    q,
    k,
    pope,
    head_dimension_at_first = True
):
    if not head_dimension_at_first:
        q = rearrange(q, 'b n h d -> b h n d')
        k = rearrange(k, 'b n h d -> b h n d')
    
    q, k = apply_pope_to_qk(pope, q, k, to_magnitude = F.softplus)

    # group query attention support

    groups = q.shape[1] // k.shape[1]
    k = repeat(k, 'b h ... -> b (g h) ...', g = groups)

    return torch.einsum('b h i d, b h j d -> b h i j', q, k)

def compute_attn_similarity(
    q,
    k,
    pope,
    allow_tf32 = True,
    head_dimension_at_first = True
):
    assert divisible_by(q.shape[1 if head_dimension_at_first else 2], k.shape[1 if head_dimension_at_first else 2])

    freqs, bias = pope
    head_dim = q.shape[-1]

    assert head_dim in {32, 48, 64, 128, 256}, f"head_dim {head_dim} not in common sizes"

    is_cuda = q.is_cuda and k.is_cuda and freqs.is_cuda and bias.is_cuda

    if TRITON_AVAILABLE and is_cuda:
        if not head_dimension_at_first:
            q = rearrange(q, 'b n h d -> b h n d')
            k = rearrange(k, 'b n h d -> b h n d')

        rotate_dim = freqs.shape[-1]
        return triton_compute_qk_similarity(q, k, freqs, bias, rotate_dim, allow_tf32 = allow_tf32)

    return compute_attn_similarity_non_fused(q, k, pope, head_dimension_at_first = head_dimension_at_first)

def flash_attn_with_pope(
    q,
    k,
    v,
    pope = None,
    causal = False,
    softmax_scale = None,
    fused = None,
    head_dimension_at_first = True
):
    fused = default(fused, TRITON_AVAILABLE and q.is_cuda)

    softmax_scale = default(softmax_scale, q.shape[-1] ** -0.5)

    if fused:
        # fused kernel expects (batch, seq, heads, dim)

        if head_dimension_at_first:
            q = rearrange(q, 'b h n d -> b n h d')
            k = rearrange(k, 'b h n d -> b n h d')
            v = rearrange(v, 'b h n d -> b n h d')

        freqs, bias = pope
        out = flash_attn(q, k, v, freqs = freqs, pope_bias = bias, causal = causal, softmax_scale = softmax_scale)

        if head_dimension_at_first:
            out = rearrange(out, 'b n h d -> b h n d')

        return out

    # non-fused manual path
    # standardize to (batch, heads, seq, dim)

    if not head_dimension_at_first:
        q = rearrange(q, 'b n h d -> b h n d')
        k = rearrange(k, 'b n h d -> b h n d')
        v = rearrange(v, 'b n h d -> b h n d')

    q, k = apply_pope_to_qk(pope, q, k, to_magnitude = F.softplus)
    
    # group query attention support

    groups = q.shape[1] // k.shape[1]
    k = repeat(k, 'b h ... -> b (g h) ...', g = groups)
    v = repeat(v, 'b h ... -> b (g h) ...', g = groups)

    # manual attention path using SDPA
    # ensure dtypes match for SDPA (apply_pope_to_qk might have upcasted to float32)

    v_dtype = v.dtype
    if q.dtype != v.dtype:
        v = v.to(q.dtype)

    out = F.scaled_dot_product_attention(
        q, k, v,
        is_causal = causal,
        scale = softmax_scale
    )

    out = out.to(v_dtype)

    if not head_dimension_at_first:
        out = rearrange(out, 'b h n d -> b n h d')

    return out
