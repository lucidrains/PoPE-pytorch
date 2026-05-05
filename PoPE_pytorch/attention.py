from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor
from torch_einops_kit import and_masks, default, divisible_by, exists

from PoPE_pytorch import PolarEmbedReturn
from PoPE_pytorch.pope import apply_pope_to_qk

try:
	from .triton_pope import triton_compute_qk_similarity
	from .triton_pope_flash_attn import flash_attn

	TRITON_AVAILABLE = True
except ImportError:
	TRITON_AVAILABLE = False

def compute_attn_similarity_non_fused(q: Tensor, k: Tensor, pope: PolarEmbedReturn, *, head_dimension_at_first: bool = True) -> Tensor:

	if not head_dimension_at_first:
		q = rearrange(q, 'b n h d -> b h n d')
		k = rearrange(k, 'b n h d -> b h n d')

	q, k = apply_pope_to_qk(pope, q, k, to_magnitude=F.softplus)

	# group query attention support

	groups = q.shape[1] // k.shape[1]
	k = repeat(k, 'b h ... -> b (g h) ...', g=groups)

	return torch.einsum('b h i d, b h j d -> b h i j', q, k)

def compute_attn_similarity(
	q: Tensor, k: Tensor, pope: PolarEmbedReturn, *, allow_tf32: bool = True, head_dimension_at_first: bool = True
) -> Tensor:

	q_heads: int = q.shape[1 if head_dimension_at_first else 2]
	k_heads: int = k.shape[1 if head_dimension_at_first else 2]
	if not divisible_by(q_heads, k_heads):
		message: str = f"I received `{q_heads = }` and `{k_heads = }`, but I need `q_heads` to be divisible by `k_heads` for grouped-query attention."
		raise ValueError(message)

	freqs, bias = pope
	head_dim: int = q.shape[-1]

	common_head_dims: tuple[int, ...] = (32, 48, 64, 128, 256)
	if head_dim not in common_head_dims:
		message: str = f"I received `{head_dim = }`, but I need `head_dim` to be one of `{common_head_dims = }` for the Triton kernel family."
		raise ValueError(message)

	is_cuda: bool = q.is_cuda and k.is_cuda and freqs.is_cuda and bias.is_cuda

	if TRITON_AVAILABLE and is_cuda:
		if not head_dimension_at_first:
			q = rearrange(q, 'b n h d -> b h n d')
			k = rearrange(k, 'b n h d -> b h n d')

		rotate_dim: int = freqs.shape[-1]
		return triton_compute_qk_similarity(q, k, freqs, bias, rotate_dim, allow_tf32=allow_tf32)

	return compute_attn_similarity_non_fused(q, k, pope, head_dimension_at_first=head_dimension_at_first)

def flash_attn_with_pope(
	q: Tensor,
	k: Tensor,
	v: Tensor,
	pos_emb: PolarEmbedReturn,
	*,
	mask: Tensor | None = None,
	causal: bool = False,
	softmax_scale: float | None = None,
	fused: bool | None = None,
	head_dimension_at_first: bool = True,
	dropout: float = 0.0,
) -> Tensor:

	seq_dim: int = 2 if head_dimension_at_first else 1
	q_len, kv_len, device = q.shape[seq_dim], k.shape[seq_dim], q.device

	fused = default(fused, TRITON_AVAILABLE and q.is_cuda)

	softmax_scale = default(softmax_scale, q.shape[-1] ** -0.5)

	if fused:
		# fused kernel expects (batch, seq, heads, dim)

		if head_dimension_at_first:
			q = rearrange(q, 'b h n d -> b n h d')
			k = rearrange(k, 'b h n d -> b n h d')
			v = rearrange(v, 'b h n d -> b n h d')

		freqs, bias = pos_emb
		out: Tensor = flash_attn(
			q, k, v, freqs=freqs, pope_bias=bias, mask=mask, causal=causal, softmax_scale=softmax_scale, dropout=dropout
		)

		if head_dimension_at_first:
			out = rearrange(out, 'b n h d -> b h n d')

		return out

	# non-fused manual path
	# standardize to (batch, heads, seq, dim)

	if not head_dimension_at_first:
		q = rearrange(q, 'b n h d -> b h n d')
		k = rearrange(k, 'b n h d -> b h n d')
		v = rearrange(v, 'b n h d -> b h n d')

	q, k = apply_pope_to_qk(pos_emb, q, k, to_magnitude=F.softplus)

	# group query attention support

	groups: int = q.shape[1] // k.shape[1]
	k = repeat(k, 'b h ... -> b (g h) ...', g=groups)
	v = repeat(v, 'b h ... -> b (g h) ...', g=groups)

	# manual attention path using SDPA
	# ensure dtypes match for SDPA (apply_pope_to_qk might have upcasted to float32)

	v_dtype: torch.dtype = v.dtype
	v_dim: int = v.shape[-1]

	if q.dtype != v.dtype:
		v = v.to(q.dtype)

	attn_mask: Tensor | None = None
	if exists(mask):
		attn_mask = rearrange(mask, 'b j -> b 1 1 j')

	if causal and (q_len < kv_len or exists(attn_mask)):
		causal_mask: Tensor = torch.ones((q_len, kv_len), dtype=torch.bool, device=device).tril(diagonal=kv_len - q_len)
		attn_mask = and_masks([attn_mask, causal_mask])
		causal = False

	out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=causal, scale=softmax_scale, dropout_p=dropout)

	# mps sdpa bug (pytorch 2.9.1) - output takes q/k dim instead of v dim
	# first v_dim elements are correct, so slicing suffices
	# only triggers in no_grad (inference). todo - remove once fixed upstream

	if out.shape[-1] != v_dim:
		out = out[..., :v_dim]

	out = out.to(v_dtype)

	if not head_dimension_at_first:
		out = rearrange(out, 'b h n d -> b n h d')

	return out
