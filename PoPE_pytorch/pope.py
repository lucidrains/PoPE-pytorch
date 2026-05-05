from __future__ import annotations

from collections.abc import Callable
from math import pi
from typing import cast, NamedTuple

import torch
import torch.nn.functional as F
from einops import einsum, rearrange
from torch import Tensor, arange, cat, is_tensor
from torch._prims_common import DeviceLikeType
from torch.amp.autocast_mode import autocast
from torch.nn import Module, Parameter

from torch_einops_kit import exists, slice_right_at_dim


class PolarEmbedReturn(NamedTuple):
	freqs: Tensor
	bias: Tensor


@autocast('cuda', enabled=False)
def apply_pope_to_qk(
	pope: PolarEmbedReturn, q: Tensor, k: Tensor, to_magnitude: Callable[..., Tensor] = F.softplus, *, return_complex: bool = False
) -> tuple[Tensor, Tensor]:
	freqs, bias = pope

	q_len, k_len, qk_dim, rotate_dim = q.shape[-2], k.shape[-2], q.shape[-1], freqs.shape[-1]

	if q_len > k_len:
		message: str = f'I received `{q_len = }` and `{k_len = }`, but I need `q_len <= k_len`.'
		raise ValueError(message)

	if rotate_dim > qk_dim:
		message: str = f'I received `{rotate_dim = }` and `{qk_dim = }`, but I need `rotate_dim <= qk_dim`.'
		raise ValueError(message)

	is_partial_rotate: bool = rotate_dim < qk_dim

	if is_partial_rotate:
		q, q_rest = q[..., :rotate_dim], q[..., rotate_dim:]
		k, k_rest = k[..., :rotate_dim], k[..., rotate_dim:]

		if return_complex:
			q_rest: Tensor = torch.polar(q_rest, torch.zeros_like(q_rest))
			k_rest: Tensor = torch.polar(k_rest, torch.zeros_like(k_rest))

	if freqs.ndim == 3:
		freqs: Tensor = rearrange(freqs, 'b n d -> b 1 n d')

	freqs_with_bias: Tensor = freqs + rearrange(bias, 'h d -> h 1 d')

	# convert q and k to polar magnitudes with activation

	q, k = to_magnitude(q), to_magnitude(k)

	# apply rotations

	freqs = slice_right_at_dim(freqs, q_len, dim=-2)

	if return_complex:
		q = torch.polar(q, freqs)
	else:
		qcos, qsin = freqs.cos(), freqs.sin()
		q = rearrange([q * qcos, q * qsin], 'two ... d -> ... (d two)')

	# handle inference

	if return_complex:
		k = torch.polar(k, freqs_with_bias)
	else:
		kcos, ksin = freqs_with_bias.cos(), freqs_with_bias.sin()
		k = rearrange([k * kcos, k * ksin], 'two ... d -> ... (d two)')

	# concat

	if is_partial_rotate:
		q = cat((q, q_rest), dim=-1)
		k = cat((k, k_rest), dim=-1)

	return q, k

class PoPE(Module):

	apply_pope_to_qk = staticmethod(apply_pope_to_qk)

	def __init__(
		self, dim: int, *, heads: int, theta: float = 10000, bias_uniform_init: bool = False, inv_freqs: Tensor | None = None
	) -> None:
		super().__init__()

		if not exists(inv_freqs):
			inv_freqs = theta ** -(arange(dim).float() / dim)

		self.register_buffer('inv_freqs', inv_freqs)

		# the learned bias on the keys

		self.bias = Parameter(torch.zeros(heads, dim))

		if bias_uniform_init:
			self.bias.uniform_(-2.0 * pi, 0.0)

	@property
	def device(self) -> DeviceLikeType:
		return cast(Tensor, self.inv_freqs).device

	@autocast('cuda', enabled=False)
	def forward(self, pos_or_seq_len: Tensor | int, offset: int = 0) -> PolarEmbedReturn:
		# get positions depending on input

		if is_tensor(pos_or_seq_len):
			pos: Tensor = pos_or_seq_len
		else:
			seq_len: int = pos_or_seq_len
			pos = arange(seq_len, device=self.device, dtype=cast(Tensor, self.inv_freqs).dtype)

		pos = pos + offset

		# freqs

		freqs: Tensor = einsum(pos, cast(Tensor, self.inv_freqs), '... i, j -> ... i j')

		# the bias, with clamping

		bias: Tensor = self.bias.clamp(-2.0 * pi, 0.0)

		return PolarEmbedReturn(freqs, bias)
