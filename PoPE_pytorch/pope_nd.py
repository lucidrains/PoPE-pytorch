from __future__ import annotations

from math import pi

import torch
from einops import einsum
from hunterMakesPy import raiseIfNone
from torch import Tensor, arange, cat, meshgrid, stack
from torch._prims_common import DeviceLikeType
from torch.amp.autocast_mode import autocast
from torch.nn import Module, Parameter, ParameterList
from torch.types import Number
from torch_einops_kit import exists

from PoPE_pytorch import PolarEmbedReturn
from PoPE_pytorch.pope import apply_pope_to_qk


class AxialPoPE(Module):

	apply_pope_to_qk = staticmethod(apply_pope_to_qk)

	def __init__(
		self, dim: int, *, heads: int, axial_dims: tuple[int, ...] | None = None, theta: float = 10000, bias_uniform_init: bool = False
	) -> None:
		super().__init__()
		self.dim: int = dim
		self.heads: int = heads

		if not exists(axial_dims):
			axial_dims = (dim,)

		self.axial_dims: tuple[int, ...] = raiseIfNone(axial_dims)
		if sum(self.axial_dims) != dim:
			message: str = f'I received `{self.axial_dims = }` and `{dim = }`, but I need `sum(self.axial_dims) == dim`.'
			raise ValueError(message)

		# inv freqs for each axial dimension

		self.inv_freqs = ParameterList()

		for axial_dim in self.axial_dims:
			inv_freqs = theta ** -(arange(axial_dim).float() / axial_dim)
			self.inv_freqs.append(Parameter(inv_freqs, requires_grad=False))

		# the learned bias on the keys

		self.bias = Parameter(torch.zeros(heads, dim))

		if bias_uniform_init:
			self.bias.uniform_(-2.0 * pi, 0.0)

	@property
	def device(self) -> torch.device:
		return self.bias.device

	@staticmethod
	def get_grid_positions(*dims: Number, device: DeviceLikeType | None = None) -> Tensor:
		grid: tuple[Tensor, ...] = meshgrid(*[arange(d, device=device).float() for d in dims], indexing='ij')
		return stack([g.flatten() for g in reversed(grid)], dim=-1)

	@autocast('cuda', enabled=False)
	def forward(self, pos_or_dims: Tensor | tuple[int, ...]) -> PolarEmbedReturn:
		# handle auto grid generation if tuple is passed

		if isinstance(pos_or_dims, tuple):
			pos: Tensor = self.get_grid_positions(*pos_or_dims, device=self.device)
		else:
			pos = pos_or_dims

		# pos shape is (..., N) where N is len(axial_dims)

		if pos.shape[-1] != len(self.axial_dims):
			message: str = (
				f'I received `{pos.shape[-1] = }` and `{len(self.axial_dims) = }`, '
				'but I need `pos.shape[-1] == len(self.axial_dims)`.'
			)
			raise ValueError(message)

		all_freqs: list[Tensor] = []

		for i, inv_freqs in enumerate(self.inv_freqs):
			# pos_i shape is (...)

			pos_i: Tensor = pos[..., i]

			# freqs_i shape is (..., axial_dim)

			freqs_i: Tensor = einsum(pos_i, inv_freqs, '... , d -> ... d')
			all_freqs.append(freqs_i)

		# concat axial freqs

		freqs: Tensor = cat(all_freqs, dim=-1)

		# the bias, with clamping

		bias: Tensor = self.bias.clamp(-2.0 * pi, 0.0)

		return PolarEmbedReturn(freqs, bias)
