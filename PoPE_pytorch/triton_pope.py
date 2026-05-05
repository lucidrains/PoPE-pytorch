from __future__ import annotations

import os
from collections.abc import Callable, Mapping

import torch
import triton
import triton.language as tl
from einops import repeat
from torch_einops_kit import divisible_by, exists
from triton.runtime.autotuner import Autotuner
from triton.runtime.jit import JITFunction


@triton.jit
def softplus(x: tl.tensor) -> tl.tensor:
	return tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))

@triton.jit
def softplus_grad(x: tl.tensor) -> tl.tensor:
	return tl.sigmoid(x)

# autotuning

_NO_AUTOTUNE: bool = os.environ.get('POPE_NO_AUTOTUNE', '0') == '1'

def _max_shared_mem(device_idx: int = 0) -> int:
	return triton.runtime.driver.active.utils.get_device_properties(device_idx)['max_shared_mem']

def _estimate_shared_bytes(bm: int, bn: int, blk_d: int, stages: int, elem_bytes: int) -> int:
	return (bm + bn) * blk_d * 4 * stages * elem_bytes

def _filter_configs(configs: list[triton.Config], blk_d: int, elem_bytes: int, device_idx: int = 0) -> list[triton.Config]:
	limit: int = _max_shared_mem(device_idx)
	valid: list[triton.Config] = [c for c in configs if _estimate_shared_bytes(c.kwargs['BM'], c.kwargs['BN'], blk_d, c.num_stages, elem_bytes) <= limit]
	return valid or configs[-1:]

def cache_by_id(fn: Callable[..., Autotuner]) -> Callable[..., Autotuner]:
	cache: dict[tuple[int | Callable[[], list[triton.Config]] | list[str], ...], Autotuner] = {}

	def inner(kernel_fn: JITFunction[torch.Tensor], *args: Callable[[], list[triton.Config]] | list[str] | int) -> Autotuner:
		key: tuple[int | Callable[[], list[triton.Config]] | list[str], ...] = (id(kernel_fn), *args)
		if key not in cache:
			cache[key] = fn(kernel_fn, *args)
		return cache[key]

	return inner

@cache_by_id
def get_autotuned_kernel(kernel_fn: JITFunction[torch.Tensor], configs_fn: Callable[[], list[triton.Config]], keys: list[str], blk_d: int, elem_bytes: int, device_idx: int = 0) -> Autotuner:
	configs: list[triton.Config] = configs_fn()
	configs = _filter_configs(configs, blk_d, elem_bytes, device_idx)
	return triton.autotune(configs, key=keys)(kernel_fn)

def _bwd_dqk_pre_hook(nargs: Mapping[str, torch.Tensor]) -> None:
	df: torch.Tensor | None = nargs.get('dFreqs')
	if df is not None:
		df.zero_()

def _bwd_dbias_pre_hook(nargs: Mapping[str, torch.Tensor]) -> None:
	bias_gradient: torch.Tensor = nargs['dBias']
	bias_gradient.zero_()

def _fwd_configs() -> list[triton.Config]:
	return [
		triton.Config({'BM': 64, 'BN': 32}, num_stages=1, num_warps=4),
		triton.Config({'BM': 32, 'BN': 64}, num_stages=1, num_warps=4),
		triton.Config({'BM': 32, 'BN': 32}, num_stages=2, num_warps=4),
		triton.Config({'BM': 16, 'BN': 16}, num_stages=1, num_warps=4),
	]

def _bwd_dqk_configs() -> list[triton.Config]:
	return [
		triton.Config({'BM': 64, 'BN': 32}, num_stages=1, num_warps=4, pre_hook=_bwd_dqk_pre_hook),
		triton.Config({'BM': 32, 'BN': 64}, num_stages=1, num_warps=4, pre_hook=_bwd_dqk_pre_hook),
		triton.Config({'BM': 32, 'BN': 32}, num_stages=2, num_warps=4, pre_hook=_bwd_dqk_pre_hook),
		triton.Config({'BM': 16, 'BN': 16}, num_stages=1, num_warps=4, pre_hook=_bwd_dqk_pre_hook),
	]

def _bwd_dbias_configs() -> list[triton.Config]:
	return [
		triton.Config({'BM': 64, 'BN': 32}, num_stages=1, num_warps=4, pre_hook=_bwd_dbias_pre_hook),
		triton.Config({'BM': 32, 'BN': 32}, num_stages=2, num_warps=4, pre_hook=_bwd_dbias_pre_hook),
		triton.Config({'BM': 16, 'BN': 16}, num_stages=1, num_warps=4, pre_hook=_bwd_dbias_pre_hook),
	]

# forward kernel

@triton.jit
def _fwd_kernel(
	Q: torch.Tensor, K: torch.Tensor, Freqs: torch.Tensor, Bias: torch.Tensor, Out: torch.Tensor,
	stride_qb: int, stride_qh: int, stride_qi: int, stride_qd: int,
	stride_kb: int, stride_kh: int, stride_kj: int, stride_kd: int,
	stride_fb: int, stride_fh: int, stride_fi: int, stride_fd: int,
	stride_bh: int, stride_bd: int,
	stride_ob: int, stride_oh: int, stride_oi: int, stride_oj: int,
	n_heads: int, seq_q: int, seq_k: int, head_dim: int, rotate_dim: int,
	BM: tl.constexpr, BN: tl.constexpr, BLOCK_D: tl.constexpr,
	ALLOW_TF32: tl.constexpr,
) -> None:
	bhid: tl.tensor = tl.program_id(0)
	blk_m: tl.tensor = tl.program_id(1)
	blk_n: tl.tensor = tl.program_id(2)

	b: tl.tensor = bhid // n_heads
	h: tl.tensor = bhid % n_heads

	off_i: tl.tensor = blk_m * BM + tl.arange(0, BM)
	off_j: tl.tensor = blk_n * BN + tl.arange(0, BN)
	off_d: tl.tensor = tl.arange(0, BLOCK_D)

	mask_i: tl.tensor = off_i < seq_q
	mask_j: tl.tensor = off_j < seq_k

	acc: tl.tensor = tl.zeros((BM, BN), dtype=tl.float32)
	q_off: int = seq_k - seq_q

	for d_start in range(0, head_dim, BLOCK_D):
		mask_d: tl.tensor = (d_start + off_d) < head_dim
		mask_r: tl.tensor = (d_start + off_d) < rotate_dim

		# load q, k

		q: tl.tensor = tl.load(Q + b * stride_qb + h * stride_qh + off_i[:, None] * stride_qi + (d_start + off_d[None, :]) * stride_qd, mask = mask_i[:, None] & mask_d[None, :], other = 0.0)
		k: tl.tensor = tl.load(K + b * stride_kb + h * stride_kh + off_j[:, None] * stride_kj + (d_start + off_d[None, :]) * stride_kd, mask = mask_j[:, None] & mask_d[None, :], other = 0.0)

		# load freqs, bias

		fq: tl.tensor = tl.load(Freqs + b * stride_fb + h * stride_fh + (q_off + off_i[:, None]) * stride_fi + (d_start + off_d[None, :]) * stride_fd, mask = mask_i[:, None] & mask_r[None, :], other = 0.0)
		fk: tl.tensor = tl.load(Freqs + b * stride_fb + h * stride_fh + off_j[:, None] * stride_fi + (d_start + off_d[None, :]) * stride_fd, mask = mask_j[:, None] & mask_r[None, :], other = 0.0)
		bias: tl.tensor = tl.load(Bias + h * stride_bh + (d_start + off_d), mask = mask_r, other = 0.0)

		# softplus activation

		act_q: tl.tensor = tl.where(mask_r[None, :], softplus(q), q)
		act_k: tl.tensor = tl.where(mask_r[None, :], softplus(k), k)

		# rotary embedding

		q_cos: tl.tensor = tl.where(mask_r[None, :], act_q * tl.cos(fq), act_q)
		q_sin: tl.tensor = tl.where(mask_r[None, :], act_q * tl.sin(fq), tl.zeros((1,)))

		th_k: tl.tensor = fk + bias[None, :]
		k_cos: tl.tensor = tl.where(mask_r[None, :], act_k * tl.cos(th_k), act_k)
		k_sin: tl.tensor = tl.where(mask_r[None, :], act_k * tl.sin(th_k), tl.zeros((1,)))

		# accumulate similarity

		acc: tl.tensor = tl.dot(q_cos, tl.trans(k_cos), acc, allow_tf32=ALLOW_TF32)
		acc = tl.dot(q_sin, tl.trans(k_sin), acc, allow_tf32=ALLOW_TF32)

	tl.store(Out + b * stride_ob + h * stride_oh + off_i[:, None] * stride_oi + off_j[None, :] * stride_oj, acc, mask = mask_i[:, None] & mask_j[None, :])

# backward kernel - computes dQ (MODE=0) or dK (MODE=1) + dFreqs
# each MODE gets its own dFreqs buffer so pre_hook can safely zero it

@triton.jit
def _bwd_kernel_dqk_df(
	dQ: torch.Tensor, dK: torch.Tensor, dFreqs: torch.Tensor, dS: torch.Tensor, Q: torch.Tensor, K: torch.Tensor, Freqs: torch.Tensor, Bias: torch.Tensor,
	stride_dqb: int, stride_dqh: int, stride_dqi: int, stride_dqd: int,
	stride_dkb: int, stride_dkh: int, stride_dkj: int, stride_dkd: int,
	stride_dfb: int, stride_dfh: int, stride_dfi: int, stride_dfd: int,
	stride_sb: int, stride_sh: int, stride_si: int, stride_sj: int,
	stride_qb: int, stride_qh: int, stride_qi: int, stride_qd: int,
	stride_kb: int, stride_kh: int, stride_kj: int, stride_kd: int,
	stride_fb: int, stride_fh: int, stride_fi: int, stride_fd: int,
	stride_bh: int, stride_bd: int,
	n_heads: int, seq_q: int, seq_k: int, head_dim: int, rotate_dim: int,
	BM: tl.constexpr, BN: tl.constexpr, BLOCK_D: tl.constexpr,
	MODE: tl.constexpr,
	ALLOW_TF32: tl.constexpr,
	HAS_DF: tl.constexpr,
) -> None:
	bhid: tl.tensor = tl.program_id(0)
	grid_idx: tl.tensor = tl.program_id(1)

	b: tl.tensor = bhid // n_heads
	h: tl.tensor = bhid % n_heads
	off_d: tl.tensor = tl.arange(0, BLOCK_D)
	q_off: int = seq_k - seq_q

	if MODE == 0:
		# compute dQ by iterating over k-blocks

		off_i: tl.tensor = grid_idx * BM + tl.arange(0, BM)
		mask_i: tl.tensor = off_i < seq_q

		for d_start in range(0, head_dim, BLOCK_D):
			mask_d: tl.tensor = (d_start + off_d) < head_dim
			mask_r: tl.tensor = (d_start + off_d) < rotate_dim

			d_q: tl.tensor = tl.zeros((BM, BLOCK_D), dtype=tl.float32)
			d_fq: tl.tensor = tl.zeros((BM, BLOCK_D), dtype=tl.float32)

			# load and precompute q-side quantities

			q: tl.tensor = tl.load(Q + b * stride_qb + h * stride_qh + off_i[:, None] * stride_qi + (d_start + off_d[None, :]) * stride_qd, mask = mask_i[:, None] & mask_d[None, :], other = 0.0)
			fq: tl.tensor = tl.load(Freqs + b * stride_fb + h * stride_fh + (q_off + off_i[:, None]) * stride_fi + (d_start + off_d[None, :]) * stride_fd, mask = mask_i[:, None] & mask_r[None, :], other = 0.0)

			sp_dq: tl.tensor = tl.where(mask_r[None, :], softplus_grad(q), tl.sqrt(1.0))
			cos_fq: tl.tensor = tl.where(mask_r[None, :], tl.cos(fq), tl.sqrt(1.0))
			sin_fq: tl.tensor = tl.where(mask_r[None, :], tl.sin(fq), tl.zeros((1,)))
			act_q: tl.tensor = tl.where(mask_r[None, :], softplus(q), q)

			for j_start in range(0, seq_k, BN):
				off_j: tl.tensor = j_start + tl.arange(0, BN)
				mask_j: tl.tensor = off_j < seq_k

				# load grad and k-side quantities

				ds: tl.tensor = tl.load(dS + b * stride_sb + h * stride_sh + off_i[:, None] * stride_si + off_j[None, :] * stride_sj, mask = mask_i[:, None] & mask_j[None, :], other = 0.0)

				k: tl.tensor = tl.load(K + b * stride_kb + h * stride_kh + off_j[:, None] * stride_kj + (d_start + off_d[None, :]) * stride_kd, mask = mask_j[:, None] & mask_d[None, :], other = 0.0)
				fk: tl.tensor = tl.load(Freqs + b * stride_fb + h * stride_fh + off_j[:, None] * stride_fi + (d_start + off_d[None, :]) * stride_fd, mask = mask_j[:, None] & mask_r[None, :], other = 0.0)
				bias: tl.tensor = tl.load(Bias + h * stride_bh + (d_start + off_d), mask = mask_r, other = 0.0)

				act_k: tl.tensor = tl.where(mask_r[None, :], softplus(k), k)
				th_k: tl.tensor = fk + bias[None, :]
				cos_tk: tl.tensor = tl.where(mask_r[None, :], tl.cos(th_k), tl.sqrt(1.0))
				sin_tk: tl.tensor = tl.where(mask_r[None, :], tl.sin(th_k), tl.zeros((1,)))

				dot_cos: tl.tensor = tl.dot(ds, (act_k * cos_tk).to(tl.float32), allow_tf32=ALLOW_TF32)
				dot_sin: tl.tensor = tl.dot(ds, (act_k * sin_tk).to(tl.float32), allow_tf32=ALLOW_TF32)

				d_q += sp_dq * (cos_fq * dot_cos + sin_fq * dot_sin)
				if HAS_DF:
					d_fq += act_q * (cos_fq * dot_sin - sin_fq * dot_cos)

			tl.store(dQ + b * stride_dqb + h * stride_dqh + off_i[:, None] * stride_dqi + (d_start + off_d[None, :]) * stride_dqd, d_q, mask = mask_i[:, None] & mask_d[None, :])
			if HAS_DF:
				tl.atomic_add(dFreqs + b * stride_dfb + h * stride_dfh + (q_off + off_i[:, None]) * stride_dfi + (d_start + off_d[None, :]) * stride_dfd, d_fq, mask = mask_i[:, None] & mask_r[None, :])

	else:
		# MODE == 1: compute dK by iterating over q-blocks

		off_j = grid_idx * BN + tl.arange(0, BN)
		mask_j = off_j < seq_k

		for d_start in range(0, head_dim, BLOCK_D):
			mask_d = (d_start + off_d) < head_dim
			mask_r = (d_start + off_d) < rotate_dim

			d_k: tl.tensor = tl.zeros((BN, BLOCK_D), dtype=tl.float32)
			d_fk: tl.tensor = tl.zeros((BN, BLOCK_D), dtype=tl.float32)

			# load and precompute k-side quantities

			k = tl.load(K + b * stride_kb + h * stride_kh + off_j[:, None] * stride_kj + (d_start + off_d[None, :]) * stride_kd, mask = mask_j[:, None] & mask_d[None, :], other = 0.0)
			fk = tl.load(Freqs + b * stride_fb + h * stride_fh + off_j[:, None] * stride_fi + (d_start + off_d[None, :]) * stride_fd, mask = mask_j[:, None] & mask_r[None, :], other = 0.0)
			bias = tl.load(Bias + h * stride_bh + (d_start + off_d), mask = mask_r, other = 0.0)

			sp_dk: tl.tensor = tl.where(mask_r[None, :], softplus_grad(k), tl.sqrt(1.0))
			th_k = fk + bias[None, :]
			cos_tk = tl.where(mask_r[None, :], tl.cos(th_k), tl.sqrt(1.0))
			sin_tk = tl.where(mask_r[None, :], tl.sin(th_k), tl.zeros((1,)))
			act_k = tl.where(mask_r[None, :], softplus(k), k)

			for i_start in range(0, seq_q, BM):
				off_i = i_start + tl.arange(0, BM)
				mask_i = off_i < seq_q

				ds = tl.load(dS + b * stride_sb + h * stride_sh + off_i[:, None] * stride_si + off_j[None, :] * stride_sj, mask = mask_i[:, None] & mask_j[None, :], other = 0.0)

				q = tl.load(Q + b * stride_qb + h * stride_qh + off_i[:, None] * stride_qi + (d_start + off_d[None, :]) * stride_qd, mask = mask_i[:, None] & mask_d[None, :], other = 0.0)
				fq = tl.load(Freqs + b * stride_fb + h * stride_fh + (q_off + off_i[:, None]) * stride_fi + (d_start + off_d[None, :]) * stride_fd, mask = mask_i[:, None] & mask_r[None, :], other = 0.0)

				act_q = tl.where(mask_r[None, :], softplus(q), q)
				cos_fq = tl.where(mask_r[None, :], tl.cos(fq), tl.sqrt(1.0))
				sin_fq = tl.where(mask_r[None, :], tl.sin(fq), tl.zeros((1,)))

				dot_cos = tl.dot(tl.trans(ds), (act_q * cos_fq).to(tl.float32), allow_tf32=ALLOW_TF32)
				dot_sin = tl.dot(tl.trans(ds), (act_q * sin_fq).to(tl.float32), allow_tf32=ALLOW_TF32)

				d_k += sp_dk * (cos_tk * dot_cos + sin_tk * dot_sin)
				if HAS_DF:
					d_fk += act_k * (cos_tk * dot_sin - sin_tk * dot_cos)

			tl.store(dK + b * stride_dkb + h * stride_dkh + off_j[:, None] * stride_dkj + (d_start + off_d[None, :]) * stride_dkd, d_k, mask = mask_j[:, None] & mask_d[None, :])
			if HAS_DF:
				tl.atomic_add(dFreqs + b * stride_dfb + h * stride_dfh + off_j[:, None] * stride_dfi + (d_start + off_d[None, :]) * stride_dfd, d_fk, mask = mask_j[:, None] & mask_r[None, :])

# backward kernel for bias gradient

@triton.jit
def _bwd_kernel_dbias(
	dBias: torch.Tensor, dS: torch.Tensor, Q: torch.Tensor, K: torch.Tensor, Freqs: torch.Tensor, Bias: torch.Tensor,
	stride_sb: int, stride_sh: int, stride_si: int, stride_sj: int,
	stride_qb: int, stride_qh: int, stride_qi: int, stride_qd: int,
	stride_kb: int, stride_kh: int, stride_kj: int, stride_kd: int,
	stride_fb: int, stride_fh: int, stride_fi: int, stride_fd: int,
	stride_bh: int, stride_bd: int,
	batch: int, n_heads: int, seq_q: int, seq_k: int, head_dim: int, rotate_dim: int,
	BM: tl.constexpr, BN: tl.constexpr, BLOCK_D: tl.constexpr,
	ALLOW_TF32: tl.constexpr,
) -> None:
	bhid: tl.tensor = tl.program_id(0)
	blk_m: tl.tensor = tl.program_id(1)

	b: tl.tensor = bhid // n_heads
	h: tl.tensor = bhid % n_heads

	off_i: tl.tensor = blk_m * BM + tl.arange(0, BM)
	mask_i: tl.tensor = off_i < seq_q
	off_d: tl.tensor = tl.arange(0, BLOCK_D)
	q_off: int = seq_k - seq_q

	for d_start in range(0, head_dim, BLOCK_D):
		mask_d: tl.tensor = (d_start + off_d) < head_dim
		mask_r: tl.tensor = (d_start + off_d) < rotate_dim

		# load q-side

		q: tl.tensor = tl.load(Q + b * stride_qb + h * stride_qh + off_i[:, None] * stride_qi + (d_start + off_d[None, :]) * stride_qd, mask = mask_i[:, None] & mask_d[None, :], other = 0.0)
		fq: tl.tensor = tl.load(Freqs + b * stride_fb + h * stride_fh + (q_off + off_i[:, None]) * stride_fi + (d_start + off_d[None, :]) * stride_fd, mask = mask_i[:, None] & mask_r[None, :], other = 0.0)

		act_q: tl.tensor = tl.where(mask_r[None, :], softplus(q), q)
		cos_fq: tl.tensor = tl.where(mask_r[None, :], tl.cos(fq), tl.sqrt(1.0))
		sin_fq: tl.tensor = tl.where(mask_r[None, :], tl.sin(fq), tl.zeros((1,)))

		d_bias: tl.tensor = tl.zeros((BLOCK_D,), dtype=tl.float32)
		bias: tl.tensor = tl.load(Bias + h * stride_bh + (d_start + off_d), mask=mask_r, other=0.0)

		for j_start in range(0, seq_k, BN):
			off_j: tl.tensor = j_start + tl.arange(0, BN)
			mask_j: tl.tensor = off_j < seq_k

			ds: tl.tensor = tl.load(dS + b * stride_sb + h * stride_sh + off_i[:, None] * stride_si + off_j[None, :] * stride_sj, mask = mask_i[:, None] & mask_j[None, :], other = 0.0)

			k: tl.tensor = tl.load(K + b * stride_kb + h * stride_kh + off_j[:, None] * stride_kj + (d_start + off_d[None, :]) * stride_kd, mask = mask_j[:, None] & mask_d[None, :], other = 0.0)
			fk: tl.tensor = tl.load(Freqs + b * stride_fb + h * stride_fh + off_j[:, None] * stride_fi + (d_start + off_d[None, :]) * stride_fd, mask = mask_j[:, None] & mask_r[None, :], other = 0.0)

			act_k: tl.tensor = tl.where(mask_r[None, :], softplus(k), k)
			th_k: tl.tensor = fk + bias[None, :]
			cos_tk: tl.tensor = tl.where(mask_r[None, :], tl.cos(th_k), tl.sqrt(1.0))
			sin_tk: tl.tensor = tl.where(mask_r[None, :], tl.sin(th_k), tl.zeros((1,)))

			# dbias = sum_ij ds_ij * d/dbias (q_cos_i . k_cos_j + q_sin_i . k_sin_j)
			#       = sum_ij ds_ij * sum_d (act_q * sin_fq)(act_k * cos_tk) - (act_q * cos_fq)(act_k * sin_tk)

			q_sin_d: tl.tensor = (act_q * sin_fq).to(tl.float32)
			q_cos_d: tl.tensor = (act_q * cos_fq).to(tl.float32)
			k_cos_d: tl.tensor = (act_k * cos_tk).to(tl.float32)
			k_sin_d: tl.tensor = (act_k * sin_tk).to(tl.float32)

			dot_sc: tl.tensor = tl.dot(tl.trans(ds), q_sin_d, allow_tf32=ALLOW_TF32)
			dot_cc: tl.tensor = tl.dot(tl.trans(ds), q_cos_d, allow_tf32=ALLOW_TF32)
			d_bias += tl.sum(k_cos_d * dot_sc - k_sin_d * dot_cc, axis=0)

		if d_start < rotate_dim:
			tl.atomic_add(dBias + h * stride_bh + (d_start + off_d), d_bias, mask=mask_d & mask_r)

# autograd wrapper

class _PoPESimilarityContext(torch.autograd.function.FunctionCtx):
	orig_freqs_shape: torch.Size
	freqs_requires_grad: bool
	bias_requires_grad: bool
	rotate_dim: int
	allow_tf32: bool
	saved_tensors: tuple[torch.Tensor, ...]

class PoPESimilarityFunction(torch.autograd.Function):
	@staticmethod
	def forward(ctx: _PoPESimilarityContext, q: torch.Tensor, k: torch.Tensor, freqs: torch.Tensor, bias: torch.Tensor, rotate_dim: int, *, allow_tf32: bool) -> torch.Tensor:
		b, h, seq_q, d = q.shape
		seq_k: int = k.shape[2]

		ctx.orig_freqs_shape = freqs.shape
		ctx.freqs_requires_grad = freqs.requires_grad
		ctx.bias_requires_grad = bias.requires_grad

		# expand freqs to (b, h, seq, rotate_dim) if needed

		if freqs.ndim == 2:
			freqs = freqs.view(1, 1, freqs.shape[0], rotate_dim).expand(b, h, freqs.shape[0], rotate_dim)
		elif freqs.ndim == 3:
			freqs = freqs.view(freqs.shape[0], 1, freqs.shape[1], rotate_dim).expand(b, h, freqs.shape[1], rotate_dim)

		freqs = freqs.contiguous()
		sim: torch.Tensor = torch.empty((b, h, seq_q, seq_k), device=q.device, dtype=q.dtype)
		blk_d: int = max(triton.next_power_of_2(d), 16)

		configs: list[triton.Config] = _filter_configs(_fwd_configs(), blk_d, q.element_size(), q.device.index)

		if _NO_AUTOTUNE:
			bm: int = configs[0].kwargs['BM']
			bn: int = configs[0].kwargs['BN']
			launch_grid_fixed: tuple[int, int, int] = (b * h, triton.cdiv(seq_q, bm), triton.cdiv(seq_k, bn))
			_fwd_kernel[launch_grid_fixed](
				q, k, freqs, bias, sim,
				q.stride(0), q.stride(1), q.stride(2), q.stride(3),
				k.stride(0), k.stride(1), k.stride(2), k.stride(3),
				freqs.stride(0), freqs.stride(1), freqs.stride(2), freqs.stride(3),
				bias.stride(0), bias.stride(1),
				sim.stride(0), sim.stride(1), sim.stride(2), sim.stride(3),
				h, seq_q, seq_k, d, rotate_dim,
				BM = bm, BN = bn, BLOCK_D = blk_d, ALLOW_TF32 = allow_tf32,
			)
		else:
			kernel: Autotuner = get_autotuned_kernel(
				_fwd_kernel, _fwd_configs, ('seq_q', 'seq_k', 'head_dim'), blk_d, q.element_size(), q.device.index
			)
			launch_grid_autotuned: Callable[[dict[str, int]], tuple[int, int, int]] = lambda META: (b * h, triton.cdiv(seq_q, META['BM']), triton.cdiv(seq_k, META['BN']))
			kernel[launch_grid_autotuned](
				q, k, freqs, bias, sim,
				q.stride(0), q.stride(1), q.stride(2), q.stride(3),
				k.stride(0), k.stride(1), k.stride(2), k.stride(3),
				freqs.stride(0), freqs.stride(1), freqs.stride(2), freqs.stride(3),
				bias.stride(0), bias.stride(1),
				sim.stride(0), sim.stride(1), sim.stride(2), sim.stride(3),
				h, seq_q, seq_k, d, rotate_dim,
				BLOCK_D = blk_d, ALLOW_TF32 = allow_tf32,
			)

		ctx.save_for_backward(q, k, freqs, bias)
		ctx.rotate_dim = rotate_dim
		ctx.allow_tf32 = allow_tf32
		return sim

	@staticmethod
	def backward(ctx: _PoPESimilarityContext, grad_sim: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, None, None]:
		q, k, freqs, bias = ctx.saved_tensors
		rotate_dim: int = ctx.rotate_dim
		allow_tf32: bool = ctx.allow_tf32
		b, h, seq_q, d = q.shape
		seq_k: int = k.shape[2]

		dq: torch.Tensor = torch.zeros_like(q, dtype=torch.float32)
		dk: torch.Tensor = torch.empty_like(k)
		has_df: bool = ctx.freqs_requires_grad
		has_db: bool = ctx.bias_requires_grad
		dfreqs: torch.Tensor | None = torch.zeros_like(freqs, dtype=torch.float32) if has_df else None
		dbias: torch.Tensor | None = torch.zeros_like(bias, dtype=torch.float32) if has_db else None

		grad_sim = grad_sim.contiguous()
		blk_d: int = max(triton.next_power_of_2(d), 16)
		elem_bytes: int = q.element_size()
		dev: int | None = q.device.index

		dqk_configs: list[triton.Config] = _filter_configs(_bwd_dqk_configs(), blk_d, elem_bytes, dev)
		dbias_configs: list[triton.Config] = _filter_configs(_bwd_dbias_configs(), blk_d, elem_bytes, dev)

		if _NO_AUTOTUNE:
			bm: int = dqk_configs[0].kwargs['BM']
			bn: int = dqk_configs[0].kwargs['BN']
		else:
			kernel_dqk: Autotuner = get_autotuned_kernel(_bwd_kernel_dqk_df, _bwd_dqk_configs, ('seq_q', 'seq_k', 'head_dim'), blk_d, elem_bytes, dev)
			kernel_dbias: Autotuner = get_autotuned_kernel(_bwd_kernel_dbias, _bwd_dbias_configs, ('seq_q', 'seq_k', 'head_dim'), blk_d, elem_bytes, dev)

		# shared stride args for both backward calls

		dfreqs_strides: Callable[[torch.Tensor | None], tuple[int, int, int, int]] = lambda df: (
			df.stride(0) if exists(df) else 0,
			df.stride(1) if exists(df) else 0,
			df.stride(2) if exists(df) else 0,
			df.stride(3) if exists(df) else 0,
		)

		# separate dFreqs buffers for MODE=0 and MODE=1
		# so autotuner pre_hook can safely zero each independently

		dfreqs_q: torch.Tensor | None = torch.zeros_like(freqs, dtype=torch.float32) if has_df else None
		dfreqs_k: torch.Tensor | None = torch.zeros_like(freqs, dtype=torch.float32) if has_df else None

		# MODE=0: compute dQ

		if _NO_AUTOTUNE:
			_bwd_kernel_dqk_df[(b * h, triton.cdiv(seq_q, bm))](
				dq, dk, dfreqs_q, grad_sim, q, k, freqs, bias,
				dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
				dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
				*dfreqs_strides(dfreqs_q),
				grad_sim.stride(0), grad_sim.stride(1), grad_sim.stride(2), grad_sim.stride(3),
				q.stride(0), q.stride(1), q.stride(2), q.stride(3),
				k.stride(0), k.stride(1), k.stride(2), k.stride(3),
				freqs.stride(0), freqs.stride(1), freqs.stride(2), freqs.stride(3),
				bias.stride(0), bias.stride(1),
				h, seq_q, seq_k, d, rotate_dim,
				BM = bm, BN = bn, BLOCK_D = blk_d,
				MODE = 0, ALLOW_TF32 = allow_tf32, HAS_DF = has_df,
			)
		else:
			grid_q: Callable[[dict[str, int]], tuple[int, int]] = lambda META: (b * h, triton.cdiv(seq_q, META['BM']))
			kernel_dqk[grid_q](
				dq, dk, dfreqs_q, grad_sim, q, k, freqs, bias,
				dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
				dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
				*dfreqs_strides(dfreqs_q),
				grad_sim.stride(0), grad_sim.stride(1), grad_sim.stride(2), grad_sim.stride(3),
				q.stride(0), q.stride(1), q.stride(2), q.stride(3),
				k.stride(0), k.stride(1), k.stride(2), k.stride(3),
				freqs.stride(0), freqs.stride(1), freqs.stride(2), freqs.stride(3),
				bias.stride(0), bias.stride(1),
				h, seq_q, seq_k, d, rotate_dim,
				BLOCK_D = blk_d,
				MODE = 0, ALLOW_TF32 = allow_tf32, HAS_DF = has_df,
			)

		# MODE=1: compute dK

		if _NO_AUTOTUNE:
			_bwd_kernel_dqk_df[(b * h, triton.cdiv(seq_k, bn))](
				dq, dk, dfreqs_k, grad_sim, q, k, freqs, bias,
				dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
				dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
				*dfreqs_strides(dfreqs_k),
				grad_sim.stride(0), grad_sim.stride(1), grad_sim.stride(2), grad_sim.stride(3),
				q.stride(0), q.stride(1), q.stride(2), q.stride(3),
				k.stride(0), k.stride(1), k.stride(2), k.stride(3),
				freqs.stride(0), freqs.stride(1), freqs.stride(2), freqs.stride(3),
				bias.stride(0), bias.stride(1),
				h, seq_q, seq_k, d, rotate_dim,
				BM = bm, BN = bn, BLOCK_D = blk_d,
				MODE = 1, ALLOW_TF32 = allow_tf32, HAS_DF = has_df,
			)
		else:
			grid_k: Callable[[dict[str, int]], tuple[int, int]] = lambda META: (b * h, triton.cdiv(seq_k, META['BN']))
			kernel_dqk[grid_k](
				dq, dk, dfreqs_k, grad_sim, q, k, freqs, bias,
				dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
				dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
				*dfreqs_strides(dfreqs_k),
				grad_sim.stride(0), grad_sim.stride(1), grad_sim.stride(2), grad_sim.stride(3),
				q.stride(0), q.stride(1), q.stride(2), q.stride(3),
				k.stride(0), k.stride(1), k.stride(2), k.stride(3),
				freqs.stride(0), freqs.stride(1), freqs.stride(2), freqs.stride(3),
				bias.stride(0), bias.stride(1),
				h, seq_q, seq_k, d, rotate_dim,
				BLOCK_D = blk_d,
				MODE = 1, ALLOW_TF32 = allow_tf32, HAS_DF = has_df,
			)

		# compute dBias

		if exists(dbias):
			if _NO_AUTOTUNE:
				_bwd_kernel_dbias[(b * h, triton.cdiv(seq_q, bm))](
					dbias, grad_sim, q, k, freqs, bias,
					grad_sim.stride(0), grad_sim.stride(1), grad_sim.stride(2), grad_sim.stride(3),
					q.stride(0), q.stride(1), q.stride(2), q.stride(3),
					k.stride(0), k.stride(1), k.stride(2), k.stride(3),
					freqs.stride(0), freqs.stride(1), freqs.stride(2), freqs.stride(3),
					bias.stride(0), bias.stride(1),
					b, h, seq_q, seq_k, d, rotate_dim,
					BM = bm, BN = bn, BLOCK_D = blk_d, ALLOW_TF32 = allow_tf32,
				)
			else:
				grid_b: Callable[[dict[str, int]], tuple[int, int]] = lambda META: (b * h, triton.cdiv(seq_q, META['BM']))
				kernel_dbias[grid_b](
					dbias, grad_sim, q, k, freqs, bias,
					grad_sim.stride(0), grad_sim.stride(1), grad_sim.stride(2), grad_sim.stride(3),
					q.stride(0), q.stride(1), q.stride(2), q.stride(3),
					k.stride(0), k.stride(1), k.stride(2), k.stride(3),
					freqs.stride(0), freqs.stride(1), freqs.stride(2), freqs.stride(3),
					bias.stride(0), bias.stride(1),
					b, h, seq_q, seq_k, d, rotate_dim,
					BLOCK_D = blk_d, ALLOW_TF32 = allow_tf32,
				)

		# sum separate dFreqs buffers and reduce to original shape

		if exists(dfreqs_q) and exists(dfreqs_k):
			dfreqs = dfreqs_q + dfreqs_k
			ndim: int = len(ctx.orig_freqs_shape)
			if ndim == 2:
				dfreqs_out: torch.Tensor | None = dfreqs.sum(dim=(0, 1)).to(q.dtype)
			elif ndim == 3:
				dfreqs_out = dfreqs.sum(dim=1).to(q.dtype)
			else:
				dfreqs_out = dfreqs.to(q.dtype)
		else:
			dfreqs_out = None

		dbias_out: torch.Tensor | None = dbias.to(q.dtype) if exists(dbias) else None
		return dq.to(q.dtype), dk, dfreqs_out, dbias_out, None, None

# public api

def triton_compute_qk_similarity(q: torch.Tensor, k: torch.Tensor, freqs: torch.Tensor, bias: torch.Tensor, rotate_dim: int, *, allow_tf32: bool = True) -> torch.Tensor:
	q_heads: int = q.shape[1]
	k_heads: int = k.shape[1]
	if not divisible_by(q_heads, k_heads):
		message: str = f"I received `{q_heads = }` and `{k_heads = }`, but I need `q_heads` to be divisible by `k_heads` for grouped-query attention."
		raise ValueError(message)

	q, k = q.contiguous(), k.contiguous()

	groups: int = q.shape[1] // k.shape[1]
	k = repeat(k, 'b h ... -> b (g h) ...', g=groups)
	bias = repeat(bias, 'h ... -> (g h) ...', g=groups)

	return PoPESimilarityFunction.apply(q, k, freqs, bias, rotate_dim, allow_tf32)
