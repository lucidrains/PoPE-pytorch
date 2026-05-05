from __future__ import annotations

import os
from collections.abc import Callable, Hashable, Sequence
from typing import Any

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.autograd import Function
from torch.autograd.function import FunctionCtx
from torch_einops_kit import default, exists
from triton.runtime.autotuner import Autotuner

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
	cache: dict[tuple[int | Hashable, ...], Autotuner] = {}
	def inner(kernel_fn: Hashable, *args: Hashable) -> Autotuner:
		key: tuple[int | Hashable, ...] = (id(kernel_fn), *args)
		if key not in cache:
			cache[key] = fn(kernel_fn, *args)
		return cache[key]

	return inner

@cache_by_id
def get_autotuned_kernel(kernel_fn: triton.JITFunction[Any], configs_fn: Callable[[], list[triton.Config]], keys: Sequence[str], blk_d: int, elem_bytes: int, device_idx: int = 0) -> Autotuner:
	configs: list[triton.Config] = configs_fn()
	configs = _filter_configs(configs, blk_d, elem_bytes, device_idx)
	return triton.autotune(configs, key = list(keys))(kernel_fn)

def _fwd_configs() -> list[triton.Config]:
	return [
		triton.Config({'BM': 64, 'BN': 64}, num_stages=1, num_warps=8),
		triton.Config({'BM': 64, 'BN': 32}, num_stages=2, num_warps=4),
		triton.Config({'BM': 32, 'BN': 64}, num_stages=2, num_warps=4),
		triton.Config({'BM': 32, 'BN': 32}, num_stages=2, num_warps=4),
		triton.Config({'BM': 16, 'BN': 16}, num_stages=1, num_warps=4),
	]

def _bwd_pre_hook(nargs: dict[str, Tensor | int | float | bool | None]) -> None:
	nargs['DQ'].zero_()
	df: Tensor | None = nargs.get('DFreqs')
	if df is not None:
		df.zero_()
	dpb: Tensor | None = nargs.get('DPopeBias')
	if dpb is not None:
		dpb.zero_()

def _bwd_configs() -> list[triton.Config]:
	return [
		triton.Config({'BM': 64, 'BN': 32}, num_stages=1, num_warps=4, pre_hook=_bwd_pre_hook),
		triton.Config({'BM': 32, 'BN': 64}, num_stages=1, num_warps=4, pre_hook=_bwd_pre_hook),
		triton.Config({'BM': 32, 'BN': 32}, num_stages=2, num_warps=4, pre_hook=_bwd_pre_hook),
		triton.Config({'BM': 32, 'BN': 32}, num_stages=1, num_warps=4, pre_hook=_bwd_pre_hook),
		triton.Config({'BM': 16, 'BN': 16}, num_stages=1, num_warps=4, pre_hook=_bwd_pre_hook),
	]

# stride helpers

def _freq_strides(freqs: Tensor | None) -> tuple[int, int, int]:
	if not exists(freqs):
		return (0, 0, 0)
	if freqs.ndim == 2:
		return (0, 0, freqs.stride(0))
	if freqs.ndim == 3:
		return (freqs.stride(0), 0, freqs.stride(1))
	return (freqs.stride(0), freqs.stride(2), freqs.stride(1))

def _mask_strides(mask: Tensor | None) -> tuple[int, int]:
	if not exists(mask):
		return (0, 0)
	return (mask.stride(0), mask.stride(1))

# activation helpers

@triton.jit
def _softplus(x: tl.tensor) -> tl.tensor:
	return tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))

@triton.jit
def _softplus_grad(x: tl.tensor) -> tl.tensor:
	return tl.sigmoid(x)

@triton.jit
def _apply_softplus(x: tl.tensor, mask_r: tl.tensor) -> tl.tensor:
	return tl.where(mask_r[None, :], _softplus(x.to(tl.float32)).to(x.dtype), x)

@triton.jit
def _apply_rotations(act: tl.tensor, freq: tl.tensor, mask_r: tl.tensor) -> tuple[tl.tensor, tl.tensor]:
	cos: tl.tensor = tl.where(mask_r[None, :], act * tl.cos(freq).to(act.dtype), act)
	sin: tl.tensor = tl.where(mask_r[None, :], act * tl.sin(freq).to(act.dtype), 0.0)
	return cos, sin

@triton.heuristics({
		'EVEN_M': lambda args: args['seqlen_q'] % args['BM'] == 0,
		'EVEN_N': lambda args: args['seqlen_k'] % args['BN'] == 0,
		'EVEN_HEADDIM': lambda args: args['headdim'] == args['BLOCK_HEADDIM'],
})
@triton.jit
def _fwd_kernel(
	Q: tl.tensor, K: tl.tensor, V: tl.tensor, Freqs: tl.tensor, PopeBias: tl.tensor, Out: tl.tensor, Lse: tl.tensor, Mask: tl.tensor,
	softmax_scale: float,
	stride_qb: int, stride_qh: int, stride_qm: int,
	stride_kb: int, stride_kh: int, stride_kn: int,
	stride_vb: int, stride_vh: int, stride_vn: int,
	stride_fb: int, stride_fh: int, stride_fi: int,
	stride_pbh: int,
	stride_ob: int, stride_oh: int, stride_om: int,
	stride_kmb: int, stride_kmn: int,
	n_heads: int, seqlen_q: int, seqlen_k: int, headdim: int, rotate_dim: int, dropout_p: float, drop_seed: int,
	HAS_POPE: tl.constexpr, IS_CAUSAL: tl.constexpr, HAS_MASK: tl.constexpr, IS_DROPOUT: tl.constexpr,
	BLOCK_HEADDIM: tl.constexpr, EVEN_M: tl.constexpr, EVEN_N: tl.constexpr, EVEN_HEADDIM: tl.constexpr,
	BM: tl.constexpr, BN: tl.constexpr,
) -> None:
	bhid: tl.tensor = tl.program_id(1)
	b: tl.tensor = bhid // n_heads
	h: tl.tensor = bhid % n_heads
	blk_m: tl.tensor = tl.program_id(0)

	off_m: tl.tensor = blk_m * BM + tl.arange(0, BM)
	off_n: tl.tensor = tl.arange(0, BN)
	off_d: tl.tensor = tl.arange(0, BLOCK_HEADDIM)

	mask_m: tl.tensor = off_m < seqlen_q
	mask_d: tl.tensor = off_d < headdim
	mask_r: tl.tensor = off_d < rotate_dim

	# load q

	q_ptr: tl.tensor = Q + b * stride_qb + h * stride_qh + off_m[:, None] * stride_qm + off_d[None, :]

	if EVEN_M & EVEN_HEADDIM:
		q: tl.tensor = tl.load(q_ptr)
	else:
		q = tl.load(q_ptr, mask=mask_m[:, None] & mask_d[None, :], other=0.0)

	# apply pope rotary to q

	q_off: int = seqlen_k - seqlen_q

	if HAS_POPE:
		q = _apply_softplus(q, mask_r)
		fq: tl.tensor = tl.load(Freqs + b * stride_fb + h * stride_fh + (q_off + off_m[:, None]) * stride_fi + off_d[None, :], mask = mask_m[:, None] & mask_r[None, :], other = 0.0).to(tl.float32)
		q_cos, q_sin = _apply_rotations(q, fq, mask_r)
	else:
		q_cos: tl.tensor = q
		q_sin: tl.tensor | None = None

	# online softmax accumulators

	max_i: tl.tensor = tl.zeros([BM], tl.float32) - float('inf')
	sum_i: tl.tensor = tl.zeros([BM], tl.float32)
	acc: tl.tensor = tl.zeros([BM, BLOCK_HEADDIM], tl.float32)

	if not IS_CAUSAL:
		end_n: int = seqlen_k
	else:
		end_n = tl.minimum((blk_m + 1) * BM + q_off, seqlen_k)

	for start_n in range(0, end_n, BN):
		col_n: tl.tensor = start_n + off_n
		cmask: tl.tensor = col_n < seqlen_k

		# load k

		k_ptr: tl.tensor = K + b * stride_kb + h * stride_kh + col_n[:, None] * stride_kn + off_d[None, :]

		if EVEN_N & EVEN_HEADDIM:
			k: tl.tensor = tl.load(k_ptr)
		else:
			k = tl.load(k_ptr, mask=cmask[:, None] & mask_d[None, :], other=0.0)

		# compute qk

		if HAS_POPE:
			k = _apply_softplus(k, mask_r)
			fk: tl.tensor = tl.load(Freqs + b * stride_fb + h * stride_fh + col_n[:, None] * stride_fi + off_d[None, :], mask = cmask[:, None] & mask_r[None, :], other = 0.0)
			bias: tl.tensor = tl.load(PopeBias + h * stride_pbh + off_d, mask = mask_r, other = 0.0)
			th_k: tl.tensor = (fk + bias[None, :]).to(tl.float32)
			k_cos, k_sin = _apply_rotations(k, th_k, mask_r)
			qk: tl.tensor = tl.dot(q_cos, tl.trans(k_cos)) + tl.dot(q_sin, tl.trans(k_sin))
		else:
			qk = tl.dot(q_cos, tl.trans(k))

		qk *= softmax_scale

		# masking

		if IS_CAUSAL:
			qk += tl.where(off_m[:, None] + q_off >= col_n[None, :], 0, float('-inf'))

		if HAS_MASK:
			mask: tl.tensor = tl.load(Mask + b * stride_kmb + col_n * stride_kmn, mask=cmask, other=False)
			qk += tl.where(mask[None, :], 0, float('-inf'))

		if not EVEN_N:
			qk += tl.where(cmask[None, :], 0, float('-inf'))

		# online softmax update

		m_j: tl.tensor = tl.max(qk, 1)
		prob: tl.tensor = tl.exp(qk - tl.where(m_j == float('-inf'), 0.0, m_j)[:, None])
		prob = tl.where(m_j[:, None] == float('-inf'), 0.0, prob)
		l_j: tl.tensor = tl.sum(prob, 1)

		m_new: tl.tensor = tl.maximum(max_i, m_j)
		m_safe: tl.tensor = tl.where(m_new == float('-inf'), 0.0, m_new)
		alpha: tl.tensor = tl.exp(max_i - m_safe)
		beta: tl.tensor = tl.exp(m_j - m_safe)

		acc *= alpha[:, None]

		# load v and accumulate

		v_ptr: tl.tensor = V + b * stride_vb + h * stride_vh + col_n[:, None] * stride_vn + off_d[None, :]

		if EVEN_N & EVEN_HEADDIM:
			v: tl.tensor = tl.load(v_ptr)
		else:
			v = tl.load(v_ptr, mask=cmask[:, None] & mask_d[None, :], other=0.0)

		if IS_DROPOUT:
			drop_offset: tl.tensor = (bhid * seqlen_q + off_m[:, None]) * seqlen_k + col_n[None, :]
			keep: tl.tensor = tl.rand(drop_seed, drop_offset) > dropout_p
			prob = tl.where(keep, prob / (1.0 - dropout_p), 0.0)

		acc += tl.dot(prob.to(v.dtype), v) * beta[:, None]
		sum_i = sum_i * alpha + l_j * beta
		max_i = m_new

	# normalize and store

	acc /= tl.where(sum_i == 0.0, 1.0, sum_i)[:, None]

	tl.store(Out + b * stride_ob + h * stride_oh + off_m[:, None] * stride_om + off_d[None, :], acc.to(Out.dtype.element_ty), mask = mask_m[:, None] & mask_d[None, :])
	tl.store(Lse + bhid * seqlen_q + off_m, max_i + tl.log(sum_i), mask = mask_m)

# backward preprocess - compute delta = rowsum(o * do)

@triton.jit
def _bwd_preprocess(
	Out: tl.tensor, DO: tl.tensor, Delta: tl.tensor,
	stride_ob: int, stride_oh: int, stride_om: int,
	stride_db: int, stride_dh: int, stride_dm: int,
	n_heads: int, seqlen_q: int, d: int,
	BM: tl.constexpr, BLOCK_D: tl.constexpr,
) -> None:
	bhid: tl.tensor = tl.program_id(1)
	b: tl.tensor = bhid // n_heads
	h: tl.tensor = bhid % n_heads

	off_m: tl.tensor = tl.program_id(0) * BM + tl.arange(0, BM)
	off_d: tl.tensor = tl.arange(0, BLOCK_D)
	mask: tl.tensor = (off_m < seqlen_q)[:, None] & (off_d < d)[None, :]

	o: tl.tensor = tl.load(Out + b * stride_ob + h * stride_oh + off_m[:, None] * stride_om + off_d[None, :], mask=mask, other=0.0).to(tl.float32)
	do: tl.tensor = tl.load(DO + b * stride_db + h * stride_dh + off_m[:, None] * stride_dm + off_d[None, :], mask=mask, other=0.0).to(tl.float32)

	tl.store(Delta + bhid * seqlen_q + off_m, tl.sum(o * do, 1), mask=off_m < seqlen_q)

# backward kernel

@triton.heuristics({
		'EVEN_M': lambda args: args['seqlen_q'] % args['BM'] == 0,
		'EVEN_N': lambda args: args['seqlen_k'] % args['BN'] == 0,
		'EVEN_HEADDIM': lambda args: args['headdim'] == args['BLOCK_HEADDIM'],
})
@triton.jit
def _bwd_kernel(
	Q: tl.tensor, K: tl.tensor, V: tl.tensor, Freqs: tl.tensor, PopeBias: tl.tensor, DO: tl.tensor, DQ: tl.tensor, DK: tl.tensor, DV: tl.tensor, DFreqs: tl.tensor, DPopeBias: tl.tensor, Lse: tl.tensor, Delta: tl.tensor, Mask: tl.tensor,
	softmax_scale: float,
	stride_qb: int, stride_qh: int, stride_qm: int,
	stride_kb: int, stride_kh: int, stride_kn: int,
	stride_vb: int, stride_vh: int, stride_vn: int,
	stride_fb: int, stride_fh: int, stride_fi: int,
	stride_pbh: int,
	stride_db: int, stride_dh: int, stride_dm: int,
	stride_dqb: int, stride_dqh: int, stride_dqm: int,
	stride_dkb: int, stride_dkh: int, stride_dkn: int,
	stride_dvb: int, stride_dvh: int, stride_dvn: int,
	stride_dfb: int, stride_dfh: int, stride_dfi: int,
	stride_kmb: int, stride_kmn: int,
	n_heads: int, seqlen_q: int, seqlen_k: int, headdim: int, rotate_dim: int, dropout_p: float, drop_seed: int,
	HAS_POPE: tl.constexpr, IS_CAUSAL: tl.constexpr, HAS_MASK: tl.constexpr, IS_DROPOUT: tl.constexpr,
	BLOCK_HEADDIM: tl.constexpr, EVEN_M: tl.constexpr, EVEN_N: tl.constexpr, EVEN_HEADDIM: tl.constexpr,
	BM: tl.constexpr, BN: tl.constexpr,
) -> None:
	bhid: tl.tensor = tl.program_id(1)
	b: tl.tensor = bhid // n_heads
	h: tl.tensor = bhid % n_heads
	blk_n: tl.tensor = tl.program_id(0)

	off_m: tl.tensor = tl.arange(0, BM)
	off_n: tl.tensor = blk_n * BN + tl.arange(0, BN)
	off_d: tl.tensor = tl.arange(0, BLOCK_HEADDIM)

	mask_n: tl.tensor = off_n < seqlen_k
	mask_d: tl.tensor = off_d < headdim
	mask_r: tl.tensor = off_d < rotate_dim

	# load k, v for this block

	k: tl.tensor = tl.load(K + b * stride_kb + h * stride_kh + off_n[:, None] * stride_kn + off_d[None, :], mask = mask_n[:, None] & mask_d[None, :], other = 0.0)
	v: tl.tensor = tl.load(V + b * stride_vb + h * stride_vh + off_n[:, None] * stride_vn + off_d[None, :], mask = mask_n[:, None] & mask_d[None, :], other = 0.0)

	# apply pope rotary to k

	if HAS_POPE:
		act_k: tl.tensor = _apply_softplus(k, mask_r)
		fk: tl.tensor = tl.load(Freqs + b * stride_fb + h * stride_fh + off_n[:, None] * stride_fi + off_d[None, :], mask = mask_n[:, None] & mask_r[None, :], other = 0.0)
		bias: tl.tensor = tl.load(PopeBias + h * stride_pbh + off_d, mask = mask_r, other = 0.0)
		th_k: tl.tensor = (fk + bias[None, :]).to(tl.float32)
		k_cos, k_sin = _apply_rotations(act_k, th_k, mask_r)
	else:
		k_cos: tl.tensor = k
		k_sin: tl.tensor | None = None

	# gradient accumulators

	d_v: tl.tensor = tl.zeros([BN, BLOCK_HEADDIM], tl.float32)
	d_k: tl.tensor = tl.zeros([BN, BLOCK_HEADDIM], tl.float32)

	q_off: int = seqlen_k - seqlen_q

	# iterate over q blocks

	for start_m in range(0, seqlen_q, BM):
		cur_m: tl.tensor = start_m + off_m
		mask_m: tl.tensor = cur_m < seqlen_q

		q: tl.tensor = tl.load(Q + b * stride_qb + h * stride_qh + cur_m[:, None] * stride_qm + off_d[None, :], mask = mask_m[:, None] & mask_d[None, :], other = 0.0)

		# recompute attention

		if HAS_POPE:
			act_q: tl.tensor = _apply_softplus(q, mask_r)
			fq: tl.tensor = tl.load(Freqs + b * stride_fb + h * stride_fh + (q_off + cur_m[:, None]) * stride_fi + off_d[None, :], mask = mask_m[:, None] & mask_r[None, :], other = 0.0).to(tl.float32)
			q_cos, q_sin = _apply_rotations(act_q, fq, mask_r)
			qk: tl.tensor = tl.dot(q_cos, tl.trans(k_cos)) + tl.dot(q_sin, tl.trans(k_sin))
		else:
			qk = tl.dot(q, tl.trans(k))

		qk *= softmax_scale

		if IS_CAUSAL:
			qk += tl.where(cur_m[:, None] + q_off >= off_n[None, :], 0, float('-inf'))

		if HAS_MASK:
			mask: tl.tensor = tl.load(Mask + b * stride_kmb + off_n * stride_kmn, mask=mask_n, other=False)
			qk += tl.where(mask[None, :], 0, float('-inf'))

		# recompute prob from lse

		lse: tl.tensor = tl.load(Lse + bhid * seqlen_q + cur_m, mask=mask_m, other=float('-inf'))
		prob: tl.tensor = tl.exp(qk - tl.where(lse == float('-inf'), 0.0, lse)[:, None])
		prob = tl.where((lse[:, None] == float('-inf')) | (~mask_m[:, None]), 0.0, prob)

		# dv, dp

		do: tl.tensor = tl.load(DO + b * stride_db + h * stride_dh + cur_m[:, None] * stride_dm + off_d[None, :], mask = mask_m[:, None] & mask_d[None, :], other = 0.0)

		if IS_DROPOUT:
			drop_offset: tl.tensor = (bhid * seqlen_q + cur_m[:, None]) * seqlen_k + off_n[None, :]
			keep: tl.tensor = tl.rand(drop_seed, drop_offset) > dropout_p
			prob_drop: tl.tensor = tl.where(keep, prob / (1.0 - dropout_p), 0.0)
		else:
			prob_drop = prob

		d_v += tl.dot(tl.trans(prob_drop.to(do.dtype)), do)
		dp: tl.tensor = tl.dot(do.to(prob.dtype), tl.trans(v.to(prob.dtype)))

		if IS_DROPOUT:
			dp = tl.where(keep, dp / (1.0 - dropout_p), 0.0)

		delta: tl.tensor = tl.load(Delta + bhid * seqlen_q + cur_m, mask=mask_m, other=0.0)
		ds: tl.tensor = prob * (dp - delta[:, None]) * softmax_scale

		# dq, dk gradients

		if HAS_POPE:
			dqkc: tl.tensor = tl.dot(ds.to(q_cos.dtype), k_cos)
			dqks: tl.tensor = tl.dot(ds.to(k_sin.dtype), k_sin)
			dq: tl.tensor = tl.where(mask_r[None, :], (dqkc * tl.cos(fq).to(dqkc.dtype) + dqks * tl.sin(fq).to(dqks.dtype)) * _softplus_grad(q.to(tl.float32)).to(q.dtype), dqkc)

			dkkc: tl.tensor = tl.dot(tl.trans(ds.to(q_cos.dtype)), q_cos)
			dkks: tl.tensor = tl.dot(tl.trans(ds.to(q_sin.dtype)), q_sin)
			d_k += tl.where(mask_r[None, :], (dkkc * tl.cos(th_k).to(dkkc.dtype) + dkks * tl.sin(th_k).to(dkks.dtype)) * _softplus_grad(k.to(tl.float32)).to(k.dtype), dkkc)

			# dfreqs, dpope_bias via atomic_add

			dfq: tl.tensor = (dqks.to(tl.float32) * q_cos.to(tl.float32) - dqkc.to(tl.float32) * q_sin.to(tl.float32)).to(DFreqs.dtype.element_ty)
			tl.atomic_add(DFreqs + b * stride_dfb + h * stride_dfh + (q_off + cur_m[:, None]) * stride_dfi + off_d[None, :], dfq, mask = mask_m[:, None] & mask_r[None, :])

			dfk: tl.tensor = (dkks.to(tl.float32) * k_cos.to(tl.float32) - dkkc.to(tl.float32) * k_sin.to(tl.float32)).to(DFreqs.dtype.element_ty)
			tl.atomic_add(DFreqs + b * stride_dfb + h * stride_dfh + off_n[:, None] * stride_dfi + off_d[None, :], dfk, mask = mask_n[:, None] & mask_r[None, :])
			tl.atomic_add(DPopeBias + h * stride_pbh + off_d, tl.sum(dfk, 0), mask = mask_r)
		else:
			dq: tl.tensor = tl.dot(ds.to(k.dtype), k)
			d_k += tl.dot(tl.trans(ds.to(q.dtype)), q)

		# dq via atomic_add (accumulated across k-blocks)

		tl.atomic_add(DQ + b * stride_dqb + h * stride_dqh + cur_m[:, None] * stride_dqm + off_d[None, :], dq.to(DQ.dtype.element_ty), mask = mask_m[:, None] & mask_d[None, :])

	# store dk, dv

	tl.store(DV + b * stride_dvb + h * stride_dvh + off_n[:, None] * stride_dvn + off_d[None, :], d_v.to(DV.dtype.element_ty), mask = mask_n[:, None] & mask_d[None, :])
	tl.store(DK + b * stride_dkb + h * stride_dkh + off_n[:, None] * stride_dkn + off_d[None, :], d_k.to(DK.dtype.element_ty), mask = mask_n[:, None] & mask_d[None, :])

# wrapper functions

def flash_attn_forward(
	q: Tensor,
	k: Tensor,
	v: Tensor,
	freqs: Tensor | None = None,
	pope_bias: Tensor | None = None,
	mask: Tensor | None = None,
	*,
	causal: bool = False,
	softmax_scale: float | None = None,
	dropout: float = 0.0,
	drop_seed: int = 0,
) -> tuple[Tensor, Tensor]:
	batch, seq_q, heads, d = q.shape
	seq_k: int = k.shape[1]

	scale: float = default(softmax_scale, d**-0.5)
	has_p: bool = exists(freqs) and exists(pope_bias)

	f_str: tuple[int, int, int] = _freq_strides(freqs) if has_p else (0, 0, 0)
	pb_str: int = pope_bias.stride(0) if has_p else 0
	rot: int = freqs.shape[-1] if has_p else 0
	m_str: tuple[int, int] = _mask_strides(mask)

	lse: Tensor = torch.empty((batch, heads, seq_q), device=q.device, dtype=torch.float32)
	o: Tensor = torch.empty_like(q)
	blk_d: int = max(triton.next_power_of_2(d), 16)
	configs: list[triton.Config] = _filter_configs(_fwd_configs(), blk_d, q.element_size(), q.device.index)

	if _NO_AUTOTUNE:
		bm, bn = configs[0].kwargs['BM'], configs[0].kwargs['BN']
		forward_launch_grid_fixed: tuple[int, int] = (triton.cdiv(seq_q, bm), batch * heads)
		_fwd_kernel[forward_launch_grid_fixed](
			q, k, v, freqs, pope_bias, o, lse, mask, scale,
			q.stride(0), q.stride(2), q.stride(1),
			k.stride(0), k.stride(2), k.stride(1),
			v.stride(0), v.stride(2), v.stride(1),
			*f_str, pb_str,
			o.stride(0), o.stride(2), o.stride(1),
			*m_str,
			heads, seq_q, seq_k, d, rot, dropout, drop_seed,
			has_p, causal, exists(mask), dropout > 0.0,
			BLOCK_HEADDIM = blk_d, BM = bm, BN = bn,
			num_warps = 4, num_stages = 1,
		)
	else:
		kernel: Autotuner = get_autotuned_kernel(_fwd_kernel, _fwd_configs, ('seqlen_q', 'seqlen_k', 'headdim'), blk_d, q.element_size(), q.device.index)
		forward_launch_grid_autotuned: Callable[[dict[str, int]], tuple[int, int]] = lambda META: (triton.cdiv(seq_q, META['BM']), batch * heads)
		kernel[forward_launch_grid_autotuned](
			q, k, v, freqs, pope_bias, o, lse, mask, scale,
			q.stride(0), q.stride(2), q.stride(1),
			k.stride(0), k.stride(2), k.stride(1),
			v.stride(0), v.stride(2), v.stride(1),
			*f_str, pb_str,
			o.stride(0), o.stride(2), o.stride(1),
			*m_str,
			heads, seq_q, seq_k, d, rot, dropout, drop_seed,
			has_p, causal, exists(mask), dropout > 0.0,
			BLOCK_HEADDIM = blk_d,
		)

	return o, lse

def flash_attn_backward(
	do: Tensor,
	q: Tensor,
	k: Tensor,
	v: Tensor,
	o: Tensor,
	lse: Tensor,
	dq: Tensor,
	dk: Tensor,
	dv: Tensor,
	dfreqs: Tensor | None = None,
	dpope_bias: Tensor | None = None,
	freqs: Tensor | None = None,
	pope_bias: Tensor | None = None,
	mask: Tensor | None = None,
	softmax_scale: float | None = None,
	dropout: float = 0.0,
	drop_seed: int = 0,
	*,
	causal: bool = False,
) -> None:
	batch, seq_q, heads, d = q.shape
	seq_k: int = k.shape[1]

	scale: float = default(softmax_scale, d**-0.5)
	blk_d: int = max(triton.next_power_of_2(d), 16)

	# preprocess: delta = rowsum(o * do)

	delta: Tensor = torch.empty_like(lse)
	bm_pre: int = 32

	_bwd_preprocess[(triton.cdiv(seq_q, bm_pre), batch * heads)](
		o, do, delta,
		o.stride(0), o.stride(2), o.stride(1),
		do.stride(0), do.stride(2), do.stride(1),
		heads, seq_q, d, bm_pre, blk_d,
	)

	has_p: bool = exists(freqs) and exists(pope_bias)

	f_str: tuple[int, int, int] = _freq_strides(freqs) if has_p else (0, 0, 0)
	df_str: tuple[int, int, int] = _freq_strides(dfreqs) if has_p and exists(dfreqs) else (0, 0, 0)
	pb_str: int = pope_bias.stride(0) if has_p else 0
	rot: int = freqs.shape[-1] if has_p else 0
	m_str: tuple[int, int] = _mask_strides(mask)

	elem_bytes: int = q.element_size()
	dev: int = q.device.index
	bwd_configs: list[triton.Config] = _filter_configs(_bwd_configs(), blk_d, elem_bytes, dev)

	if _NO_AUTOTUNE:
		bm, bn = bwd_configs[0].kwargs['BM'], bwd_configs[0].kwargs['BN']
		nw: int = 4 if d > 32 else 2
		backward_launch_grid_fixed: tuple[int, int] = (triton.cdiv(seq_k, bn), batch * heads)
		_bwd_kernel[backward_launch_grid_fixed](
			q, k, v, freqs, pope_bias, do, dq, dk, dv, dfreqs, dpope_bias, lse, delta, mask, scale,
			q.stride(0), q.stride(2), q.stride(1),
			k.stride(0), k.stride(2), k.stride(1),
			v.stride(0), v.stride(2), v.stride(1),
			*f_str, pb_str,
			do.stride(0), do.stride(2), do.stride(1),
			dq.stride(0), dq.stride(2), dq.stride(1),
			dk.stride(0), dk.stride(2), dk.stride(1),
			dv.stride(0), dv.stride(2), dv.stride(1),
			*df_str, *m_str,
			heads, seq_q, seq_k, d, rot, dropout, drop_seed,
			has_p, causal, exists(mask), dropout > 0.0,
			BLOCK_HEADDIM = blk_d, BM = bm, BN = bn,
			num_warps = nw, num_stages = 1,
		)
	else:
		kernel: Autotuner = get_autotuned_kernel(_bwd_kernel, _bwd_configs, ('seqlen_q', 'seqlen_k', 'headdim'), blk_d, elem_bytes, dev)
		backward_launch_grid_autotuned: Callable[[dict[str, int]], tuple[int, int]] = lambda META: (triton.cdiv(seq_k, META['BN']), batch * heads)
		kernel[backward_launch_grid_autotuned](
			q, k, v, freqs, pope_bias, do, dq, dk, dv, dfreqs, dpope_bias, lse, delta, mask, scale,
			q.stride(0), q.stride(2), q.stride(1),
			k.stride(0), k.stride(2), k.stride(1),
			v.stride(0), v.stride(2), v.stride(1),
			*f_str, pb_str,
			do.stride(0), do.stride(2), do.stride(1),
			dq.stride(0), dq.stride(2), dq.stride(1),
			dk.stride(0), dk.stride(2), dk.stride(1),
			dv.stride(0), dv.stride(2), dv.stride(1),
			*df_str, *m_str,
			heads, seq_q, seq_k, d, rot, dropout, drop_seed,
			has_p, causal, exists(mask), dropout > 0.0,
			BLOCK_HEADDIM = blk_d,
		)

# autograd wrapper

class FlashAttnFunction(Function):
	@staticmethod
	def forward(
		ctx: FunctionCtx,
		q: Tensor,
		k: Tensor,
		v: Tensor,
		freqs: Tensor | None = None,
		pope_bias: Tensor | None = None,
		mask: Tensor | None = None,
		causal: bool = False,
		softmax_scale: float | None = None,
		dropout: float = 0.0,
	) -> Tensor:
		drop_seed: int = int(torch.randint(0, 2**31 - 1, (1,), device=q.device).item()) if dropout > 0. else 0
		o, lse = flash_attn_forward(
			q,
			k,
			v,
			freqs=freqs,
			pope_bias=pope_bias,
			mask=mask,
			causal=causal,
			softmax_scale=softmax_scale,
			dropout=dropout,
			drop_seed=drop_seed,
		)
		ctx.save_for_backward(q, k, v, freqs, pope_bias, mask, o, lse)
		ctx.causal = causal
		ctx.softmax_scale = softmax_scale
		ctx.dropout = dropout
		ctx.drop_seed = drop_seed
		return o

	@staticmethod
	def backward(
		ctx: FunctionCtx,
		do: Tensor,
	) -> tuple[Tensor, Tensor, Tensor, Tensor | None, Tensor | None, None, None, None, None]:
		do = do.contiguous()
		q, k, v, f, pb, m, o, lse = ctx.saved_tensors

		dq: Tensor = torch.zeros_like(q, dtype=torch.float32)
		dk: Tensor = torch.zeros_like(k)
		dv: Tensor = torch.zeros_like(v)
		df: Tensor | None = torch.zeros_like(f) if exists(f) else None
		dpb: Tensor | None = torch.zeros_like(pb) if exists(pb) else None

		flash_attn_backward(do, q, k, v, o, lse, dq, dk, dv, df, dpb, f, pb, m, ctx.softmax_scale, ctx.dropout, ctx.drop_seed, causal=ctx.causal)
		return dq.to(q.dtype), dk, dv, df, dpb, None, None, None, None

# public api

def flash_attn(
	q: Tensor,
	k: Tensor,
	v: Tensor,
	freqs: Tensor | None = None,
	pope_bias: Tensor | None = None,
	mask: Tensor | None = None,
	causal: bool = False,
	softmax_scale: float | None = None,
	dropout: float = 0.0,
) -> Tensor:
	q, k, v = (t.contiguous() for t in (q, k, v))
	return FlashAttnFunction.apply(q, k, v, freqs, pope_bias, mask, causal, softmax_scale, dropout)
