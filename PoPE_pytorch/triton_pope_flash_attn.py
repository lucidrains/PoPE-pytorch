from math import ceil, sqrt
from functools import partial

import torch
from torch import Tensor
from torch.autograd import Function

from einops import repeat, rearrange

import triton
import triton.language as tl

# helpers

def exists(v):
    return v is not None

def default(val, d):
    return val if exists(val) else d

# kernels

@triton.heuristics({
    "EVEN_M": lambda args: args["seqlen_q"] % args["BM"] == 0,
    "EVEN_N": lambda args: args["seqlen_k"] % args["BN"] == 0,
    "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
})
@triton.jit
def _fwd_kernel(
    Q, K, V, Freqs, PopeBias, Out, Lse,
    softmax_scale, SQB, SQH, SQM, SKB, SKH, SKN, SVB, SVH, SVN, 
    SFB, SFH, SFI, SPBH, SOB, SOH, SOM,
    n_heads, seqlen_q, seqlen_k, headdim, rotate_dim,
    HAS_POPE: tl.constexpr, IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr, EVEN_M: tl.constexpr, EVEN_N: tl.constexpr, EVEN_HEADDIM: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr,
):
    bh = tl.program_id(1)
    b, h = bh // n_heads, bh % n_heads
    sm = tl.program_id(0)

    om, on, od = sm * BM + tl.arange(0, BM), tl.arange(0, BN), tl.arange(0, BLOCK_HEADDIM)
    mm, md, mr = om < seqlen_q, od < headdim, od < rotate_dim

    qp = Q + b * SQB + h * SQH + om[:, None] * SQM + od[None, :]

    if EVEN_M & EVEN_HEADDIM:
        q = tl.load(qp)
    else:
        q = tl.load(qp, mask=mm[:, None] & md[None, :], other=0.0)

    if HAS_POPE:
        q = tl.where(mr[None, :], tl.where(q > 20., q, tl.log(1.0 + tl.exp(q.to(tl.float32))).to(q.dtype)), q)
        q_off = seqlen_k - seqlen_q
        fqp = Freqs + b * SFB + h * SFH + (q_off + om[:, None]) * SFI + od[None, :]
        fq = tl.load(fqp, mask=mm[:, None] & mr[None, :], other=0.0).to(tl.float32)
        qc, qs = tl.where(mr[None, :], q * tl.cos(fq).to(q.dtype), q), tl.where(mr[None, :], q * tl.sin(fq).to(q.dtype), 0.0)
    else:
        qc, qs = q, None

    m_i, l_i = tl.zeros([BM], tl.float32) - float("inf"), tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, BLOCK_HEADDIM], tl.float32)

    en = seqlen_k if not IS_CAUSAL else tl.minimum((sm + 1) * BM, seqlen_k)

    for sn in range(0, en, BN):
        cn = sn + on
        mn = cn < seqlen_k

        kp = K + b * SKB + h * SKH + cn[:, None] * SKN + od[None, :]
        k = tl.load(kp, mask=mn[:, None] & md[None, :], other=0.0) if not (EVEN_N & EVEN_HEADDIM) else tl.load(kp)

        if HAS_POPE:
            k = tl.where(mr[None, :], tl.where(k > 20., k, tl.log(1.0 + tl.exp(k.to(tl.float32))).to(k.dtype)), k)
            fkp = Freqs + b * SFB + h * SFH + cn[:, None] * SFI + od[None, :]
            fk = tl.load(fkp, mask=mn[:, None] & mr[None, :], other=0.0)
            pb = tl.load(PopeBias + h * SPBH + od, mask=mr, other=0.0)
            tk = (fk + pb[None, :]).to(tl.float32)
            kc, ks = tl.where(mr[None, :], k * tl.cos(tk).to(k.dtype), k), tl.where(mr[None, :], k * tl.sin(tk).to(k.dtype), 0.0)
            qk = tl.dot(qc, tl.trans(kc)) + tl.dot(qs, tl.trans(ks))
        else:
            qk = tl.dot(qc, tl.trans(k))

        qk *= softmax_scale

        if IS_CAUSAL:
            qk += tl.where(om[:, None] >= cn[None, :], 0, float("-inf"))

        if not EVEN_N:
            qk += tl.where(mn[None, :], 0, float("-inf"))

        mij = tl.max(qk, 1)
        p = tl.exp(qk - tl.where(mij == float("-inf"), 0.0, mij)[:, None])
        p = tl.where(mij[:, None] == float("-inf"), 0.0, p)
        lij = tl.sum(p, 1)

        mi_new = tl.maximum(m_i, mij)
        mi_safe = tl.where(mi_new == float("-inf"), 0.0, mi_new)
        al, be = tl.exp(m_i - mi_safe), tl.exp(mij - mi_safe)

        acc *= al[:, None]

        vp = V + b * SVB + h * SVH + cn[:, None] * SVN + od[None, :]
        v = tl.load(vp, mask=mn[:, None] & md[None, :], other=0.0) if not (EVEN_N & EVEN_HEADDIM) else tl.load(vp)
        acc += tl.dot(p.to(v.dtype), v) * be[:, None]

        l_i = l_i * al + lij * be
        m_i = mi_new

    acc /= tl.where(l_i == 0.0, 1.0, l_i)[:, None]

    tl.store(Out + b * SOB + h * SOH + om[:, None] * SOM + od[None, :], acc.to(Out.dtype.element_ty), mask=mm[:, None] & md[None, :])
    tl.store(Lse + bh * seqlen_q + om, m_i + tl.log(l_i), mask=mm)

@triton.jit
def _bwd_preprocess(Out, DO, Delta, SOB, SOH, SOM, SDB, SDH, SDM, n_heads, seqlen_q, d, BM: tl.constexpr, BLOCK_D: tl.constexpr):
    hb = tl.program_id(1)
    b, h = hb // n_heads, hb % n_heads
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    dd = tl.arange(0, BLOCK_D)
    mask = (m < seqlen_q)[:, None] & (dd < d)[None, :]

    o = tl.load(Out + b * SOB + h * SOH + m[:, None] * SOM + dd[None, :], mask = mask, other = 0.0).to(tl.float32)
    do = tl.load(DO + b * SDB + h * SDH + m[:, None] * SDM + dd[None, :], mask = mask, other = 0.0).to(tl.float32)

    tl.store(Delta + hb * seqlen_q + m, tl.sum(o * do, 1), mask = m < seqlen_q)

@triton.heuristics({
    "EVEN_M": lambda args: args["seqlen_q"] % args["BM"] == 0,
    "EVEN_N": lambda args: args["seqlen_k"] % args["BN"] == 0,
    "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
})
@triton.jit
def _bwd_kernel(
    Q, K, V, Freqs, PopeBias, DO, DQ, DK, DV, DFreqs, DPopeBias, Lse, Delta,
    softmax_scale, SQB, SQH, SQM, SKB, SKH, SKN, SVB, SVH, SVN, SFB, SFH, SFI, SPBH,
    SDB, SDH, SDM, SDQB, SDQH, SDQM, SDKB, SDKH, SDKN, SDVB, SDVH, SDVN, SDFB, SDFH, SDFI,
    n_heads, seqlen_q, seqlen_k, headdim, rotate_dim, HAS_POPE: tl.constexpr, IS_CAUSAL: tl.constexpr, BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr, EVEN_N: tl.constexpr, EVEN_HEADDIM: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
):
    hb = tl.program_id(1)
    b, h = hb // n_heads, hb % n_heads
    sn = tl.program_id(0)

    om, on, od = tl.arange(0, BM), sn * BN + tl.arange(0, BN), tl.arange(0, BLOCK_HEADDIM)
    mn, md, mr = on < seqlen_k, od < headdim, od < rotate_dim

    k = tl.load(K + b * SKB + h * SKH + on[:, None] * SKN + od[None, :], mask = mn[:, None] & md[None, :], other = 0.0)
    v = tl.load(V + b * SVB + h * SVH + on[:, None] * SVN + od[None, :], mask = mn[:, None] & md[None, :], other = 0.0)

    if HAS_POPE:
        pk = tl.where(mr[None, :], tl.where(k > 20., k, tl.log(1.0 + tl.exp(k.to(tl.float32))).to(k.dtype)), k)
        fk = tl.load(Freqs + b * SFB + h * SFH + on[:, None] * SFI + od[None, :], mask = mn[:, None] & mr[None, :], other = 0.0)
        pb = tl.load(PopeBias + h * SPBH + od, mask = mr, other = 0.0)
        tk = (fk + pb[None, :]).to(tl.float32)
        kc, ks = tl.where(mr[None, :], pk * tl.cos(tk).to(k.dtype), pk), tl.where(mr[None, :], pk * tl.sin(tk).to(k.dtype), 0.0)
    else:
        kc, ks = k, None

    dv, dk = tl.zeros([BN, BLOCK_HEADDIM], tl.float32), tl.zeros([BN, BLOCK_HEADDIM], tl.float32)
    q_off = seqlen_k - seqlen_q

    for sm in range(0, seqlen_q, BM):
        m = sm + om
        mm = m < seqlen_q

        q = tl.load(Q + b * SQB + h * SQH + m[:, None] * SQM + od[None, :], mask = mm[:, None] & md[None, :], other = 0.0)

        if HAS_POPE:
            pq = tl.where(mr[None, :], tl.where(q > 20., q, tl.log(1.0 + tl.exp(q.to(tl.float32))).to(q.dtype)), q)
            fq = tl.load(Freqs + b * SFB + h * SFH + (q_off + m[:, None]) * SFI + od[None, :], mask = mm[:, None] & mr[None, :], other = 0.0).to(tl.float32)
            qc, qs = tl.where(mr[None, :], pq * tl.cos(fq).to(q.dtype), pq), tl.where(mr[None, :], pq * tl.sin(fq).to(q.dtype), 0.0)
            qk = tl.dot(qc, tl.trans(kc)) + tl.dot(qs, tl.trans(ks))
        else:
            qk = tl.dot(q, tl.trans(k))

        qk *= softmax_scale

        if IS_CAUSAL:
            qk += tl.where(m[:, None] >= on[None, :], 0, float("-inf"))

        lse = tl.load(Lse + hb * seqlen_q + m, mask = mm, other = float("-inf"))
        p = tl.exp(qk - tl.where(lse == float("-inf"), 0.0, lse)[:, None])
        p = tl.where((lse[:, None] == float("-inf")) | (~mm[:, None]), 0.0, p)

        do = tl.load(DO + b * SDB + h * SDH + m[:, None] * SDM + od[None, :], mask = mm[:, None] & md[None, :], other = 0.0)
        dv += tl.dot(tl.trans(p.to(do.dtype)), do)
        dp = tl.dot(do.to(p.dtype), tl.trans(v.to(p.dtype)))
        
        delta = tl.load(Delta + hb * seqlen_q + m, mask = mm, other = 0.0)
        ds = p * (dp - delta[:, None]) * softmax_scale

        if HAS_POPE:
            dqkc, dqks = tl.dot(ds.to(qc.dtype), kc), tl.dot(ds.to(ks.dtype), ks)
            dq = tl.where(mr[None, :], (dqkc * tl.cos(fq).to(dqkc.dtype) + dqks * tl.sin(fq).to(dqks.dtype)) * tl.sigmoid(q.to(tl.float32)).to(q.dtype), dqkc)
            dkkc, dkks = tl.dot(tl.trans(ds.to(qc.dtype)), qc), tl.dot(tl.trans(ds.to(qs.dtype)), qs)
            dk += tl.where(mr[None, :], (dkkc * tl.cos(tk).to(dkkc.dtype) + dkks * tl.sin(tk).to(dkks.dtype)) * tl.sigmoid(k.to(tl.float32)).to(k.dtype), dkkc)
            dfq = (dqks.to(tl.float32) * qc.to(tl.float32) - dqkc.to(tl.float32) * qs.to(tl.float32)).to(DFreqs.dtype.element_ty)
            tl.atomic_add(DFreqs + b * SDFB + h * SDFH + (q_off + m[:, None]) * SDFI + od[None, :], dfq, mask = mm[:, None] & mr[None, :])
            dfk = (dkks.to(tl.float32) * kc.to(tl.float32) - dkkc.to(tl.float32) * ks.to(tl.float32)).to(DFreqs.dtype.element_ty)
            tl.atomic_add(DFreqs + b * SDFB + h * SDFH + on[:, None] * SDFI + od[None, :], dfk, mask = mn[:, None] & mr[None, :])
            tl.atomic_add(DPopeBias + h * SPBH + od, tl.sum(dfk, 0), mask = mr)
        else:
            dq = tl.dot(ds.to(k.dtype), k)
            dk += tl.dot(tl.trans(ds.to(q.dtype)), q)

        tl.atomic_add(DQ + b * SDQB + h * SDQH + m[:, None] * SDQM + od[None, :], dq.to(DQ.dtype.element_ty), mask = mm[:, None] & md[None, :])

    tl.store(DV + b * SDVB + h * SDVH + on[:, None] * SDVN + od[None, :], dv.to(DV.dtype.element_ty), mask = mn[:, None] & md[None, :])
    tl.store(DK + b * SDKB + h * SDKH + on[:, None] * SDKN + od[None, :], dk.to(DK.dtype.element_ty), mask = mn[:, None] & md[None, :])

# wrapper functions

def flash_attn_forward(
    q, k, v,
    freqs = None,
    pope_bias = None,
    causal = False,
    softmax_scale = None
):
    batch, seq_q, heads, d = q.shape
    seq_k = k.shape[1]
    softmax_scale = default(softmax_scale, d ** -0.5)

    has_p = exists(freqs) and exists(pope_bias)
    rot = freqs.shape[-1] if has_p else 0

    if has_p:
        if freqs.ndim == 2:
            fs = (0, 0, freqs.stride(0))
        elif freqs.ndim == 3:
            fs = (freqs.stride(0), 0, freqs.stride(1))
        else:
            fs = (freqs.stride(0), freqs.stride(2), freqs.stride(1))
    else:
        fs = (0, 0, 0)

    ps = pope_bias.stride(0) if has_p else 0

    lse = torch.empty((batch, heads, seq_q), device = q.device, dtype = torch.float32)
    o = torch.empty_like(q)

    BD = max(triton.next_power_of_2(d), 16)
    BM, BN = (64, 32) if d <= 64 else (32, 32)
    num_warps = 2 if d <= 32 else 4

    _fwd_kernel[(triton.cdiv(seq_q, BM), batch * heads)](
        q, k, v, freqs, pope_bias, o, lse, softmax_scale,
        q.stride(0), q.stride(2), q.stride(1), k.stride(0), k.stride(2), k.stride(1), v.stride(0), v.stride(2), v.stride(1),
        *fs, ps, o.stride(0), o.stride(2), o.stride(1), heads, seq_q, seq_k, d, rot, has_p, causal, BD, 
        EVEN_M=True, EVEN_N=True, EVEN_HEADDIM=True, BM=BM, BN=BN, num_warps=num_warps, num_stages=1
    )

    return o, lse

def flash_attn_backward(
    do, q, k, v, o, lse,
    dq, dk, dv,
    dfreqs = None,
    dpope_bias = None,
    freqs = None,
    pope_bias = None,
    causal = False,
    softmax_scale = None
):
    batch, seq_q, heads, d = q.shape
    seq_k = k.shape[1]
    softmax_scale = default(softmax_scale, d ** -0.5)

    delta = torch.empty_like(lse)
    BD = max(triton.next_power_of_2(d), 16)

    _bwd_preprocess[(triton.cdiv(seq_q, 32), batch * heads)](
        o, do, delta,
        o.stride(0), o.stride(2), o.stride(1),
        do.stride(0), do.stride(2), do.stride(1),
        heads, seq_q, d, 32, BD
    )

    has_p = exists(freqs) and exists(pope_bias)
    rot = freqs.shape[-1] if has_p else 0

    if has_p:
        if freqs.ndim == 2:
            fs = (0, 0, freqs.stride(0))
        elif freqs.ndim == 3:
            fs = (freqs.stride(0), 0, freqs.stride(1))
        else:
            fs = (freqs.stride(0), freqs.stride(2), freqs.stride(1))

        if exists(dfreqs):
            if dfreqs.ndim == 2:
                dfs = (0, 0, dfreqs.stride(0))
            elif dfreqs.ndim == 3:
                dfs = (dfreqs.stride(0), 0, dfreqs.stride(1))
            else:
                dfs = (dfreqs.stride(0), dfreqs.stride(2), dfreqs.stride(1))
        else:
            dfs = (0, 0, 0)
    else:
        fs, dfs = (0, 0, 0), (0, 0, 0)

    ps = pope_bias.stride(0) if has_p else 0

    _bwd_kernel[(triton.cdiv(seq_k, 32), batch * heads)](
        q, k, v, freqs, pope_bias, do, dq, dk, dv, dfreqs, dpope_bias, lse, delta, softmax_scale,
        q.stride(0), q.stride(2), q.stride(1), k.stride(0), k.stride(2), k.stride(1), v.stride(0), v.stride(2), v.stride(1),
        *fs, ps, do.stride(0), do.stride(2), do.stride(1), dq.stride(0), dq.stride(2), dq.stride(1),
        dk.stride(0), dk.stride(2), dk.stride(1), dv.stride(0), dv.stride(2), dv.stride(1), *dfs, heads, seq_q, seq_k, d, rot, has_p, causal, BD, BM=32, BN=32, num_warps=2 if d <= 32 else 4, num_stages=1
    )

class FlashAttnFunction(Function):
    @staticmethod
    def forward(
        ctx,
        q, k, v,
        freqs = None,
        pope_bias = None,
        causal = False,
        softmax_scale = None
    ):
        o, lse = flash_attn_forward(q, k, v, freqs, pope_bias, causal, softmax_scale)
        ctx.save_for_backward(q, k, v, freqs, pope_bias, o, lse)
        ctx.causal, ctx.softmax_scale = causal, softmax_scale
        return o

    @staticmethod
    def backward(ctx, do):
        do = do.contiguous()
        q, k, v, f, pb, o, lse = ctx.saved_tensors

        dq, dk, dv = torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)
        df, dpb = torch.zeros_like(f) if exists(f) else None, torch.zeros_like(pb) if exists(pb) else None

        flash_attn_backward(do, q, k, v, o, lse, dq, dk, dv, df, dpb, f, pb, ctx.causal, ctx.softmax_scale)
        return dq, dk, dv, df, dpb, None, None

def flash_attn(
    q, k, v,
    freqs = None,
    pope_bias = None,
    causal = False,
    softmax_scale = None
):
    q, k, v = map(lambda t: t.contiguous(), (q, k, v))
    return FlashAttnFunction.apply(q, k, v, freqs, pope_bias, causal, softmax_scale)
