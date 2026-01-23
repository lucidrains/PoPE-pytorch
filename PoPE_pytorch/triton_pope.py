import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from einops import repeat

# helper functions

def exists(v):
    return v is not None

def divisible_by(num, den):
    return (num % den) == 0

# activation and its derivative with numerical stability

@triton.jit
def softplus_fwd(x):
    return tl.where(x > 20., x, tl.log(1. + tl.exp(x)))

@triton.jit
def softplus_bwd(x):
    return tl.sigmoid(x)

# forward kernel with optimized tiling

@triton.jit
def _fwd_kernel(
    Q, K, Freqs, Bias, Out,
    stride_qb, stride_qh, stride_qi, stride_qd,
    stride_kb, stride_kh, stride_kj, stride_kd,
    stride_fb, stride_fh, stride_fi, stride_fd,
    stride_bh, stride_bd,
    stride_ob, stride_oh, stride_oi, stride_oj,
    n_heads, seq_q, seq_k, head_dim, rotate_dim,
    BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr, BLOCK_D: tl.constexpr,
    ALLOW_TF32: tl.constexpr
):
    batch_head_idx = tl.program_id(0)
    i_block_idx = tl.program_id(1)
    j_block_idx = tl.program_id(2)

    batch_idx = batch_head_idx // n_heads
    head_idx = batch_head_idx % n_heads

    off_i = i_block_idx * BLOCK_I + tl.arange(0, BLOCK_I)
    off_j = j_block_idx * BLOCK_J + tl.arange(0, BLOCK_J)
    off_d = tl.arange(0, BLOCK_D)

    mask_i = off_i < seq_q
    mask_j = off_j < seq_k

    acc = tl.zeros((BLOCK_I, BLOCK_J), dtype = tl.float32)

    q_offset = seq_k - seq_q

    for d_offset in range(0, head_dim, BLOCK_D):
        d_mask = (d_offset + off_d) < head_dim
        rotate_mask = (d_offset + off_d) < rotate_dim

        q = tl.load(Q + batch_idx * stride_qb + head_idx * stride_qh + off_i[:, None] * stride_qi + (d_offset + off_d[None, :]) * stride_qd, mask = mask_i[:, None] & d_mask[None, :], other = 0.0)
        k = tl.load(K + batch_idx * stride_kb + head_idx * stride_kh + off_j[:, None] * stride_kj + (d_offset + off_d[None, :]) * stride_kd, mask = mask_j[:, None] & d_mask[None, :], other = 0.0)

        fq = tl.load(Freqs + batch_idx * stride_fb + (head_idx % n_heads) * stride_fh + (q_offset + off_i[:, None]) * stride_fi + (d_offset + off_d[None, :]) * stride_fd, mask = mask_i[:, None] & rotate_mask[None, :], other = 0.0)
        fk = tl.load(Freqs + batch_idx * stride_fb + (head_idx % n_heads) * stride_fh + off_j[:, None] * stride_fi + (d_offset + off_d[None, :]) * stride_fd, mask = mask_j[:, None] & rotate_mask[None, :], other = 0.0)
        
        bias = tl.load(Bias + head_idx * stride_bh + (d_offset + off_d), mask = rotate_mask, other = 0.0)

        aq = tl.where(rotate_mask[None, :], softplus_fwd(q), q)
        bk = tl.where(rotate_mask[None, :], softplus_fwd(k), k)
        
        qc = tl.where(rotate_mask[None, :], aq * tl.cos(fq), aq)
        qs = tl.where(rotate_mask[None, :], aq * tl.sin(fq), 0.0)
        
        theta_k = fk + bias[None, :]
        kc = tl.where(rotate_mask[None, :], bk * tl.cos(theta_k), bk)
        ks = tl.where(rotate_mask[None, :], bk * tl.sin(theta_k), 0.0)
        
        acc = tl.dot(qc, tl.trans(kc), acc, allow_tf32 = ALLOW_TF32)
        acc = tl.dot(qs, tl.trans(ks), acc, allow_tf32 = ALLOW_TF32)

    tl.store(Out + batch_idx * stride_ob + head_idx * stride_oh + off_i[:, None] * stride_oi + off_j[None, :] * stride_oj, acc, mask = mask_i[:, None] & mask_j[None, :])

@triton.jit
def _bwd_kernel_dqk_df(
    dQ, dK, dFreqs, dS, Q, K, Freqs, Bias,
    stride_dqb, stride_dqh, stride_dqi, stride_dqd,
    stride_dkb, stride_dkh, stride_dkj, stride_dkd,
    stride_dfb, stride_dfh, stride_dfi, stride_dfd,
    stride_sb, stride_sh, stride_si, stride_sj,
    stride_qb, stride_qh, stride_qi, stride_qd,
    stride_kb, stride_kh, stride_kj, stride_kd,
    stride_fb, stride_fh, stride_fi, stride_fd,
    stride_bh, stride_bd,
    n_heads, seq_q, seq_k, head_dim, rotate_dim,
    BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr, BLOCK_D: tl.constexpr,
    MODE: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    HAS_DF: tl.constexpr
):
    batch_head_idx = tl.program_id(0)
    grid_idx = tl.program_id(1)
    
    batch_idx = batch_head_idx // n_heads
    head_idx = batch_head_idx % n_heads
    off_d = tl.arange(0, BLOCK_D)

    q_offset = seq_k - seq_q

    if MODE == 0:
        off_outer = grid_idx * BLOCK_I + tl.arange(0, BLOCK_I)
        mask_outer = off_outer < seq_q

        for d_offset in range(0, head_dim, BLOCK_D):
            d_mask = (d_offset + off_d) < head_dim
            rotate_mask = (d_offset + off_d) < rotate_dim

            acc_dq = tl.zeros((BLOCK_I, BLOCK_D), dtype = tl.float32)
            acc_df = tl.zeros((BLOCK_I, BLOCK_D), dtype = tl.float32)
            
            q = tl.load(Q + batch_idx * stride_qb + head_idx * stride_qh + off_outer[:, None] * stride_qi + (d_offset + off_d[None, :]) * stride_qd, mask = mask_outer[:, None] & d_mask[None, :], other = 0.0)
            fq = tl.load(Freqs + batch_idx * stride_fb + head_idx * stride_fh + (q_offset + off_outer[:, None]) * stride_fi + (d_offset + off_d[None, :]) * stride_fd, mask = mask_outer[:, None] & rotate_mask[None, :], other = 0.0)
            daq_dq = tl.where(rotate_mask[None, :], softplus_bwd(q), 1.0)
            qc_f = tl.where(rotate_mask[None, :], tl.cos(fq), 1.0)
            qs_f = tl.where(rotate_mask[None, :], tl.sin(fq), 0.0)
            aq = tl.where(rotate_mask[None, :], softplus_fwd(q), q)
            
            for j_start in range(0, seq_k, BLOCK_J):
                off_inner = j_start + tl.arange(0, BLOCK_J)
                mask_inner = off_inner < seq_k

                ds = tl.load(dS + batch_idx * stride_sb + head_idx * stride_sh + off_outer[:, None] * stride_si + off_inner[None, :] * stride_sj, mask = mask_outer[:, None] & mask_inner[None, :], other = 0.0)
                ik = tl.load(K + batch_idx * stride_kb + head_idx * stride_kh + off_inner[:, None] * stride_kj + (d_offset + off_d[None, :]) * stride_kd, mask = mask_inner[:, None] & d_mask[None, :], other = 0.0)
                ifk = tl.load(Freqs + batch_idx * stride_fb + head_idx * stride_fh + off_inner[:, None] * stride_fi + (d_offset + off_d[None, :]) * stride_fd, mask = mask_inner[:, None] & rotate_mask[None, :], other = 0.0)
                ibias = tl.load(Bias + head_idx * stride_bh + (d_offset + off_d), mask = rotate_mask, other = 0.0)
                
                ibk = tl.where(rotate_mask[None, :], softplus_fwd(ik), ik)
                theta_k = ifk + ibias[None, :]
                ikc_f = tl.where(rotate_mask[None, :], tl.cos(theta_k), 1.0)
                iks_f = tl.where(rotate_mask[None, :], tl.sin(theta_k), 0.0)
                
                v1 = tl.dot(ds, (ibk * ikc_f).to(tl.float32), allow_tf32 = ALLOW_TF32)
                v2 = tl.dot(ds, (ibk * iks_f).to(tl.float32), allow_tf32 = ALLOW_TF32)
                
                acc_dq += daq_dq * (qc_f * v1 + qs_f * v2)
                if HAS_DF: acc_df += aq * (qc_f * v2 - qs_f * v1)
            
            tl.store(dQ + batch_idx * stride_dqb + head_idx * stride_dqh + off_outer[:, None] * stride_dqi + (d_offset + off_d[None, :]) * stride_dqd, acc_dq, mask = mask_outer[:, None] & d_mask[None, :])
            if HAS_DF:
                tl.atomic_add(dFreqs + batch_idx * stride_dfb + head_idx * stride_dfh + (q_offset + off_outer[:, None]) * stride_dfi + (d_offset + off_d[None, :]) * stride_dfd, acc_df, mask = mask_outer[:, None] & rotate_mask[None, :])
            
    else: # MODE == 1
        off_outer = grid_idx * BLOCK_J + tl.arange(0, BLOCK_J)
        mask_outer = off_outer < seq_k

        for d_offset in range(0, head_dim, BLOCK_D):
            d_mask = (d_offset + off_d) < head_dim
            rotate_mask = (d_offset + off_d) < rotate_dim

            acc_dk = tl.zeros((BLOCK_J, BLOCK_D), dtype = tl.float32)
            acc_df = tl.zeros((BLOCK_J, BLOCK_D), dtype = tl.float32)
            
            k = tl.load(K + batch_idx * stride_kb + head_idx * stride_kh + off_outer[:, None] * stride_kj + (d_offset + off_d[None, :]) * stride_kd, mask = mask_outer[:, None] & d_mask[None, :], other = 0.0)
            fk = tl.load(Freqs + batch_idx * stride_fb + head_idx * stride_fh + off_outer[:, None] * stride_fi + (d_offset + off_d[None, :]) * stride_fd, mask = mask_outer[:, None] & rotate_mask[None, :], other = 0.0)
            bias = tl.load(Bias + head_idx * stride_bh + (d_offset + off_d), mask = rotate_mask, other = 0.0)
            
            dbk_dk = tl.where(rotate_mask[None, :], softplus_bwd(k), 1.0)
            theta_k = fk + bias[None, :]
            kc_f = tl.where(rotate_mask[None, :], tl.cos(theta_k), 1.0)
            ks_f = tl.where(rotate_mask[None, :], tl.sin(theta_k), 0.0)
            bk = tl.where(rotate_mask[None, :], softplus_fwd(k), k)
            
            for inner_start in range(0, seq_q, BLOCK_I):
                off_inner = inner_start + tl.arange(0, BLOCK_I)
                mask_inner = off_inner < seq_q

                ds = tl.load(dS + batch_idx * stride_sb + head_idx * stride_sh + off_inner[:, None] * stride_si + off_outer[None, :] * stride_sj, mask = mask_inner[:, None] & mask_outer[None, :], other = 0.0)
                iq = tl.load(Q + batch_idx * stride_qb + head_idx * stride_qh + off_inner[:, None] * stride_qi + (d_offset + off_d[None, :]) * stride_qd, mask = mask_inner[:, None] & d_mask[None, :], other = 0.0)
                ifq = tl.load(Freqs + batch_idx * stride_fb + (head_idx % n_heads) * stride_fh + (q_offset + off_inner[:, None]) * stride_fi + (d_offset + off_d[None, :]) * stride_fd, mask = mask_inner[:, None] & rotate_mask[None, :], other = 0.0)
                
                iaq = tl.where(rotate_mask[None, :], softplus_fwd(iq), iq)
                iqc_f = tl.where(rotate_mask[None, :], tl.cos(ifq), 1.0)
                iqs_f = tl.where(rotate_mask[None, :], tl.sin(ifq), 0.0)
                
                v1 = tl.dot(tl.trans(ds), (iaq * iqc_f).to(tl.float32), allow_tf32 = ALLOW_TF32)
                v2 = tl.dot(tl.trans(ds), (iaq * iqs_f).to(tl.float32), allow_tf32 = ALLOW_TF32)
                
                acc_dk += dbk_dk * (kc_f * v1 + ks_f * v2)
                if HAS_DF: acc_df += bk * (kc_f * v2 - ks_f * v1)
            
            tl.store(dK + batch_idx * stride_dkb + head_idx * stride_dkh + off_outer[:, None] * stride_dkj + (d_offset + off_d[None, :]) * stride_dkd, acc_dk, mask = mask_outer[:, None] & d_mask[None, :])
            if HAS_DF:
                tl.atomic_add(dFreqs + batch_idx * stride_dfb + head_idx * stride_dfh + off_outer[:, None] * stride_dfi + (d_offset + off_d[None, :]) * stride_dfd, acc_df, mask = mask_outer[:, None] & rotate_mask[None, :])

@triton.jit
def _bwd_kernel_dbias_optimized(
    dBias, dS, Q, K, Freqs, Bias,
    stride_sb, stride_sh, stride_si, stride_sj,
    stride_qb, stride_qh, stride_qi, stride_qd,
    stride_kb, stride_kh, stride_kj, stride_kd,
    stride_fb, stride_fh, stride_fi, stride_fd,
    stride_bh, stride_bd,
    batch, n_heads, seq_q, seq_k, head_dim, rotate_dim,
    BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr, BLOCK_D: tl.constexpr,
    ALLOW_TF32: tl.constexpr
):
    batch_head_idx = tl.program_id(0)
    i_block_idx = tl.program_id(1)

    batch_idx = batch_head_idx // n_heads
    head_idx = batch_head_idx % n_heads

    off_i = i_block_idx * BLOCK_I + tl.arange(0, BLOCK_I)
    mask_i = off_i < seq_q
    off_d = tl.arange(0, BLOCK_D)

    q_offset = seq_k - seq_q

    for d_offset in range(0, head_dim, BLOCK_D):
        d_mask = (d_offset + off_d) < head_dim
        rotate_mask = (d_offset + off_d) < rotate_dim
        
        q = tl.load(Q + batch_idx * stride_qb + head_idx * stride_qh + off_i[:, None] * stride_qi + (d_offset + off_d[None, :]) * stride_qd, mask = mask_i[:, None] & d_mask[None, :], other = 0.0)
        fq = tl.load(Freqs + batch_idx * stride_fb + (head_idx % n_heads) * stride_fh + (q_offset + off_i[:, None]) * stride_fi + (d_offset + off_d[None, :]) * stride_fd, mask = mask_i[:, None] & rotate_mask[None, :], other = 0.0)
        aq = tl.where(rotate_mask[None, :], softplus_fwd(q), q)
        qc_f = tl.where(rotate_mask[None, :], tl.cos(fq), 1.0)
        qs_f = tl.where(rotate_mask[None, :], tl.sin(fq), 0.0)
        
        local_dbias = tl.zeros((BLOCK_D,), dtype = tl.float32)
        bias = tl.load(Bias + head_idx * stride_bh + (d_offset + off_d), mask = rotate_mask, other = 0.0)

        for j_start in range(0, seq_k, BLOCK_J):
            off_j = j_start + tl.arange(0, BLOCK_J)
            mask_j = off_j < seq_k

            ds = tl.load(dS + batch_idx * stride_sb + head_idx * stride_sh + off_i[:, None] * stride_si + off_j[None, :] * stride_sj, mask = mask_i[:, None] & mask_j[None, :], other = 0.0)
            ik = tl.load(K + batch_idx * stride_kb + head_idx * stride_kh + off_j[:, None] * stride_kj + (d_offset + off_d[None, :]) * stride_kd, mask = mask_j[:, None] & d_mask[None, :], other = 0.0)
            ifk = tl.load(Freqs + batch_idx * stride_fb + (head_idx % n_heads) * stride_fh + off_j[:, None] * stride_fi + (d_offset + off_d[None, :]) * stride_fd, mask = mask_j[:, None] & rotate_mask[None, :], other = 0.0)
            
            ibk = tl.where(rotate_mask[None, :], softplus_fwd(ik), ik)
            theta_k = ifk + bias[None, :]
            ikc_f = tl.where(rotate_mask[None, :], tl.cos(theta_k), 1.0)
            iks_f = tl.where(rotate_mask[None, :], tl.sin(theta_k), 0.0)
            
            A_id = (aq * qs_f).to(tl.float32)
            B_id = (aq * qc_f).to(tl.float32)
            K_jd = (ibk * ikc_f).to(tl.float32)
            L_jd = (ibk * iks_f).to(tl.float32)
            
            X_jd = tl.dot(tl.trans(ds), A_id, allow_tf32 = ALLOW_TF32)
            Y_jd = tl.dot(tl.trans(ds), B_id, allow_tf32 = ALLOW_TF32)
            local_dbias += tl.sum(K_jd * X_jd - L_jd * Y_jd, axis = 0)

        if d_offset < rotate_dim:
            tl.atomic_add(dBias + head_idx * stride_bh + (d_offset + off_d), local_dbias, mask = d_mask & rotate_mask)

class PoPESimilarityFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, freqs, bias, rotate_dim, allow_tf32):
        b, h, seq_q, d = q.shape
        _, _, seq_k, _ = k.shape
        ctx.orig_freqs_shape = freqs.shape
        ctx.freqs_requires_grad = freqs.requires_grad
        ctx.bias_requires_grad = bias.requires_grad

        if freqs.ndim == 2:
            freqs = freqs.view(1, 1, freqs.shape[0], rotate_dim).expand(b, h, freqs.shape[0], rotate_dim)
        elif freqs.ndim == 3:
            freqs = freqs.view(freqs.shape[0], 1, freqs.shape[1], rotate_dim).expand(b, h, freqs.shape[1], rotate_dim)
        
        freqs = freqs.contiguous()
        sim = torch.empty((b, h, seq_q, seq_k), device = q.device, dtype = q.dtype)
        grid = (b * h, triton.cdiv(seq_q, 32), triton.cdiv(seq_k, 32))
        
        _fwd_kernel[grid](
            q, k, freqs, bias, sim,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            freqs.stride(0), freqs.stride(1), freqs.stride(2), freqs.stride(3),
            bias.stride(0), bias.stride(1),
            sim.stride(0), sim.stride(1), sim.stride(2), sim.stride(3),
            h, seq_q, seq_k, d, rotate_dim,
            BLOCK_I = 32, BLOCK_J = 32, BLOCK_D = 64, ALLOW_TF32 = allow_tf32
        )
        ctx.save_for_backward(q, k, freqs, bias)
        ctx.rotate_dim, ctx.allow_tf32 = rotate_dim, allow_tf32
        return sim

    @staticmethod
    def backward(ctx, grad_output):
        q, k, freqs, bias = ctx.saved_tensors
        rotate_dim, allow_tf32 = ctx.rotate_dim, ctx.allow_tf32
        b, h, seq_q, d = q.shape
        _, _, seq_k, _ = k.shape
        
        dq, dk = torch.empty_like(q), torch.empty_like(k)
        has_df, has_db = ctx.freqs_requires_grad, ctx.bias_requires_grad
        dfreqs = torch.zeros(freqs.shape, device=freqs.device, dtype=torch.float32) if has_df else None
        dbias = torch.zeros(bias.shape, device=bias.device, dtype=torch.float32) if has_db else None
        
        grad_output = grad_output.contiguous()
        grid_q = (b * h, triton.cdiv(seq_q, 32))
        _bwd_kernel_dqk_df[grid_q](
            dq, dk, dfreqs, grad_output, q, k, freqs, bias,
            dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
            dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
            dfreqs.stride(0) if exists(dfreqs) else 0,
            dfreqs.stride(1) if exists(dfreqs) else 0,
            dfreqs.stride(2) if exists(dfreqs) else 0,
            dfreqs.stride(3) if exists(dfreqs) else 0,
            grad_output.stride(0), grad_output.stride(1), grad_output.stride(2), grad_output.stride(3),
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            freqs.stride(0), freqs.stride(1), freqs.stride(2), freqs.stride(3),
            bias.stride(0), bias.stride(1),
            h, seq_q, seq_k, d, rotate_dim,
            BLOCK_I = 32, BLOCK_J = 32, BLOCK_D = 64, MODE = 0, ALLOW_TF32 = allow_tf32, HAS_DF = has_df
        )
        
        grid_k = (b * h, triton.cdiv(seq_k, 32))
        _bwd_kernel_dqk_df[grid_k](
            dq, dk, dfreqs, grad_output, q, k, freqs, bias,
            dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
            dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
            dfreqs.stride(0) if exists(dfreqs) else 0,
            dfreqs.stride(1) if exists(dfreqs) else 0,
            dfreqs.stride(2) if exists(dfreqs) else 0,
            dfreqs.stride(3) if exists(dfreqs) else 0,
            grad_output.stride(0), grad_output.stride(1), grad_output.stride(2), grad_output.stride(3),
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            freqs.stride(0), freqs.stride(1), freqs.stride(2), freqs.stride(3),
            bias.stride(0), bias.stride(1),
            h, seq_q, seq_k, d, rotate_dim,
            BLOCK_I = 32, BLOCK_J = 32, BLOCK_D = 64, MODE = 1, ALLOW_TF32 = allow_tf32, HAS_DF = has_df
        )
        
        if exists(dbias):
            grid_bias = (b * h, triton.cdiv(seq_q, 32))
            _bwd_kernel_dbias_optimized[grid_bias](
                dbias, grad_output, q, k, freqs, bias,
                grad_output.stride(0), grad_output.stride(1), grad_output.stride(2), grad_output.stride(3),
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                freqs.stride(0), freqs.stride(1), freqs.stride(2), freqs.stride(3),
                bias.stride(0), bias.stride(1),
                b, h, seq_q, seq_k, d, rotate_dim,
                BLOCK_I = 32, BLOCK_J = 32, BLOCK_D = 64, ALLOW_TF32 = allow_tf32
            )

        if exists(dfreqs):
            if len(ctx.orig_freqs_shape) == 2:
                dfreqs_out = dfreqs.sum(dim=(0, 1)).to(q.dtype)
            elif len(ctx.orig_freqs_shape) == 3:
                dfreqs_out = dfreqs.sum(dim=1).to(q.dtype)
            else:
                dfreqs_out = dfreqs.to(q.dtype)
        else: dfreqs_out = None

        dbias_out = dbias.to(q.dtype) if exists(dbias) else None
        return dq, dk, dfreqs_out, dbias_out, None, None

def triton_compute_qk_similarity(q, k, freqs, bias, rotate_dim, allow_tf32 = True):

    assert divisible_by(q.shape[1], k.shape[1])

    groups = q.shape[1] // k.shape[1]
    k = repeat(k, 'b h ... -> b (g h) ...', g = groups)
    bias = repeat(bias, 'h ... -> (g h) ...', g = groups)

    return PoPESimilarityFunction.apply(q, k, freqs, bias, rotate_dim, allow_tf32)
