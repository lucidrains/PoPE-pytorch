import torch
import torch.nn.functional as F
import pytest

from PoPE_pytorch.pope import PoPE
from PoPE_pytorch.attention import compute_attn_similarity, compute_attn_similarity_non_fused

# helpers

def exists(v):
    return v is not None

# tests

@pytest.mark.skipif(not torch.cuda.is_available(), reason = 'CUDA not available')
def test_triton_multi_config():
    torch.set_float32_matmul_precision('highest')
    device = torch.device('cuda')
    
    configs = [
        (1, 8, 128, 128, 64, 0),
        (1, 8, 128, 128, 64, 32),
        (2, 4, 256, 256, 128, 64),
    ]
    
    for batch, heads, seq_q, seq_k, dim, rotate_dim in configs:
        pope = PoPE(rotate_dim, heads = heads).to(device) if rotate_dim > 0 else None
        
        q = torch.randn(batch, heads, seq_q, dim, device = device, requires_grad = True)
        k = torch.randn(batch, heads, seq_k, dim, device = device, requires_grad = True)
        
        pos_embed = pope(max(seq_q, seq_k)) if exists(pope) else (torch.empty(max(seq_q, seq_k), 0, device=device), torch.empty(heads, 0, device=device))

        # Reference
        sim_ref = compute_attn_similarity_non_fused(q, k, pos_embed)
        grad_output = torch.randn_like(sim_ref)
        sim_ref.backward(grad_output, retain_graph = True)
        
        dq_ref, dk_ref = q.grad.clone(), k.grad.clone()
        db_ref = pope.bias.grad.clone() if exists(pope) and exists(pope.bias.grad) else None
        
        q.grad.zero_(), k.grad.zero_()
        if exists(pope) and exists(pope.bias.grad):
            pope.bias.grad.zero_()
        
        # Triton
        sim_triton = compute_attn_similarity(q, k, pos_embed, allow_tf32 = False)
        sim_triton.backward(grad_output)
        
        assert torch.allclose(dq_ref, q.grad, rtol = 1e-3, atol = 1e-4)
        assert torch.allclose(dk_ref, k.grad, rtol = 1e-3, atol = 1e-4)

        if exists(db_ref):
             assert torch.allclose(db_ref, pope.bias.grad, rtol = 1e-3, atol = 1e-4)

@pytest.mark.skipif(not torch.cuda.is_available(), reason = 'CUDA not available')
def test_triton_non_contiguous():
    device = torch.device('cuda')
    batch, heads, seq, dim, rotate_dim = 1, 8, 256, 64, 32
    
    q = torch.randn(batch, heads, seq, dim, device = device, requires_grad = True)
    k = torch.randn(batch, heads, seq, dim, device = device, requires_grad = True)
    
    # Non-contiguous slices
    q_nc = q[:, :, :128, :].detach().requires_grad_(True)
    k_nc = k[:, :, :128, :].detach().requires_grad_(True)
    
    pope = PoPE(rotate_dim, heads = heads).to(device)
    freqs, bias = pope(256)
    
    # Isolation: detach and re-require grad for reference and triton pass
    f_ref, b_ref = freqs[:128].detach().requires_grad_(True), bias.detach().requires_grad_(True)
    f_tri, b_tri = freqs[:128].detach().requires_grad_(True), bias.detach().requires_grad_(True)
    
    sim_ref = compute_attn_similarity_non_fused(q_nc, k_nc, (f_ref, b_ref))
    grad_output = torch.randn_like(sim_ref)
    sim_ref.backward(grad_output)
    dq_ref = q_nc.grad.clone()
    
    q_nc.grad.zero_()
    
    sim_triton = compute_attn_similarity(q_nc, k_nc, (f_tri, b_tri), allow_tf32 = False)
    sim_triton.backward(grad_output)
    
    assert torch.allclose(dq_ref, q_nc.grad, rtol = 1e-3, atol = 1e-4)

@pytest.mark.skipif(not torch.cuda.is_available(), reason = 'CUDA not available')
def test_triton_dfreqs():
    device = torch.device('cuda')
    batch, heads, seq_q, seq_k, dim, rotate_dim = 1, 2, 64, 64, 32, 16
    
    q = torch.randn(batch, heads, seq_q, dim, device = device, requires_grad = True)
    k = torch.randn(batch, heads, seq_k, dim, device = device, requires_grad = True)
    freqs = torch.randn(seq_k, rotate_dim, device = device, requires_grad = True)
    bias = torch.randn(heads, rotate_dim, device = device, requires_grad = True)
    
    f_ref, b_ref = freqs.detach().requires_grad_(True), bias.detach().requires_grad_(True)
    f_tri, b_tri = freqs.detach().requires_grad_(True), bias.detach().requires_grad_(True)

    sim_ref = compute_attn_similarity_non_fused(q, k, (f_ref, b_ref))
    grad_output = torch.randn_like(sim_ref)
    sim_ref.backward(grad_output, retain_graph = True)
    df_ref = f_ref.grad.clone()
    
    # Triton
    sim_triton = compute_attn_similarity(q, k, (f_tri, b_tri), allow_tf32 = False)
    sim_triton.backward(grad_output)
    
    assert torch.allclose(df_ref, f_tri.grad, rtol = 1e-3, atol = 1e-4)
