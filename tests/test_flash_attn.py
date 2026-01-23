import torch
import pytest
from PoPE_pytorch import PoPE
from PoPE_pytorch.attention import flash_attn_with_pope, compute_attn_similarity
from einops import rearrange

def exists(v):
    return v is not None

@pytest.mark.skipif(not torch.cuda.is_available(), reason = 'CUDA must be available')
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize('causal', [True, False])
@pytest.mark.parametrize('seq_len_q', [1, 256])
def test_fused_vs_manual(dtype, causal, seq_len_q):
    device = 'cuda'
    
    batch, heads, dim = 2, 4, 64
    seq_len_k = 256
    rotate_dim = 32
    
    torch.manual_seed(42)
    
    q = torch.randn(batch, seq_len_q, heads, dim, device = device, dtype = dtype, requires_grad = True)
    k = torch.randn(batch, seq_len_k, heads, dim, device = device, dtype = dtype, requires_grad = True)
    v = torch.randn(batch, seq_len_k, heads, dim, device = device, dtype = dtype, requires_grad = True)
    
    pope_module = PoPE(rotate_dim, heads = heads).to(device)
    pope = pope_module(seq_len_k)
    
    do = torch.randn(batch, seq_len_q, heads, dim, device = device, dtype = dtype)
    
    # 1. Manual Path
    qc1, kc1, vc1 = q.clone().detach().requires_grad_(True), k.clone().detach().requires_grad_(True), v.clone().detach().requires_grad_(True)
    pc1 = [t.clone().detach().requires_grad_(True) for t in pope]
    
    out_manual = flash_attn_with_pope(qc1, kc1, vc1, pope = pc1, causal = causal, fused = False, head_dimension_at_first = False)
    out_manual.backward(do)
    
    # 2. Fused Path
    qc2, kc2, vc2 = q.clone().detach().requires_grad_(True), k.clone().detach().requires_grad_(True), v.clone().detach().requires_grad_(True)
    pc2 = [t.clone().detach().requires_grad_(True) for t in pope]
    
    out_fused = flash_attn_with_pope(qc2, kc2, vc2, pope = pc2, causal = causal, fused = True, head_dimension_at_first = False)
    out_fused.backward(do)
    
    # check parity
    def get_max_diff(a, b):
        return (a - b).abs().max().item()

    diff_out = get_max_diff(out_manual, out_fused)
    diff_dq = get_max_diff(qc1.grad, qc2.grad)
    diff_dk = get_max_diff(kc1.grad, kc2.grad)
    diff_dv = get_max_diff(vc1.grad, vc2.grad)
    
    print(f"\n[{dtype}, causal={causal}] Out Diff: {diff_out:.6f}, dQ Diff: {diff_dq:.6f}, dK Diff: {diff_dk:.6f}, dV Diff: {diff_dv:.6f}")

    atol = 5e-2 if dtype != torch.float32 else 2e-2
    
    assert torch.allclose(out_manual, out_fused, atol = atol), f"Out diff too large: {diff_out}"
    assert torch.allclose(qc1.grad, qc2.grad, atol = atol), f"dQ diff too large: {diff_dq}"
    assert torch.allclose(kc1.grad, kc2.grad, atol = atol), f"dK diff too large: {diff_dk}"
    assert torch.allclose(vc1.grad, vc2.grad, atol = atol), f"dV diff too large: {diff_dv}"
    
    if exists(pc1[0].grad) and exists(pc2[0].grad):
        diff_df = get_max_diff(pc1[0].grad, pc2[0].grad)
        assert torch.allclose(pc1[0].grad, pc2[0].grad, atol = atol), f"dfreqs diff too large: {diff_df}"

    if exists(pc1[1].grad) and exists(pc2[1].grad):
        diff_dpb = get_max_diff(pc1[1].grad, pc2[1].grad)
        assert torch.allclose(pc1[1].grad, pc2[1].grad, atol = atol), f"dpope_bias diff too large: {diff_dpb}"

@pytest.mark.skipif(not torch.cuda.is_available(), reason = 'CUDA must be available')
@pytest.mark.parametrize('seq_len_q', [1, 1024])
def test_compute_attn_similarity(seq_len_q):
    device = 'cuda'
    batch, heads, seq_len_k, dim = 1, 8, 1024, 64
    rotate_dim = 32

    q = torch.randn(batch, seq_len_q, heads, dim, device = device)
    k = torch.randn(batch, seq_len_k, heads, dim, device = device)
    v = torch.randn(batch, seq_len_k, heads, dim, device = device)

    pope_module = PoPE(rotate_dim, heads = heads).to(device)
    pope = pope_module(seq_len_k)

    # 1. Compute similarity then aggregate
    # We pass head_dimension_at_first = False because our q, k are (b, n, h, d)
    sim = compute_attn_similarity(q, k, pope, head_dimension_at_first = False)
    
    softmax_scale = dim ** -0.5
    attn = (sim * softmax_scale).softmax(dim = -1)
    
    # head-first for the einsum (b h n d)
    v_head_first = rearrange(v, 'b n h d -> b h n d')
    out_similarity = torch.einsum('b h i j, b h j d -> b h i d', attn, v_head_first)
    out_similarity = rearrange(out_similarity, 'b h n d -> b n h d')

    # 2. Use complete flash_attn_with_pope
    out_attention = flash_attn_with_pope(q, k, v, pope = pope, head_dimension_at_first = False)

    # 3. Validate parity
    diff = (out_similarity - out_attention).abs().max().item()
    print(f"\n[seq_len_q={seq_len_q}] Similarity vs Attention Diff: {diff:.6f}")
    assert diff < 5e-4
