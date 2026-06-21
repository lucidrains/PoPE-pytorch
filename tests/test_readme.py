import torch
import pytest
from PoPE_pytorch import PoPE, AxialPoPE, compute_attn_similarity, flash_attn_with_pope

def test_readme_usage():
    # define pope
    pope = PoPE(64, heads = 8)

    # pass in sequence length
    pos_emb = pope(1024)

    # queries and keys in attention
    q = torch.randn(1, 8, 1024, 64)
    k = torch.randn(1, 8, 1024, 64)

    # training
    rotated_q, rotated_k = pope.apply_pope_to_qk(pos_emb, q, k)
    assert rotated_q.shape == (1, 8, 1024, 128)
    assert rotated_k.shape == (1, 8, 1024, 128)

    # inference
    rotated_q, rotated_k = pope.apply_pope_to_qk(pos_emb, q[..., -1:, :], k)
    assert rotated_q.shape == (1, 8, 1, 128)
    assert rotated_k.shape == (1, 8, 1024, 128)

def test_readme_axial_pope():
    # axial pope for images (e.g. 32x32)
    # split 64 dim into 32 (x) and 32 (y)
    pope = AxialPoPE(
        dim = 64,
        heads = 8,
        axial_dims = (32, 32)
    )

    pos_emb = pope((32, 32)) # (1024, 64) frequencies
    assert pos_emb.freqs.shape == (1024, 64)

    # for video (e.g. 8 frames, 16x16 frames)
    # split 96 dim into 32 (t), 32 (x), 32 (y)
    pope_video = AxialPoPE(
        dim = 96,
        heads = 8,
        axial_dims = (32, 32, 32)
    )

    pos_emb_video = pope_video((8, 16, 16)) # (2048, 96) frequencies
    assert pos_emb_video.freqs.shape == (2048, 96)

    # queries and keys
    q = torch.randn(1, 8, 2048, 96)
    k = torch.randn(1, 8, 2048, 96)

    rotated_q, rotated_k = AxialPoPE.apply_pope_to_qk(pos_emb_video, q, k)
    assert rotated_q.shape == (1, 8, 2048, 192)

@pytest.mark.skipif(not torch.cuda.is_available(), reason = 'CUDA must be available')
def test_readme_fused_similarity():
    # define pope
    pope = PoPE(dim = 64, heads = 8).cuda()

    # get rotations
    pos_emb = pope(1024)

    # queries and keys
    q = torch.randn(1, 8, 1024, 64).cuda()
    k = torch.randn(1, 8, 1024, 64).cuda()

    # fused attention similarity, avoiding expanding 64 to 128
    sim = compute_attn_similarity(q, k, pos_emb) # (1, 8, 1024, 1024)
    assert sim.shape == (1, 8, 1024, 1024)

    attn = sim.softmax(dim = -1) # the usual in attention..

@pytest.mark.skipif(not torch.cuda.is_available(), reason = 'CUDA must be available')
def test_readme_fused_flash_attn():
    # pope
    pope = PoPE(dim = 32, heads = 8).cuda()

    # queries, keys, values for attention
    q = torch.randn(2, 8, 1024, 64).cuda()
    k = torch.randn(2, 8, 1024, 64).cuda()
    v = torch.randn(2, 8, 1024, 64).cuda()

    pos_emb = pope(1024)

    mask = torch.ones((2, 1024)).bool().cuda()

    out = flash_attn_with_pope(q, k, v, pos_emb = pos_emb, causal = True, mask = mask)

    assert out.shape == (2, 8, 1024, 64)

@pytest.mark.skipif(not torch.cuda.is_available(), reason = 'CUDA must be available')
def test_readme_axial_pope_with_latents():
    # 16x16 image patches + 4 latents
    num_patches = 256
    num_latents = 4
    seq_len = num_patches + num_latents

    pope = AxialPoPE(dim = 64, heads = 8, axial_dims = (32, 32)).cuda()

    # generate positions for the 16x16 grid
    pos_emb = pope((16, 16))

    # apply positions to first 256 tokens, leave 4 latents unrotated
    pos_indices = torch.arange(num_patches, device = 'cuda')

    q = torch.randn(1, 8, seq_len, 64).cuda()
    k = torch.randn(1, 8, seq_len, 64).cuda()
    v = torch.randn(1, 8, seq_len, 64).cuda()

    # pass indices to handle unrotated latents
    out = flash_attn_with_pope(
        q, k, v,
        pos_emb = pos_emb,
        pope_pos_emb_indices = pos_indices
    )

    assert out.shape == (1, 8, seq_len, 64)
