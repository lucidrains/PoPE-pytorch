import torch
import torch.nn.functional as F
from PoPE_pytorch import PoPE
from PoPE_pytorch.attention import pope_attention

def test_researcher_pope():
    print("--- Researcher Persona: Exploring PoPE ---")
    
    # 1. Generation
    print("\n[Stage 1: Generation]")
    dim = 64
    heads = 4
    seq_len = 128
    
    pope_module = PoPE(dim = dim, heads = heads)
    pope_embeddings = pope_module(seq_len)
    
    print(f"PoPE Freqs shape: {pope_embeddings.freqs.shape}") # Should be (128, dim)
    print(f"PoPE Bias shape: {pope_embeddings.bias.shape}")   # Should be (heads, dim)

    # 2. Application (Training)
    print("\n[Stage 2: Application - Training]")
    q = torch.randn(1, heads, seq_len, dim, requires_grad = True)
    k = torch.randn(1, heads, seq_len, dim, requires_grad = True)
    v = torch.randn(1, heads, seq_len, dim, requires_grad = True)
    
    # Found this function in pope.py
    rotated_q, rotated_k = pope_module.apply_pope_to_qk(pope_embeddings, q, k)
    
    print(f"Rotated Q shape: {rotated_q.shape}")
    print(f"Rotated K shape: {rotated_k.shape}")
    
    # Confusion: The user asked to apply to values 'v' as well.
    # But apply_pope_to_qk only takes q and k.
    # Let's try to see if there is another way or if it's missing.
    try:
        # rotated_v = apply_pope_to_v(pope_embeddings, v) # This doesn't exist
        print("CONFUSION: No function found for applying PoPE to values (v).")
    except NameError:
        pass

    # 3. Application (Inference)
    print("\n[Stage 3: Application - Inference]")
    # In inference, we might only have one new query token but all keys
    q_new = torch.randn(1, heads, 1, dim)
    
    # Getting embeddings for the last position (offset = seq_len - 1)
    pope_new = pope_module(1, offset = seq_len - 1)
    
    rotated_q_new, _ = pope_module.apply_pope_to_qk(pope_new, q_new, k)
    print(f"Rotated Q (inference) shape: {rotated_q_new.shape}")

    # 4. Fused Attention
    print("\n[Stage 4: Fused Attention]")
    # Using the high-level API
    if torch.cuda.is_available():
        q_cuda = q.cuda()
        k_cuda = k.cuda()
        v_cuda = v.cuda()
        pope_module.cuda()
        pope_emb_cuda = pope_module(seq_len)
        
        # This function seems to handle the fusion
        out = pope_attention(q_cuda, k_cuda, v_cuda, pope = pope_emb_cuda, causal = True)
        print(f"Fused attention output shape: {out.shape}")
    else:
        print("CUDA not available, skipping fused check.")

if __name__ == "__main__":
    test_researcher_pope()
