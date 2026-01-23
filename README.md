<img src="./pope.png" width="400px"></img>

## PoPE-pytorch (wip)

Efficient implementation (and explorations) into [polar coordinate positional embedding (PoPE)](https://arxiv.org/abs/2509.10534) - from [Gopalakrishnan](https://agopal42.github.io/) et al. under Schmidhuber

## Install

```shell
$ pip install PoPE-pytorch
```

## Usage

```python
import torch
from PoPE_pytorch import PoPE

# define pope

pope = PoPE(64, heads = 8)

# pass in sequence length

pos_embed = pope(1024)

# queries and keys in attention

q = torch.randn(1, 8, 1024, 64)
k = torch.randn(1, 8, 1024, 64)

# training

rotated_q, rotated_k = pope.apply_pope_to_qk(pos_embed, q, k)

# inference

rotated_q, rotated_k = pope.apply_pope_to_qk(pos_embed, q[..., -1:, :], k)
```

### Fused Attention Similarity

```python
import torch
from PoPE_pytorch import PoPE, compute_attn_similarity

# define pope

pope = PoPE(dim = 64, heads = 8).cuda()

# get rotations

pos_emb = pope(1024)

# queries and keys

q = torch.randn(1, 8, 1024, 64).cuda()
k = torch.randn(1, 8, 1024, 64).cuda()

# fused attention similarity, avoiding expanding 64 to 128

sim = compute_attn_similarity(q, k, pos_emb) # (1, 8, 1024, 1024)

attn = sim.softmax(dim = -1) # the usual in attention..
```

### Fused Flash Attention

```python
import torch
from PoPE_pytorch import PoPE, flash_attn_with_pope

# pope

pope = PoPE(dim = 32, heads = 8).cuda()

# queries, keys, values for attention

q = torch.randn(1, 8, 1024, 64).cuda()
k = torch.randn(1, 8, 1024, 64).cuda()
v = torch.randn(1, 8, 1024, 64).cuda()

pope_emb = pope(1024)

out = flash_attn_with_pope(q, k, v, pope = pope_emb, causal = True)

assert out.shape == (1, 8, 1024, 64)
```

## Citations

```bibtex
@misc{gopalakrishnan2025decouplingwhatwherepolar,
    title   = {Decoupling the "What" and "Where" With Polar Coordinate Positional Embeddings}, 
    author  = {Anand Gopalakrishnan and Robert Csordás and Jürgen Schmidhuber and Michael C. Mozer},
    year    = {2025},
    eprint  = {2509.10534},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/2509.10534}, 
}
```
