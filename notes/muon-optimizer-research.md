# Muon Optimizer: Research Notes

## Core Algorithm

Muon (MomentUm Orthogonalized by Newton-Schulz) — optimizer for 2D weight matrices.

1. Compute SGD-momentum (Nesterov) on gradients
2. Post-process with Newton-Schulz (NS) iteration to orthogonalize
3. NS approximates UV^T from SVD — nearest semi-orthogonal matrix
4. Only for 2D params; embeddings/biases/layernorm use AdamW

### Newton-Schulz Iteration (quintic, 5 steps)

```python
def newtonschulz5(G, steps=5, eps=1e-7):
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X
```

### Usage (always hybrid)

```python
# Muon for 2D hidden weights, AdamW for everything else
hidden_weights = [p for p in model.body.parameters() if p.ndim >= 2]
other_params = [p for p in model.parameters() if p.ndim < 2]
non_hidden = [*model.head.parameters(), *model.embed.parameters()]

param_groups = [
    dict(params=hidden_weights, use_muon=True, lr=0.02, weight_decay=0.01),
    dict(params=other_params + non_hidden, use_muon=False,
         lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01),
]
```

### Critical Notes

- Q, K, V should be optimized **separately** (not as a single QKV matrix)
- Default hyperparams (momentum=0.95, nesterov=True, ns_steps=5) generally work
- LR has built-in muP scaling — transfers across model widths
- Computational overhead: <1% of total training FLOPs
- Memory: ~33% savings vs AdamW (no second moment for 2D params)

## Availability

- PyTorch native: `torch.optim.Muon` (v2.9+)
- Standalone: [github.com/KellerJordan/Muon](https://github.com/KellerJordan/Muon)
- Flash-Muon (Triton kernels): [github.com/nil0x9/flash-muon](https://github.com/nil0x9/flash-muon)
- NVIDIA NeMo: [Emerging-Optimizers](https://github.com/NVIDIA-NeMo/Emerging-Optimizers)

## Scaling Evidence

| System | Scale | Result |
|--------|-------|--------|
| nanogpt-speedrun | 124M, GPT-2 | 1.35x faster, single largest contributor |
| Moonlight (Moonshot AI) | 3B/16B MoE, 5.7T tokens | 2x compute efficiency vs AdamW |
| Kimi K2 | 1T params, 15.5T tokens | Zero loss spikes (MuonClip variant) |
| Essential AI | 4B | Retains efficiency at large batch sizes |

## CRITICAL: Muon + Diffusion Models

**Standard image diffusion: FAILS.** "Speedrunning ImageNet Diffusion" (arxiv 2512.12386):
Muon plateaued early, FID 48.70 (catastrophic). Adam beats Muon by 400K iterations.

**Masked diffusion LMs: UNTESTED but promising.** The key difference:
- Image diffusion: noise prediction loss in continuous space
- Masked diffusion LM: cross-entropy loss over discrete tokens (= weighted MLM)
- Muon excels at language modeling with cross-entropy
- **Our hypothesis: Muon should work for MDLM because the loss surface resembles LM, not diffusion**

## Key Variants

| Variant | Key Idea | Reference |
|---------|----------|-----------|
| MuonClip | + QK-Clip for stability | Kimi K2 paper |
| AdaMuon | + element-wise second momentum | OpenReview |
| NorMuon | Neuron-wise normalization | OpenReview |
| Newton-Muon | Right preconditioning from input moments | arxiv 2604.01472 |
| Flash-Muon | Custom Triton kernels for NS | github |
| MuonAll | Extends to all params for finetuning | arxiv 2511.06086 |

## Theoretical Foundation

Jeremy Bernstein's "modular duality" derivation:
1. Metrize layers with RMS-to-RMS operator norm
2. Solve steepest descent under spectral norm constraint
3. Solution = UV^T from gradient SVD
4. Approximate with Newton-Schulz iterations

Muon = steepest descent under the spectral norm. muP emerges automatically.
Blog: [jeremybernste.in/writing/deriving-muon](https://jeremybernste.in/writing/deriving-muon)
