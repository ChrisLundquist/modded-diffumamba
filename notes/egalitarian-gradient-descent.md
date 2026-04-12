# Egalitarian Gradient Descent (EGD)

## Paper
- **Title:** Egalitarian Gradient Descent: A Simple Approach to Accelerated Grokking
- **Authors:** Ali Saheb Pasand, Elvis Dohmatob (Mila)
- **Published:** ICLR 2026
- **arXiv:** https://arxiv.org/abs/2510.04930

## Core Idea

EGD modifies the gradient matrix to equalize learning speed across all principal directions:

```
G̃ = (G G^T)^{-1/2} G
```

This preserves the singular vectors (directions) of G but sets all singular values to 1.
Every direction in parameter space evolves at exactly the same rate, eliminating the
asymmetric convergence that causes grokking plateaus.

EGD is a "whitened" variant of natural gradient descent:
- Natural gradient: `(GG^T)^{-1} G`
- EGD: `(GG^T)^{-1/2} G` (square root version)

## Key Results
- Eliminates grokking plateau on modular addition, sparse parity, toy classification
- Generalization happens from the start of training instead of after a long plateau
- Tested on small algorithmic tasks, not large-scale language models

## Connection to Muon Optimizer

Muon's Newton-Schulz iteration computes an approximation of UV^T from the gradient's SVD.
This is closely related to EGD:

- **EGD:** `G̃ = U V^T` (exact, via `(GG^T)^{-1/2} G`)
- **Muon/NS5:** `G̃ ≈ U S' V^T` where S' ≈ Uniform(0.5, 1.5) (approximate, via quintic iteration)

Both equalize gradient singular values. Muon is an approximate, computationally cheaper
version of EGD that doesn't require explicit SVD computation. The Newton-Schulz iteration
converges to the polar factor UV^T, which is exactly what EGD computes.

## Connection to Our flat_weight Finding

Our flat ELBO weighting (weight=1 across all noise timesteps, instead of 1/t) operates
at a different level but shares the same principle:

| Method | What it equalizes | Level |
|--------|-------------------|-------|
| EGD | Gradient singular values | Optimizer (gradient transform) |
| Muon/NS5 | Gradient singular values (approx) | Optimizer (gradient transform) |
| flat_weight | Timestep loss contributions | Loss function |
| Min-SNR | Timestep gradient magnitudes (clamped) | Loss function |

The fact that flat_weight helped Adam more than Muon (+3.4% vs +2.3%) is consistent with
this interpretation: Muon already partially equalizes gradient directions, so adding
loss-level equalization has diminishing returns. Adam has no such built-in equalization,
so it benefits more from the loss-level fix.

## Related Work

- **Min-SNR Weighting** (Hang et al., ICCV 2023, arXiv 2303.09556): Frames diffusion
  training as multi-task learning with gradient conflicts across timesteps. Uses
  `w_t = min{SNR(t), gamma}` to clamp weights and reduce gradient conflict.

- **GradNorm** (Chen et al., ICML 2018): Normalizes gradient magnitudes across tasks
  in multi-task learning.

- **"Training Optimal Large Diffusion Language Models"** (arXiv 2510.03280): Compares
  ELBO loss (1/t weighting) vs MaskGIT loss (no reweighting) for masked diffusion LMs.
  MaskGIT converges faster initially but ELBO achieves better final performance.

## Implications for This Project

1. **Muon ≈ approximate EGD** — this is a known connection but worth keeping in mind
   when interpreting optimizer comparisons.

2. **The "right" approach might be Min-SNR** rather than binary flat vs 1/t — it's a
   principled middle ground that clips extreme weights while preserving useful signal.

3. **Applying EGD directly** would mean replacing Muon's NS5 with exact SVD-based
   orthogonalization. Too expensive for large matrices, but could be tested on small configs.

4. **The deeper question**: should we equalize at the loss level (flat_weight, Min-SNR),
   the optimizer level (Muon, EGD), or both? Our experiments suggest loss-level
   equalization is more impactful, at least at this scale.
