# Muon-Optimal Loss Weighting and Newton-Schulz Tuning for Masked Diffusion LMs

## Hypothesis

Muon's Newton-Schulz orthogonalization projects gradient updates onto the
spectral-norm unit ball, equalizing singular values. This makes Muon
*insensitive to gradient direction quality* but *highly sensitive to
gradient scale variance across samples in a batch*. The optimizer's
momentum buffer accumulates raw gradients before orthogonalization, so
timestep-dependent loss weights that create 100-1000x scale ratios
(ELBO's 1/t) corrupt the momentum statistics, while flat weighting
(scale ratio = 1) lets Muon shine.

We hypothesize three things:

1. **Soft-clamped weighting beats both flat and min-SNR for Muon.**
   Flat weighting (w=1) discards all information about which timesteps
   matter more. A "Muon-optimal" weighting should be *nearly flat* but
   with a mild bias toward intermediate timesteps (where most learning
   signal lives). Concretely: `w(t) = clamp(1/t, max=gamma)` with
   gamma in [1.5, 3.0] -- tighter than standard min-SNR (gamma=5)
   but not completely flat.

2. **Scheduled weighting (flat early -> mild min-SNR late) captures the
   best of both worlds.** Early training needs to learn all timesteps
   roughly equally (flat favors this). Late training benefits from
   focusing on harder intermediate timesteps. A linear anneal from
   gamma=1 (flat) to gamma=3 over the course of training should
   outperform static flat.

3. **Newton-Schulz iteration count interacts with loss weight scale
   variance.** Fewer NS steps (3) produce a rougher orthogonalization
   that is *more tolerant* of scale variance (because the projection
   is less precise). More steps (7) produce tighter orthogonalization
   that amplifies any scale-variance-induced momentum corruption.
   Therefore: ns_steps=3 might rescue Muon+minSNR(gamma=5), while
   ns_steps=7 should widen the gap between flat and ELBO.

## Method

### Experiment 1: Muon-Optimal Gamma Sweep (7 runs)

Find the best static loss weighting for Muon by sweeping gamma finely
between flat and min-SNR.

```
Config: quokka (36M), 1000 steps, bs=8, --no_time_cond, cosine LR
Optimizer: muon (lr=0.02, momentum=0.95, ns_steps=5)
Metric: val_loss (ELBO-weighted, for cross-config comparability)

Sweep grid:
  --loss_weight flat                     # gamma=inf (baseline: 6.54)
  --loss_weight minsnr --minsnr_gamma 1.5
  --loss_weight minsnr --minsnr_gamma 2.0
  --loss_weight minsnr --minsnr_gamma 3.0
  --loss_weight minsnr --minsnr_gamma 5.0  # baseline: 6.97
  --loss_weight minsnr --minsnr_gamma 10.0
  --loss_weight minsnr --minsnr_gamma 20.0
```

Note: gamma=infinity is equivalent to flat. As gamma decreases, the
weighting becomes more ELBO-like. We expect the optimum to be at
gamma=1.5-3.0, strictly between flat (6.54) and minsnr-5 (6.97).

**Implementation:** No code changes needed. All configs already exist in
train.py (`--loss_weight minsnr --minsnr_gamma X`). The autoresearch.py
runner can handle this as a new mode.

### Experiment 2: Scheduled Loss Weighting (4 runs)

Anneal gamma from flat to a target value over training. Requires a small
code change to `model.py::compute_loss()` to accept a `training_progress`
argument.

```
Config: quokka, 1000 steps, bs=8, --no_time_cond, cosine LR, muon
Sweep:
  static_flat                            # gamma=inf throughout (control)
  anneal_flat_to_gamma3                  # gamma: inf -> 3 linearly
  anneal_flat_to_gamma5                  # gamma: inf -> 5 linearly
  anneal_gamma3_to_flat                  # gamma: 3 -> inf (reverse, control)
```

**Implementation changes required:**

In `train.py`, pass `step/max_steps` to `model.compute_loss()`:
```python
# In training loop, after forward:
loss, metrics = model.compute_loss(x_0, training_progress=step/args.max_steps)
```

In `model.py::compute_loss()`, add scheduled gamma:
```python
def compute_loss(self, x_0, training_progress=None):
    ...
    if lw == "minsnr":
        weight = dsigma / torch.expm1(sigma)
        gamma = self.config.minsnr_gamma
        if self.config.loss_weight_schedule == "anneal" and training_progress is not None:
            # Anneal from flat (large gamma) to target gamma
            gamma = 1000.0 * (1 - training_progress) + gamma * training_progress
        weight = torch.clamp(weight, max=gamma)
```

Add new config fields:
```python
# In DiffuMamba3Config:
loss_weight_schedule: str = "static"  # "static" or "anneal"
```

Add CLI arg:
```python
p.add_argument("--loss_weight_schedule", type=str, default="static",
               choices=["static", "anneal"])
```

### Experiment 3: Newton-Schulz Steps x Loss Weight (6 runs)

Test whether NS iteration count interacts with loss weight tolerance.

```
Config: quokka, 1000 steps, bs=8, --no_time_cond, cosine LR, muon
Grid: {ns_steps=3, ns_steps=5, ns_steps=7} x {flat, minsnr_gamma5}
```

**Implementation changes required:**

Add CLI arg to train.py:
```python
p.add_argument("--ns_steps", type=int, default=5,
               help="Newton-Schulz iterations for Muon (default 5)")
```

In `build_optimizer()`, pass `ns_steps` to the Muon param group:
```python
dict(params=muon_params, use_muon=True,
     lr=args.muon_lr, momentum=args.muon_momentum,
     weight_decay=args.muon_wd, ns_steps=args.ns_steps),
```

This is already plumbed through -- `MuonAdamW.__init__` reads `ns_steps`
from the group dict and `_muon_step` passes it to
`zeropower_via_newtonschulz5`. We just need the CLI arg.

### Experiment 4 (stretch): MuonClip for Non-Flat Weights (4 runs)

Test whether MuonClip's QK-clip stabilization mechanism allows Muon to
tolerate ELBO-like weightings that normally cause failure.

```
Config: quokka, 1000 steps, bs=8, --no_time_cond, cosine LR, muon
Grid: {muonclip_off, muonclip_tau100} x {flat, minsnr_gamma5}
```

**Implementation:** This requires adding the QK-clip post-step to
`MuonAdamW._muon_step()`. After the parameter update, for any weight
matrices that are query/key projections, compute the max QK score and
rescale if above threshold tau. However, since our backbone is Mamba-3
(SSM, not attention), QK-clip does not directly apply. **Defer this
experiment** unless we add attention layers later.

**Alternative for SSM:** Implement a simpler "gradient-scale clip" that
clips the momentum buffer's per-parameter norm before NS orthogonalization:
```python
# In _muon_step, before NS:
if clip_momentum:
    buf_norm = buf.norm()
    if buf_norm > clip_threshold:
        buf.mul_(clip_threshold / buf_norm)
```

## Expected Outcome

### Experiment 1 (Gamma Sweep)
- **Confirm:** Optimal gamma for Muon is in [1.5, 3.0], achieving
  val_loss 6.50-6.53 (beating flat's 6.54).
- **Reject if:** Flat (gamma=inf) remains best, meaning ANY timestep
  reweighting hurts Muon. This would strengthen the "Muon needs perfectly
  uniform gradients" interpretation.

### Experiment 2 (Scheduled Weighting)
- **Confirm:** anneal_flat_to_gamma3 beats static_flat by 0.02-0.05 nats
  (val_loss ~6.50-6.52), as early flat training builds a stable
  foundation and late min-SNR focuses on harder timesteps.
- **Reject if:** Static flat still wins, suggesting Muon's momentum
  buffer cannot tolerate *any* non-stationarity in gradient scale
  distribution even when introduced gradually.
- **Reverse control** (gamma3->flat) should perform worst, confirming
  directionality matters.

### Experiment 3 (NS Steps x Loss Weight)
- **Confirm:** ns_steps=3 + minsnr_gamma5 closes the gap with flat
  (e.g., 6.60 vs 6.97 at ns_steps=5), while ns_steps=7 + flat
  matches or slightly beats ns_steps=5 + flat.
- **Reject if:** NS steps have no interaction with loss weight. This
  would mean the momentum corruption hypothesis is wrong, and the
  problem lies elsewhere (e.g., in the NS coefficients themselves).

### Experiment 4 (Momentum Clip)
- **Confirm:** Momentum-norm clipping rescues Muon+minSNR to val_loss
  ~6.60-6.70 (vs 6.97 unclipped).
- **Reject if:** Clipping hurts flat weighting too, suggesting it
  destroys useful gradient information.

## Risk/Cost

### GPU Hours
- Experiment 1: 7 runs x ~8 min each (quokka, 1000 steps on 9070 XT) = ~56 min
- Experiment 2: 4 runs x ~8 min = ~32 min (plus ~30 min implementation)
- Experiment 3: 6 runs x ~8 min = ~48 min
- Experiment 4: 4 runs x ~8 min = ~32 min (plus ~1 hr implementation)
- **Total: ~3 hours GPU time + ~1.5 hours implementation**

### What Could Go Wrong
1. **Experiment 1 may show flat is already optimal.** If the loss
   landscape is such that ANY timestep bias corrupts Muon's momentum,
   the gamma sweep will show a monotonic curve with flat best. Still
   useful: confirms the "Muon needs flat" hypothesis definitively.

2. **Scheduled weighting may be too noisy at 1000 steps.** The anneal
   happens over only 1000 steps, so the transition may be too abrupt.
   Mitigation: also test at 2000 steps if budget allows.

3. **NS steps=3 may produce poor orthogonalization regardless.** The
   Newton-Schulz iteration converges quintically, so 3 steps may not
   converge at all for ill-conditioned gradients. The CANS/Chebyshev
   literature suggests the quintic coefficients (a=3.4445, b=-4.7750,
   c=2.0315) are optimized for 5+ steps. Mitigation: also test
   ns_steps=4 if ns_steps=3 diverges.

4. **Confound: gradient clipping.** The current `--grad_clip 1.0` already
   partially addresses gradient scale variance. We should run all
   experiments with the same grad_clip (1.0) to avoid confounding, and
   optionally repeat the best configs with grad_clip=0 to isolate the
   effect.

## Literature Support

### Muon's Sensitivity to Gradient Scale (Theoretical Basis)

Muon orthogonalizes the gradient update via Newton-Schulz iteration,
projecting onto the spectral-norm unit ball. Ma et al. (2025) show
that this acts as a *spectral preconditioner* -- Muon's convergence
is governed by the gradient Lipschitz constant under spectral norm,
which can be much smaller than the Euclidean one
([Preconditioning Benefits of Spectral Orthogonalization in Muon](https://arxiv.org/abs/2601.13474)).
Shen et al. (2025) further prove Muon optimizes under spectral norm
constraints, implying that per-sample gradient scale variance (from
loss weighting) directly impacts the quality of the momentum buffer
that feeds into NS orthogonalization
([MUON Optimizes Under Spectral Norm Constraints](https://opt-ml.org/papers/2025/paper137.pdf)).

### MuonClip: Stabilizing Muon at Scale

The Kimi K2 team (Moonshot AI, 2025) developed MuonClip to address
attention logit explosion during Muon training at 1T+ parameter scale.
QK-Clip rescales query/key weight matrices when max attention scores
exceed a threshold tau, applied after each Muon step. This achieved
zero loss spikes over 15.5T tokens of pretraining
([Kimi K2: Open Agentic Intelligence](https://arxiv.org/abs/2507.20534),
[Deep-dive into MuonClip](https://fireworks.ai/blog/muonclip)).
While QK-Clip targets attention layers (not SSMs), the principle --
post-step norm control to prevent runaway activations -- is directly
applicable to our momentum clipping variant.

### Newton-Schulz Iteration Count Optimization

Recent work on accelerating the NS iteration shows that the default
5-step quintic iteration can be improved. Cesista (2025) optimized
the polynomial coefficients for fewer iterations
([Squeezing 1-2% Efficiency from Muon](https://leloykun.github.io/ponder/muon-opt-coeffs/)).
The CANS approach (arXiv 2506.10935) uses Chebyshev-type polynomials
to achieve better orthogonalization in fewer matmuls. Critically,
optimal learning rate and momentum must be co-tuned with NS step count
-- fewer iterations require smaller step sizes, confirming that
ns_steps is a non-trivial hyperparameter
([Accelerating Newton-Schulz Iteration via Chebyshev-type Polynomials](https://arxiv.org/abs/2506.10935)).

### Muon Failure on Image Diffusion

"Speedrunning ImageNet Diffusion" (arXiv 2512.12386) shows Muon
achieves catastrophically bad FID (48.70) on image diffusion with
standard MSE loss, which implicitly uses SNR-dependent weighting.
Our finding that Muon works with flat weighting on MDLM but fails
with ELBO weighting provides a mechanistic explanation: image
diffusion's MSE loss creates the same gradient scale variance as ELBO
([Speedrunning ImageNet Diffusion](https://arxiv.org/abs/2512.12386)).

### Loss Weighting in Masked Diffusion LMs

The min-SNR strategy (Hang et al., ICCV 2023/2025) demonstrates 3.4x
faster convergence by clamping extreme timestep weights, originally for
continuous diffusion
([Improved Noise Schedule for Diffusion Training](https://openreview.net/forum?id=j3U6CJLhqw)).
Frequency-informed masking (arXiv 2509.05056) shows that curriculum-like
strategies can improve MDLM training by biasing toward rare tokens
([Masked Diffusion LMs with Frequency-Informed Training](https://arxiv.org/abs/2509.05056)).
The connection between loss weighting and noise scheduling is further
explored in "Loss Functions in Diffusion Models: A Comparative Study"
(arXiv 2507.01516), which shows all common objectives are weighted
integrals of ELBOs.

### Muon Variants and Convergence Theory

The Newton-Muon optimizer (arXiv 2604.01472) and Mousse (arXiv
2603.09697) extend Muon with curvature-aware preconditioning, while
MuonBP (arXiv 2510.16981) reduces cost via block-periodic
orthogonalization. These confirm that the NS step is the critical
component and that modifications to it (count, coefficients, frequency)
have measurable impact on training dynamics.

## Implementation Priority

1. **Experiment 1 (Gamma Sweep):** Zero code changes, run immediately.
   Add a `muon_gamma_sweep` mode to autoresearch.py.
2. **Experiment 3 (NS Steps):** One CLI arg addition, trivial. Run
   alongside Experiment 1.
3. **Experiment 2 (Scheduled Weighting):** Small code change to
   model.py and train.py. Run after analyzing Experiments 1 & 3.
4. **Experiment 4 (Momentum Clip):** Moderate code change. Run only
   if Experiments 1-3 suggest momentum corruption is the mechanism.
