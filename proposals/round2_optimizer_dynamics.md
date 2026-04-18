# Round 2: Diagnosing the NS Steps x Gamma Interaction in Muon

## Title

**Gradient Spectrum Forensics: Why Newton-Schulz Iteration Count
Reverses Loss-Weight Rankings in Masked Diffusion Training**

## Motivation

Round 1 (training_dynamics.md) predicted that ns_steps would interact
with loss weighting, but the observed pattern defies the simple
"rougher orthogonalization tolerates more scale variance" story:

| Config         | ns=3  | ns=5  | ns=7  |
|----------------|-------|-------|-------|
| gamma=1.5      | 7.48  | 6.39  | 7.57  |
| gamma=5        | 6.46  | 7.26  | 6.41  |
| flat           | 7.54  | 7.60  | pend. |

The ranking *completely reverses*: ns=5 strongly favors gamma=1.5
(6.39 vs 7.26 for gamma=5), while ns=3 and ns=7 both strongly favor
gamma=5 (6.46, 6.41 vs 7.48, 7.57 for gamma=1.5). This is a ~1 nat
swing, far too large to dismiss as noise even at n=1 -- but we must
confirm it is real before building theory.

The 5k-step validation (muon_gamma1.5: 5.52, muon_flat: 5.62,
adam_minsnr: 5.95) confirms Muon+gamma1.5 as the leader at ns=5.
The Adam comparison (adam_gamma1.5: 6.79 ~= adam_minsnr: 6.77) shows
gamma=1.5 does nothing special for Adam, so this is Muon-specific.

## Hypothesis

**The NS quintic polynomial has a resonance at ns=5 that makes it
uniquely sensitive to the gradient singular value distribution, and
different gamma values produce different gradient spectra that either
align or misalign with this resonance.**

Specific mechanism:

1. **Frobenius normalization creates a coupling.** Before NS iteration,
   Muon normalizes the momentum buffer by its Frobenius norm:
   `X = X / X.norm()`. The Frobenius norm equals sqrt(sum of squared
   singular values). Loss weighting (gamma) changes which timesteps
   dominate the gradient, altering the *shape* of the singular value
   distribution of the accumulated gradient matrix. A low gamma (1.5)
   concentrates gradient energy on intermediate timesteps, producing a
   gradient matrix with *fewer dominant singular values* (lower
   effective rank). A high gamma (5+) or flat weighting spreads energy
   more evenly, producing a *higher effective rank* gradient matrix.

2. **NS iteration count controls convergence basin.** The quintic
   polynomial (a=3.4445, b=-4.7750, c=2.0315) converges for initial
   singular values in (0, 1] after Frobenius normalization. With 5
   steps, the polynomial pushes all singular values toward 1, but its
   convergence rate depends on the initial distribution. Per the
   convergence analysis (arXiv 2601.19156), the residual error after
   k steps is doubly exponential in k, but the *constant factor*
   depends on the ratio sigma_max/sigma_min of the normalized matrix.

3. **The parity effect.** At ns=3 and ns=7 (both odd, flanking 5),
   the polynomial may converge to a different fixed point than at ns=5
   for certain spectral distributions. More precisely: the quintic
   coefficients were chosen to "maximize slope at zero" (Keller
   Jordan), meaning they aggressively inflate small singular values.
   At exactly 5 iterations, this inflation is well-calibrated for
   moderate condition numbers. At 3 iterations, under-inflation leaves
   small singular values partially suppressed -- which happens to
   benefit the higher-effective-rank gradient matrices produced by
   gamma=5. At 7 iterations, over-inflation distorts the spectrum,
   again favoring a different gradient shape.

**Alternative hypothesis (noise):** These are all n=1 runs at 1000
steps. A ~1 nat swing is large but not impossible from bad random
seeds interacting with the small quokka model. The first experiment
below tests this directly.

## Method

### Phase 1: Confirm the Interaction is Real (12 runs, ~30 min)

**Replicate the 2x3 grid with 3 seeds each.**

```
Config: quokka, 1000 steps, bs=32, --no_time_cond, cosine LR
Optimizer: muon (lr=0.02, momentum=0.95)

Grid: {ns_steps=3, ns_steps=5} x {gamma=1.5, gamma=5}
Seeds: 3 per cell (seed=42, seed=137, seed=2024)
Total: 2 x 2 x 3 = 12 runs
```

We drop ns=7 from the replication to save budget. If ns=3 and ns=5
show the reversal with 3 seeds each, it is real. Specifically:

- **Confirm if:** Mean(ns=5, gamma=1.5) < Mean(ns=5, gamma=5) AND
  Mean(ns=3, gamma=5) < Mean(ns=3, gamma=1.5), with non-overlapping
  95% CIs (or at least consistent sign across all 3 seeds).
- **Reject if:** The ranking is inconsistent across seeds within the
  same cell, indicating n=1 noise.

**Implementation:** Add `--seed` argument to train.py (torch manual
seed + torch.cuda manual seed). Currently missing from CLI but trivial:

```python
# In train.py parse_args():
p.add_argument("--seed", type=int, default=42)

# In train(), before model creation:
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
```

### Phase 2: Gradient Spectrum Forensics (4 instrumented runs, ~15 min)

**Log the singular value distribution of Muon's momentum buffer at
each step, for the 4 corners of the confirmed grid.**

Instrument `MuonAdamW._muon_step()` to periodically compute and log
the SVD of the momentum buffer before and after NS orthogonalization:

```python
# In _muon_step(), every 50 steps, for the largest parameter:
if self._log_step % 50 == 0 and p.numel() == self._largest_param_size:
    with torch.no_grad():
        U_pre = update.clone()
        svs_pre = torch.linalg.svdvals(U_pre.float())
        
        U_post = zeropower_via_newtonschulz5(update, steps=ns_steps)
        svs_post = torch.linalg.svdvals(U_post.float())
        
        # Log to wandb or stdout
        self._sv_log.append({
            "step": self._log_step,
            "pre_condition_number": (svs_pre[0] / svs_pre[-1]).item(),
            "pre_effective_rank": (svs_pre.sum()**2 / (svs_pre**2).sum()).item(),
            "post_max_sv": svs_post[0].item(),
            "post_min_sv": svs_post[-1].item(),
            "post_sv_std": svs_post.std().item(),
            "ns_residual": (1.0 - svs_post).abs().max().item(),
        })
```

Key metrics to extract:
- **Pre-NS condition number** (sigma_max / sigma_min of momentum buffer)
- **Pre-NS effective rank** (sum(sigma)^2 / sum(sigma^2), Roy & Bhatt)
- **Post-NS residual** (max |sigma_i - 1| after orthogonalization)
- **NS convergence quality** (how close to true UV^T)

**Prediction:** gamma=1.5 produces lower effective rank (fewer dominant
modes) than gamma=5. At ns=5, this lower-rank structure is well-served
by the quintic polynomial. At ns=3, the polynomial under-converges on
the dominant modes, and the higher-rank structure from gamma=5 is more
forgiving.

### Phase 3: NS Convergence Curve (6 instrumented runs, ~20 min)

**Measure the orthogonalization quality at each NS iteration (1-7)
for different gamma values, to find the actual convergence curve.**

Modify `zeropower_via_newtonschulz5` to return intermediate states:

```python
def zeropower_via_newtonschulz5_instrumented(G, max_steps=7):
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    
    residuals = []
    for i in range(max_steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
        # Measure: how close are singular values to 1?
        svs = torch.linalg.svdvals(X.float())
        residuals.append({
            "ns_step": i + 1,
            "max_residual": (1.0 - svs).abs().max().item(),
            "mean_residual": (1.0 - svs).abs().mean().item(),
            "min_sv": svs[-1].item(),
            "max_sv": svs[0].item(),
        })
    
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X, residuals
```

Run this on saved gradient snapshots from Phase 2 (no additional
training needed, just replay the SVD analysis on stored tensors).

```
Grid: gamma={1.5, 5.0, flat} x ns_measured={1..7}
Per saved gradient at steps {100, 500, 900}
```

**Prediction:** The convergence curve will show that at ns=5, the
gamma=1.5 gradient has residual < 0.01 (well-converged) while the
gamma=5 gradient has residual ~0.05-0.1 (partially converged, with
some singular values still far from 1). At ns=3, the gamma=5 gradient
will be the one that happens to land in a "good enough" approximation
while gamma=1.5 is still in an unstable intermediate state.

### Phase 4: Targeted Fixes (8 runs, ~20 min)

Based on Phase 2-3 findings, test fixes that should break the
ns x gamma coupling:

**4a. Spectral-norm normalization instead of Frobenius.**
Replace `X = X / X.norm()` with `X = X / svdvals(X)[0]` in the NS
initialization. This changes the initial singular value distribution
fed to the polynomial, potentially making it less sensitive to gamma.
(Motivated by Chen & Chow 2014 who show spectral normalization
improves NS stability.)

```python
# In zeropower_via_newtonschulz5:
# Current: X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
# Proposed: normalize by largest singular value instead
s_max = torch.linalg.svdvals(X.float())[..., :1].to(X.dtype)
X = X / (s_max.unsqueeze(-1) + 1e-7)
```

**4b. Adaptive NS steps.** Use a fixed target residual instead of
fixed step count. Run NS until max|sigma_i - 1| < threshold, up to
a maximum of 10 steps.

```python
def zeropower_adaptive(G, tol=0.01, max_steps=10):
    ...
    for i in range(max_steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
        # Early stop if converged (check every 2 steps to amortize SVD cost)
        if (i + 1) % 2 == 0 and i >= 3:
            svs = torch.linalg.svdvals(X.float())
            if (1.0 - svs).abs().max() < tol:
                break
    ...
```

**4c. NorMuon-style row normalization.** After NS orthogonalization,
normalize each row of the update to have unit norm. This decouples
the per-neuron update scale from the global spectral structure.
(Motivated by NorMuon, arXiv 2510.05491.)

```python
# After NS orthogonalization:
update = zeropower_via_newtonschulz5(update, steps=ns_steps)
# NorMuon-style: normalize per neuron (row)
update = update / (update.norm(dim=-1, keepdim=True) + 1e-7)
update *= max(1, update.size(-2) / update.size(-1)) ** 0.5
```

```
Grid for 4a-4c: {spectral_norm, adaptive_ns, normuon_row} x {gamma=1.5, gamma=5}
Plus 2 controls: {standard_ns5_gamma1.5, standard_ns5_gamma5}
Total: 8 runs
```

**Prediction:** If the Frobenius normalization coupling is the
mechanism, 4a (spectral norm) should eliminate the ns x gamma
interaction: both gamma values should perform similarly at ns=5.
If the issue is convergence quality, 4b (adaptive NS) should work.
If neither helps but 4c does, the problem is in per-neuron scale
variance, not global spectral structure.

## Expected Outcome

### Best Case
Phase 1 confirms the interaction is real (non-overlapping CIs). Phase
2 reveals that gamma=1.5 produces gradients with condition number ~5-10
while gamma=5 produces condition number ~2-3. Phase 3 shows the NS
polynomial converges non-monotonically for the high-condition-number
case, explaining why ns=5 is a sweet spot. Phase 4a (spectral norm
init) eliminates the interaction and gives gamma=1.5 performance at all
NS step counts, yielding a Muon variant that is robust to loss
weighting. Final result: val_loss ~6.35-6.40 reliably across NS steps,
with a clear mechanistic explanation and a 1-line fix to Muon.

### Worst Case
Phase 1 shows the interaction is noise (inconsistent across seeds).
Still valuable: confirms gamma=1.5 + ns=5 as the best config with
tighter error bars (mean +/- std from 3 seeds), and the gradient
spectrum data from Phase 2 advances understanding of Muon dynamics
on diffusion objectives.

### Middle Case
Interaction is real but the mechanism is not Frobenius normalization.
Phase 4a-4c all fail to eliminate it. This points toward a deeper
interaction in the momentum buffer dynamics (Nesterov momentum +
time-varying gradient scale from loss weighting), requiring a more
invasive investigation (e.g., replacing Nesterov with plain momentum,
or using EMA of gradient singular values for normalization).

## Risk / Cost

### Budget
| Phase | Runs | Time/run | Total time | Code changes |
|-------|------|----------|------------|--------------|
| 1     | 12   | 2.5 min  | 30 min     | +seed arg    |
| 2     | 4    | 4 min*   | 16 min     | SVD logging  |
| 3     | 0    | offline  | 5 min CPU  | instrumented NS |
| 4     | 8    | 2.5 min  | 20 min     | NS variants  |
| **Total** | **24** | | **~75 min** | ~1 hr impl |

*Phase 2 runs are slower due to periodic SVD computation.

At ~500 runs/day capacity, this is well within a single session.
Phases are sequential (each depends on prior results), so the full
experiment takes ~3 hours including analysis time.

### Risks

1. **SVD computation overhead.** Computing `torch.linalg.svdvals` on
   the momentum buffer (384x768 for quokka) every 50 steps adds ~10ms
   per logged step. At 1000 steps total, this is 20 SVD calls x 10ms
   = 0.2s overhead. Negligible.

2. **Spectral norm init (Phase 4a) requires SVD per NS call.** This
   is expensive: one SVD per parameter per step. For the quokka model
   with ~20 Muon parameters, this adds ~200ms/step. Mitigation: only
   compute the leading singular value via power iteration (1-2 matmuls,
   ~1ms), which suffices for normalization.

3. **Phase 1 may show the interaction is noise.** In this case,
   Phases 2-4 become exploratory rather than confirmatory. We still
   run Phase 2 (gradient spectrum logging) because the data is
   independently valuable for understanding Muon on diffusion.

4. **The quokka model (31.5M) may not generalize.** The ns x gamma
   interaction could be an artifact of the small model size where
   gradient matrices are small enough that NS convergence is marginal.
   Mitigation: if Phase 1 confirms the interaction, replicate the 4
   corners at the "small" config (84M) to check scale robustness.

## Literature Support

### Newton-Schulz Convergence and the Spectral Sensitivity Problem

The NS iteration's convergence rate depends critically on the initial
singular value distribution of the input matrix. Chen & Chow (2014)
show that Frobenius normalization can "shrink small singular values
excessively," causing slow or non-monotonic convergence, and propose
spectral-norm scaling as a fix
([A Stable Scaling of Newton-Schulz](https://faculty.cc.gatech.edu/~echow/pubs/chen-chow-2014.pdf)).
The CANS work (arXiv 2506.10935) formalizes this: the NS polynomial
must converge on the interval [sigma_min/norm, sigma_max/norm], and
Frobenius normalization sets norm = sqrt(sum sigma_i^2), which
compresses the interval non-uniformly depending on the spectral shape
([Accelerating Newton-Schulz via Chebyshev Polynomials](https://arxiv.org/abs/2506.10935)).

### Muon's Quintic Polynomial and the "Cursed Coefficients"

Keller Jordan's quintic (a=3.4445, b=-4.7750, c=2.0315) was chosen
to "maximize slope at zero," meaning it prioritizes inflating tiny
singular values rapidly. The convergence analysis (arXiv 2601.19156)
proves that with these coefficients, 5 steps suffice for transformers,
but the constant factor depends on the condition number of the
Frobenius-normalized gradient. Critically, NS5 and NS32 yield similar
results in practice -- but this was demonstrated on autoregressive LM
training (Adam-like gradient spectrum), not diffusion training where
loss weighting drastically alters the gradient spectrum
([Convergence of Muon with Newton-Schulz](https://arxiv.org/abs/2601.19156)).

### Min-SNR and Gradient Conflict

Hang et al. (ICCV 2023) show that diffusion training suffers from
gradient conflict between timesteps: optimal gradients for t~0 point
in opposite directions to those for t~1. Min-SNR-gamma reduces this
conflict by clamping the 1/t weight. Lower gamma clamps more
aggressively, producing a gradient matrix where conflicting components
are suppressed -- i.e., a lower effective rank
([Min-SNR Weighting Strategy](https://arxiv.org/abs/2303.09556)).
This directly connects gamma to the gradient spectral shape that feeds
into Muon's NS iteration.

### Mousse and Curvature-Aware Preconditioning

Mousse (arXiv 2603.09697) identifies that standard Muon "treats all
eigen-directions as geometrically equivalent," ignoring curvature
disparities. It pre-conditions the gradient with Shampoo-style
Kronecker factors before NS orthogonalization, effectively "sphering"
the gradient spectrum. This is the same problem we are diagnosing: the
gradient's spectral shape (controlled by gamma) interacts with NS
convergence (controlled by ns_steps). Mousse's 12% speedup validates
that this interaction matters
([Mousse: Curvature-Aware Preconditioning](https://arxiv.org/abs/2603.09697)).

### Newton-Muon and Adaptive NS

Newton-Muon (arXiv 2604.01472) reinterprets Muon as an implicit
Newton method and shows that the right preconditioning (input second
moment) matters. Their finding that "NS5 and NS32 are close" holds
for standard LM training -- our result that NS3/NS5/NS7 dramatically
differ for diffusion training suggests the gradient spectrum under
diffusion loss weighting falls outside the "safe" convergence basin
of the quintic polynomial
([The Newton-Muon Optimizer](https://arxiv.org/abs/2604.01472)).

### NorMuon and Per-Neuron Normalization

NorMuon (arXiv 2510.05491) adds neuron-wise adaptive scaling after
NS orthogonalization, achieving "low condition numbers AND uniform
neuron norms." This directly addresses one possible mechanism for the
ns x gamma interaction: if different gamma values produce per-neuron
gradient scale variance, NS orthogonalization preserves this variance
(it operates on the global spectrum, not per-row), and NorMuon's
row normalization would fix it
([NorMuon](https://arxiv.org/abs/2510.05491)).

### AdaMuon and RMS Alignment

AdaMuon (arXiv 2507.11005) aligns Muon's update scale with Adam's
via RMS matching and weight decay. Their key insight: Muon's lack of
second-moment estimation makes it sensitive to gradient magnitude
shifts. Loss weighting is exactly such a magnitude shift, applied
per-sample rather than per-parameter. AdaMuon's approach suggests
that matching the update RMS across timestep-weighted gradients could
stabilize Muon under varying gamma
([AdaMuon: Adaptive Muon Optimizer](https://arxiv.org/abs/2507.11005)).

## Implementation Priority

1. **Phase 1 (seed replication):** Add `--seed` to train.py (2 min),
   write a `ns_gamma_replicate` mode for autoresearch.py (~15 min).
   Run immediately. This is the gating experiment.

2. **Phase 2 (spectrum logging):** Add SVD instrumentation to
   `MuonAdamW._muon_step` (~20 min). Run the 4 instrumented runs.
   Analyze with a quick matplotlib script.

3. **Phase 3 (NS convergence curve):** Write the instrumented NS
   function (~10 min). Run offline on saved gradients from Phase 2.
   Plot convergence curves for each gamma.

4. **Phase 4 (targeted fixes):** Implement the 3 NS variants
   (~30 min). Run the 8-run grid. This is the payoff experiment.

## Decision Criteria

After Phase 1:
- If interaction is NOT confirmed (rankings inconsistent across seeds):
  STOP. Report gamma=1.5 + ns=5 as best config with error bars.
  Proceed to scaling experiments (proposals/scaling_convergence.md).
- If interaction IS confirmed: proceed to Phase 2-4.

After Phase 4:
- If spectral-norm init (4a) eliminates the interaction: adopt it as
  the default Muon configuration. Write up the finding.
- If adaptive NS (4b) works: adopt it, noting the throughput cost.
- If NorMuon row-norm (4c) works: adopt it, noting the connection
  to the NorMuon paper.
- If none work: the mechanism is in the momentum dynamics, not the NS
  step. Propose a Phase 5 investigating momentum buffer conditioning
  (EMA of gradient second moments, a la AdaMuon).
