# Agent Prompt: Weight-Geometric Study of Optimizer Behavior

Paste the block below into an Opus agent invocation (or a fresh agent).
Give it tool access to Bash, Read, Write, Grep, Glob (all tools).

---

## Task

You are analyzing saved checkpoints from the DiffuMamba3 project to produce a
weight-geometric study of optimizer behavior. All weights are trained; we did
NOT save optimizer state or gradients. Produce a new analysis script, run it,
and write up findings.

## Background

- Repo: `/home/durandal/claude/diffusion-lm-autoresearch/`
- Python env: `.venv/bin/python3`
- Model arch: `model.py` (DiffuMamba3, bidirectional Mamba-3 + MLP blocks)
- Training: `train.py` (MuonAdamW with variants base/vs/mousse, AdamW baseline)
- 63 checkpoints in `checkpoints/` at bf16

## What was already researched

Read FIRST before coding:
- `proposals/geometric_analysis_literature.md` — literature survey with 17 refs,
  including the critical finding that orthogonality gap is the WRONG metric
  for cumulative Muon weights (sums of orthogonal matrices aren't orthogonal)

Key takeaways already established:
1. Muon orthogonalizes the update, not the cumulative weight. Use stable rank,
   SVD entropy, and PL-alpha from Martin & Mahoney instead
2. Moonshot Muon-at-scale (arXiv 2502.16982) is the only published comparison
   of post-training per-matrix SVD spectra for Muon vs AdamW
3. Mamba wrinkle: `in_proj` is a fused 5-way projection (x, z, B, C, dt) —
   SLICE BEFORE SVD. `out_proj` is the cleanest single-purpose target
4. PL-alpha is noisy for matrices with d<500. Report with wide error bars
5. Safe: cast bf16→fp32 before `torch.linalg.svdvals`. Use `svdvals` not
   full `svd` (10-100× faster, we only need singular values)

## Data available

Three analyses are feasible without retraining:

### 1. Temporal trajectory (strongest)
5-point trajectory on the SAME model (10L×640d, 111.7M, seed 42):
- `checkpoints/10L640d_50k_step10000.pt`
- `checkpoints/10L640d_50k_step20000.pt`
- `checkpoints/10L640d_50k_step30000.pt`
- `checkpoints/10L640d_50k_step40000.pt`
- `checkpoints/10L640d_50k_step50000.pt`
- Plus: `checkpoints/10L640d_10k.pt` (separate 10k-step run, different LR schedule)

### 2. Optimizer comparison (paired, same seed)
At step 10k on quokka (31.5M), seed 42:
- `checkpoints/opt10k_muon_s42.pt` — base Muon
- `checkpoints/opt10k_muon_vs_s42.pt` — Muon-VS
- `checkpoints/opt10k_mousse_s42.pt` — Mousse
- `checkpoints/opt10k_adam_s42.pt` — pure AdamW
- Also seeds 137, 2024 for variance bars

### 3. Seed replicates
3 seeds × 2 configs × 3 variants for within-condition variance estimates:
- `checkpoints/final10k_new_best_s{42,137,2024}.pt` (Muon-VS + out_proj, FineWeb-Edu)
- `checkpoints/final10k_old_best_s{42,137,2024}.pt` (Muon-VS + out_proj, FineWeb)
- Many more in `checkpoints/` — use `ls checkpoints/` to see

## Metrics to compute (literature-backed)

For each 2D weight matrix W in a checkpoint:

1. **Stable rank**: `‖W‖_F² / ‖W‖_2²` — robust proxy for effective rank,
   dimensionless, bounded [1, min(m,n)]
2. **SVD entropy**: `H(σ²/‖σ‖²)` — low entropy = concentrated spectrum (Adam),
   high entropy = flat spectrum (Muon). Moonshot's primary metric.
3. **PL-alpha** (optional, d ≥ 500 only): power-law exponent of tail of σ²,
   Martin-Mahoney's heavy-tailed-self-regularization criterion
4. **Max / min / condition number**: σ_1, σ_min, σ_1/σ_min
5. **Per-layer ‖W_t − W_0‖_F / ‖W_0‖_F**: how much each layer moved over
   training. Note: we don't have step-0 checkpoints, so use step10k as the
   reference "early" state for the temporal-trajectory study.

## Mamba-specific slicing (critical)

The `in_proj` weight for Mamba3 blocks is a fused projection. You MUST slice
it before SVD analysis. Read `model.py` or `ssm.py` to find the actual slice
boundaries. For PureSSM it's:
- `x` (d_inner), `z` (d_inner), `B` (nheads × d_state), `C` (nheads × d_state), `dt` (nheads)
- Total output dim: `d_inner*2 + nheads*(2*d_state + 1)`

Recommend focusing analysis on:
- `mamba_fwd.out_proj.weight` and `mamba_bwd.out_proj.weight` (clean targets)
- MLP weights `mlp.w1`, `mlp.w2`, `mlp.w3` (for SwiGLU) — these are full-rank
  targets with no internal structure, ideal for optimizer comparison
- Skip `in_proj` unless you implement slicing

## Deliverables

1. **New file**: `/home/durandal/claude/diffusion-lm-autoresearch/analyze_weight_geometry.py`
   - Loads a specified checkpoint, computes all metrics per 2D matrix
   - Handles bf16 → fp32 cast for SVD
   - Emits per-matrix and per-layer summaries to JSON
   - Should be re-runnable on any checkpoint

2. **Run three comparisons**:
   (a) Trajectory on 10L×640d (5 points) — which layers change most; do spectra
       flatten or concentrate as training progresses
   (b) Optimizer paired comparison at 10k — do Muon/Muon-VS/Mousse produce
       measurably different spectra vs Adam? Include the multi-seed variance.
   (c) Sanity: embedding table (Adam-trained) vs block weights (Muon-trained)
       — compare directly in the SAME model. Confounded by layer type but
       worth plotting.

3. **Plots** (matplotlib, save to `results/geometry/`):
   - Stable-rank vs training-step for each layer type
   - SVD entropy histograms by optimizer
   - σ-spectrum CDFs overlaid for Muon vs Adam on the same weight matrix
   - Per-layer change magnitude heatmap (layer × step)

4. **Writeup**: `results/geometry/REPORT.md`
   - 1-page executive summary
   - Figures inline with 1-paragraph interpretation each
   - Explicit list of claims we CAN defend vs claims we CAN'T (reviewer's eye)
   - Compare our findings to the Moonshot paper numbers where possible

## Constraints

- Total budget: 4 hours wall clock
- bf16 CAST to fp32 before SVD, use `torch.linalg.svdvals`
- Do NOT use `torch.linalg.svd` (returns full U/V, 10-100× slower)
- Skip matrices with d < 500 for PL-alpha (too noisy)
- Embedding table is 50304 × 384 for quokka; fp32 SVD fits in ~100MB RAM
- For MLP matrices at 10L640d, the 640×1280 matrices are the interesting ones

## Watchouts

- Don't claim orthogonality of cumulative weights from Muon — that's the
  trap documented in the literature survey
- Multi-seed variance: Muon-VS improved loss by only 0.04 nats but its
  seed-level weight variance may dwarf that. Report variance.
- Layer-type confound: comparing Adam-embedding vs Muon-block tells you
  as much about layer type as optimizer. Flag this explicitly.
- Cast to fp32 BEFORE any arithmetic that could underflow — singular
  values near 1e-4 in bf16 round to zero

## Success criteria

You're done when:
1. The script runs end-to-end on all 3 comparisons
2. Plots show a visible signal (e.g. Muon entropy > Adam entropy) OR
   show a negative result with error bars that make the nullity clear
3. REPORT.md enumerates what we can and cannot defend, with specific refs
4. Everything is committed to the repo with a clear commit message

Read the literature-survey file first. Ask for clarification if any of the
above is ambiguous before starting.
