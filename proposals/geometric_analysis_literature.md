# Weight-Spectral / Geometric Analysis of Optimizers: Literature + Plan

## 1. Established work (what's already published)

**Martin & Mahoney, HT-SR line (2018–2021).** Empirical spectral density (ESD)
of W^T W fit to truncated power law; exponent alpha in [2,6] correlates with
generalization across ~100 CV/NLP models. Tool: `weightwatcher`. Works without
data. Predicts test accuracy from weights alone. (arXiv:1901.08276;
JMLR 22:20-410; weightwatcher.ai.)

**Pennington & Bahri (ICML 2017), Pennington & Worah (NeurIPS 2017).** RMT for
NN Hessian and Jacobian spectra; Marchenko-Pastur + heavy-tail universality
classes. Foundational but about Hessian/Jacobian, not W.

**Ghorbani et al. (ICML 2019), "Hessian eigenvalue density".** Lanczos-based
bulk+outlier characterization under different optimizers. Relevant template
for plotting ESDs across optimizers.

**Low-rank simplicity bias (Huh et al. 2021; Galanti et al. 2022).** Effective
rank of W decreases over training regardless of optimizer; depth-dependent.
This is the baseline conclusion you'd need to beat/contextualize.

**Muon-specific spectral claims.**
- Jordan's blog (kellerjordan.github.io/posts/muon): shows update spectra,
  NOT cumulative weight spectra. Update is close to UV^T by construction.
- Bernstein, "Deriving Muon": frames Muon as spectral-norm steepest descent
  (dual norm = nuclear).
- "Muon is Scalable for LLM Training" (Moonshot, arXiv:2502.16982): reports
  Muon produces flatter singular-value distribution and higher SVD entropy
  than AdamW on >90% of 2D weights in trained models. **This is the
  closest-to-our-study published result.**
- "Muon Under Spectral Norm Constraints" (arXiv:2506.15054): proves Muon
  with decoupled WD implicitly bounds spectral norm of W (not just update).
- NorMuon (arXiv:2510.05491), Mousse (arXiv:2603.09697), Variance-Adaptive
  Muon / Muon-VS (arXiv:2601.14603), "What Really Matters in Matrix-Whitening
  Optimizers?" (arXiv:2510.25000): all focus on loss curves and wall-clock;
  none publish post-training weight ESDs.
- Conda (arXiv:2509.24218), LDAdam (arXiv:2410.16103): document Adam's
  low-rank / high-condition-number update pathology.

## 2. Answers to specific questions

**Q1 (SOTA):** Moonshot's Muon-at-scale paper is the only peer-ish work
reporting per-matrix SVD-entropy comparison Muon vs AdamW. Weightwatcher/HT-SR
is the dominant framework for weight-only analysis. Jordan does NOT measure
cumulative weight orthogonality, only update.

**Q2 (is orthogonality gap right?):** No, not directly. Muon orthogonalizes
the update. W_t = W_0 + sum_s eta_s O_s where O_s ~ U_s V_s^T. The sum of
orthogonal matrices is NOT orthogonal; in the limit it's closer to a random
matrix with equalized singular values (isotropic ESD), not W^T W = I.
Report **stable rank / SVD entropy / sigma-histogram flatness**, not
||W^T W - I||. Weight decay + spectral-norm bound (2506.15054) means sigma_max
is controlled; that's the right orthogonality-adjacent claim.

**Q3 (Mamba-specific):** in_proj is a 5-way fused projection
(x, z, B, C, dt-gate) — its spectrum is an average over heterogeneous
sub-blocks, which muddles interpretation. **Slice in_proj into its
sub-projections before SVD.** out_proj is a clean d_inner -> d_model linear
and is the cleanest target. A matrix is initialized structured (S4D-Real /
HiPPO-style) and is usually NOT in Muon routing — confirm in your code
before including. Conv1d is small and not 2D-Muon-routed.

**Q4 (Muon-VS vs Muon):** VS normalizes per-element momentum variance
**before** NS. NS is scale-invariant in the sense that orthogonalization
projects to the Stiefel manifold regardless of input scale, but VS changes
the *direction* of the pre-NS momentum (different elements contribute
differently), which changes which singular vectors the update emphasizes.
Expectation: similar final isotropy, possibly different per-layer magnitude
distribution. Your ||W_t - W_0|| per layer should differ; final ESD shape
likely similar. This is a testable, publishable distinction.

**Q5 (Mousse without optimizer state):** You can still measure final weight
ESD, stable rank, alpha-hat. What you cannot reconstruct: the Kronecker
preconditioner itself. You can say "Mousse's Kronecker whitening produces
final weight spectrum X relative to Muon's Y"; you cannot say why. That is
still a valid empirical contribution — nobody has published it.

**Q6 (pitfalls):**
- Effective rank is fine but secondary; **stable rank ||W||_F^2/||W||_2^2**
  and **PL alpha** are more cited. Report both.
- bf16 -> fp32 cast is safe for SVD of well-conditioned matrices, but SVD
  of near-degenerate sigma is unstable at any precision (PyTorch docs warn).
  Use `torch.linalg.svdvals` in fp32; report condition numbers; drop layers
  with cond > 1e6 or flag them.
- Reviewer attacks you should pre-empt: (a) "different LR schedules produce
  different spectra trivially" -> control for final loss, or report spectra
  conditioned on matched val_loss checkpoints; (b) "n=1 per optimizer"
  -> you have seed 42 only; get at least one more seed if claiming effect;
  (c) "embedding != hidden weight" -> Muon doesn't route embeddings, so
  embedding-vs-block comparison is intra-model but optimizer-confounded
  with layer type (not clean); (d) HT-SR alpha is noisy on matrices with
  d<500 — several of your layers are borderline.
- Checkpoint selection: one step-10k snapshot can be noisy. The 111M
  trajectory (10k..50k) is your strongest asset.

**Q7 (minimum viable):** Claim = "Muon-family optimizers produce
quantifiably flatter singular-value spectra than Adam at matched training
budget, and Muon-VS / Mousse variants do not meaningfully change final
spectrum shape relative to Muon despite loss differences." Evidence =
SVD entropy + stable rank + PL alpha across all 2D Muon-routed matrices
for 4 optimizers at 10k, plus trajectory (10k..50k) showing when flatness
emerges. That's a workshop paper / blog post, not NeurIPS, but defensible.

**Q8 (top-3 insight-per-hour):**
1. **Singular value distribution (svdvals in fp32) + stable rank + PL alpha
   per 2D matrix, 4 optimizers, 31M checkpoint.** 1 afternoon. Cleanest
   deliverable. Replicates Moonshot's claim on a different model family
   (Mamba vs Transformer) and a different task (MDLM vs AR).
2. **Trajectory ESD on 111M, steps 10k..50k.** When does Muon's isotropy
   appear? Single-seed is fine here — you're showing a temporal curve,
   not a group contrast.
3. **Per-layer ||W_t - W_0||_F / ||W_0||_F and sigma_max(W) trajectory.**
   Tests the Muon-spectral-norm-bound claim (2506.15054) empirically on
   your SSM. Cheap once (1) is done.

Skip: orthogonality gap ||W^T W - I||, Hessian analysis (needs data+grads),
NTK. All high cost / low marginal insight given constraints.

## Key refs
- Martin & Mahoney, JMLR 22:20-410 (HT-SR).
- Moonshot, arXiv:2502.16982 (Muon scalable; SVD-entropy comparison).
- arXiv:2506.15054 (Muon spectral-norm bound).
- Bernstein, "Deriving Muon" (spectral-norm steepest descent).
- Mousse arXiv:2603.09697; NorMuon arXiv:2510.05491;
  Muon-VS arXiv:2601.14603.
- Huh et al. 2021 (low-rank simplicity bias baseline).
- `weightwatcher` pip package for PL alpha fits.
