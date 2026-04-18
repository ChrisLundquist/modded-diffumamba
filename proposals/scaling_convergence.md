# Scaling & Convergence: Does Muon's Advantage Persist?

## Title

Muon-flat vs Adam-minsnr convergence trajectory at 2x-5x training budget and 2.4x model scale

## Hypothesis

**Primary:** Muon-flat's advantage over Adam-minsnr will persist and likely *grow* at longer training (5000 steps) and larger model scale (84M "small" config), with no crossover.

**Rationale:** Three lines of evidence support this:

1. **"Muon is Scalable" (Liu et al., Feb 2025):** Scaling law experiments show Muon achieves ~2x compute efficiency vs AdamW, and this holds at scales up to 16B params / 5.7T tokens. The advantage is *persistent* -- no crossover even far past Chinchilla-optimal budget.

2. **"Practical Efficiency of Muon" (Essential AI, May 2025):** Across 100M-4B parameter models, "Muon strictly lower bounds AdamW on [loss] curves, even to the end of training far beyond the Chinchilla-optimal budget (i.e., there is no crossover)." Muon requires 10-15% fewer tokens for the same loss.

3. **The flat-weighting interaction is structurally clean:** Muon's Newton-Schulz orthogonalization normalizes gradient directions. Flat loss weighting produces uniform gradient scale across timesteps. This combination has no internal conflict, unlike ELBO/minsnr which create scale variance that fights Muon's momentum. This structural alignment should remain stable across scales.

**Counter-hypothesis (what could go wrong):** The above evidence is all from autoregressive LMs with cross-entropy loss. Masked diffusion uses the *same* cross-entropy but with a masking forward process and time-integrated weighting. At longer training, masked diffusion's loss landscape might develop different curvature properties that favor Adam's per-parameter adaptivity. Additionally, at 84M scale with only 1B training tokens (~12 tokens/param), we are well below Chinchilla-optimal (~20 tokens/param), which could change dynamics.

## Method

### Phase 1: Long Training (quokka 31.5M, 5000 steps)

Run the two champions head-to-head at 5x the 1000-step budget where they were originally compared.

**Experiment 1a: muon_flat_5k**
```bash
python train.py --config quokka --batch_size 8 --max_steps 5000 \
    --val_every 250 --log_every 50 --warmup_steps 200 \
    --no_time_cond --lr_schedule cosine \
    --optimizer muon --loss_weight flat --muon_lr 0.02 --adam_lr 3e-4
```

**Experiment 1b: adam_minsnr_5k**
```bash
python train.py --config quokka --batch_size 8 --max_steps 5000 \
    --val_every 250 --log_every 50 --warmup_steps 200 \
    --no_time_cond --lr_schedule cosine \
    --optimizer adam --loss_weight minsnr --adam_lr 3e-4
```

**Experiment 1c: adam_flat_5k** (control -- is Adam better with flat weighting at scale?)
```bash
python train.py --config quokka --batch_size 8 --max_steps 5000 \
    --val_every 250 --log_every 50 --warmup_steps 200 \
    --no_time_cond --lr_schedule cosine \
    --optimizer adam --loss_weight flat --adam_lr 3e-4
```

**What to track:**
- val_loss at every 250 steps (ELBO-weighted for comparability, as train.py already does)
- Loss trajectory shape: is the gap constant, widening, or narrowing?
- Wall-clock time per step (Muon has NS overhead, check it doesn't dominate)
- Tokens/second throughput

**Tokens per experiment:** 5000 steps x 8 batch x 1024 seq = 40.96M tokens
**Estimated GPU time:** ~25 min each on 9070 XT with Mamba3 Triton, ~45 min with PureSSM

### Phase 2: Model Scale (small 84M, 3000 steps)

Scale up to the "small" config (84M params) and run at a moderate step count. 3000 steps at batch 8 = 24.6M tokens (~0.3 tokens/param, firmly in the underfitting regime, but sufficient to measure optimizer trajectory differences).

**Experiment 2a: muon_flat_small_3k**
```bash
python train.py --config small --batch_size 8 --max_steps 3000 \
    --val_every 250 --log_every 50 --warmup_steps 200 \
    --no_time_cond --lr_schedule cosine \
    --optimizer muon --loss_weight flat --muon_lr 0.02 --adam_lr 3e-4 \
    --muon_wd 0.01 --adam_wd 0.01
```

**Experiment 2b: adam_minsnr_small_3k**
```bash
python train.py --config small --batch_size 8 --max_steps 3000 \
    --val_every 250 --log_every 50 --warmup_steps 200 \
    --no_time_cond --lr_schedule cosine \
    --optimizer adam --loss_weight minsnr --adam_lr 3e-4 \
    --adam_wd 0.01
```

**Experiment 2c: muon_flat_small_3k_wd** (weight decay ablation, per "Muon is Scalable" finding that WD is critical at scale)
```bash
python train.py --config small --batch_size 8 --max_steps 3000 \
    --val_every 250 --log_every 50 --warmup_steps 200 \
    --no_time_cond --lr_schedule cosine \
    --optimizer muon --loss_weight flat --muon_lr 0.02 --adam_lr 3e-4 \
    --muon_wd 0.0 --adam_wd 0.0
```

**What to track:**
- Same metrics as Phase 1
- VRAM peak usage (small config is tighter at 84M -- verify it fits in 16GB with grad checkpointing)
- Whether the muon_flat vs adam_minsnr ranking is preserved at 84M
- Whether weight decay matters (it should, per "Muon is Scalable")

**Small config details** (from model.py):
- d_model=512, n_layers=8, d_state=64, headdim=64, expand=2, seq_len=512
- Note: seq_len is 512 (not 1024), so tokens/step = 8 x 512 = 4096
- Tokens per 3k-step experiment: 12.3M tokens
- Estimated GPU time: ~30-40 min each (larger model, shorter seq)

### Phase 3 (optional, if Phase 1-2 confirm hypothesis): Convergence Crossover Test

If Muon-flat is winning at 5k steps and the gap is still widening, push to 10k steps on quokka to hunt for any late-training crossover.

```bash
python train.py --config quokka --batch_size 8 --max_steps 10000 \
    --val_every 500 --log_every 100 --warmup_steps 400 \
    --no_time_cond --lr_schedule cosine \
    --optimizer muon --loss_weight flat --muon_lr 0.02 --adam_lr 3e-4
```

This would see 81.9M tokens, roughly 2.3 tokens/param for quokka 36M. Still far from convergence on 1B tokens, but enough to check trajectory.

### autoresearch.py Integration

Add a new mode `scaling_convergence` to autoresearch.py:

```python
def scaling_convergence(args):
    """Phase 1+2: Muon-flat vs Adam-minsnr at longer training and larger scale."""
    base_args = {
        "batch_size": args.batch_size,
        "val_every": 250,
        "log_every": 50,
        "no_time_cond": True,
        "lr_schedule": "cosine",
    }

    experiments = [
        # Phase 1: quokka, 5000 steps
        ("muon_flat_quokka_5k", {**base_args, "config": "quokka",
            "max_steps": 5000, "warmup_steps": 200,
            "optimizer": "muon", "loss_weight": "flat",
            "muon_lr": 0.02, "adam_lr": 3e-4}),
        ("adam_minsnr_quokka_5k", {**base_args, "config": "quokka",
            "max_steps": 5000, "warmup_steps": 200,
            "optimizer": "adam", "loss_weight": "minsnr",
            "adam_lr": 3e-4}),
        ("adam_flat_quokka_5k", {**base_args, "config": "quokka",
            "max_steps": 5000, "warmup_steps": 200,
            "optimizer": "adam", "loss_weight": "flat",
            "adam_lr": 3e-4}),
        # Phase 2: small 84M, 3000 steps
        ("muon_flat_small_3k", {**base_args, "config": "small",
            "max_steps": 3000, "warmup_steps": 200,
            "optimizer": "muon", "loss_weight": "flat",
            "muon_lr": 0.02, "adam_lr": 3e-4,
            "muon_wd": 0.01, "adam_wd": 0.01}),
        ("adam_minsnr_small_3k", {**base_args, "config": "small",
            "max_steps": 3000, "warmup_steps": 200,
            "optimizer": "adam", "loss_weight": "minsnr",
            "adam_lr": 3e-4, "adam_wd": 0.01}),
        ("muon_flat_small_3k_nowd", {**base_args, "config": "small",
            "max_steps": 3000, "warmup_steps": 200,
            "optimizer": "muon", "loss_weight": "flat",
            "muon_lr": 0.02, "adam_lr": 3e-4,
            "muon_wd": 0.0, "adam_wd": 0.0}),
    ]

    results = []
    for name, exp_args in experiments:
        record = run_experiment(name, exp_args)
        results.append(record)

    # Analysis: compare trajectory shapes
    print("\n" + "=" * 60)
    print("SCALING & CONVERGENCE SUMMARY")
    print("=" * 60)
    for config_name in ["quokka", "small"]:
        config_results = [r for r in results if config_name in r["name"]]
        config_results.sort(key=lambda r: r["val_loss"])
        print(f"\n  {config_name}:")
        for r in config_results:
            print(f"    {r['name']:>30s}: val_loss={r['val_loss']:.4f}, "
                  f"time={r['elapsed_seconds']:.0f}s")

    return results
```

## Expected Outcome

### If hypothesis is confirmed (most likely):
- muon_flat beats adam_minsnr at 5000 steps on quokka by >= 0.04 nats (currently 0.04 at 1000 steps)
- The gap is constant or widening when plotted against training tokens
- muon_flat beats adam_minsnr at 3000 steps on small (84M) config
- Weight decay matters for Muon at 84M (no-WD variant diverges or underperforms)

### If hypothesis is partially rejected:
- Gap narrows at 5k steps (Muon converges faster but Adam catches up) -- would suggest Muon is a "fast starter" but not fundamentally better for diffusion LMs
- Ranking flips at 84M -- would suggest the flat-weighting advantage is scale-dependent
- muon_minsnr becomes competitive at longer training -- would suggest 1000 steps was too short for minsnr to show its benefit with Muon

### If hypothesis is fully rejected:
- Adam-minsnr overtakes Muon-flat before 5000 steps -- would mean the 1000-step result was misleading, and the masked diffusion loss landscape favors per-parameter adaptivity at convergence

### Decision criteria:
- **Gap >= 0.04 nats at 5k, same sign at 84M:** Muon-flat is the production default. Proceed to longer runs and downstream eval.
- **Gap 0.01-0.04 nats, inconsistent across scales:** Results are noisy. Need multiple seeds (2-3) to confirm. Budget an additional ~3 hours of GPU time.
- **Gap < 0.01 or sign flip:** Adam-minsnr is simpler and more robust. Default to Adam unless Muon shows wall-clock savings.

## Risk/Cost

### GPU time budget:
| Experiment | Steps | Est. time (PureSSM) | Est. time (Mamba3 Triton) |
|---|---|---|---|
| Phase 1: 3x quokka 5k | 15,000 total | ~2.25 hrs | ~1.25 hrs |
| Phase 2: 3x small 3k | 9,000 total | ~2.0 hrs | ~1.0 hrs |
| **Total** | **24,000 steps** | **~4.25 hrs** | **~2.25 hrs** |

Phase 3 (optional 10k step run) adds another ~50 min with Mamba3 Triton.

### Risks:
1. **VRAM OOM on small config:** The 84M "small" model with d_model=512, 8 layers, batch_size=8 should fit in 16GB with gradient checkpointing enabled (PureSSM backend). If it doesn't, reduce batch_size to 4 and double max_steps to 6000 to keep total tokens constant.

2. **Cosine schedule mismatch:** The cosine LR schedule decays relative to max_steps. A 5000-step run's LR at step 1000 differs from the original 1000-step run's LR at step 1000. This means we cannot directly compare loss-at-step-1000 between the two budgets. We compare only final val_loss and trajectory shape.

3. **Single seed noise:** At 0.04 nat gap, single-seed results are suggestive but not conclusive. If the gap is close, budget 2 additional seeds per config (2x GPU time for the ambiguous configs only).

4. **Data regime concern:** At 84M params with 1B tokens, we're at ~12 tokens/param. The "Muon is Scalable" paper's results are at higher tokens/param ratios. The optimizer comparison might not transfer to the deeply underfitting regime. This is a feature, not a bug -- we want to know if Muon's advantage is specifically a "sample efficiency" effect or structural.

5. **PureSSM speed:** If Mamba3 Triton is broken on RDNA4 and we're stuck on PureSSM, the 84M model will be slow. Cap Phase 2 experiments at 90 min each; if they'd exceed that, reduce to 2000 steps.

## Literature Support

### Muon scaling (strong support for hypothesis)
- **"Muon is Scalable for LLM Training"** (Liu et al., Moonshot AI, Feb 2025): Demonstrated Muon at 3B/16B MoE with 5.7T tokens. Key finding: ~2x compute efficiency over AdamW with weight decay + per-parameter update scale calibration. Scaling laws hold across model sizes.
- **"Practical Efficiency of Muon for Pretraining"** (Essential AI, May 2025): Across 100M-4B models, Muon "strictly lower bounds AdamW on loss curves, even to the end of training far beyond the Chinchilla-optimal budget (no crossover)." 10-15% token savings. This is the strongest evidence that our advantage should persist.
- **NorMuon** (Oct 2025): Improved Muon variant with 21.74% better training efficiency than Adam, 11.31% over Muon. Suggests even more headroom exists.

### Masked diffusion loss weighting (moderate support)
- **"Masked Diffusion Language Models with Frequency-Informed Training"** (Sep 2025, BabyLM): Frequency-based masking (prioritize rare tokens) improves sample-efficient MDLM training. Orthogonal to our optimizer question but suggests further loss reweighting improvements are possible on top of flat weighting.
- **"Soft-Masked Diffusion Language Models"** (Hersche et al., ICLR 2026): Soft masking (blend mask embedding with top-k predictions) improves perplexity. Trains at 169M scale. Uses standard MDLM objective -- does not explore optimizer choice. A future direction for this project.

### Diffusion + Mamba (context)
- **DiffuMamba** (Nov 2025, revised Feb 2026): The foundational work. Scales to 1.3B params with bidirectional Mamba-2, matching Transformer-based diffusion. Uses AdamW -- never tested Muon. Our project is the first to combine Muon with masked diffusion LMs.

### Key gap in literature
No paper has studied Muon specifically on masked diffusion language models. The Muon scaling papers all use standard autoregressive cross-entropy. Our flat-weighting insight (Muon needs uniform gradient scale) is novel and has not been externally validated. This experiment would be the first data point on whether that insight transfers across training budgets and model scales.

Sources:
- [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982)
- [Practical Efficiency of Muon for Pretraining](https://arxiv.org/abs/2505.02222)
- [NorMuon: Scalable Efficient LLM Optimization](https://arxiv.org/abs/2510.05491)
- [Masked Diffusion Language Models with Frequency-Informed Training](https://arxiv.org/abs/2509.05056)
- [Soft-Masked Diffusion Language Models (ICLR 2026)](https://arxiv.org/abs/2510.17206)
- [DiffuMamba: High-Throughput Diffusion LMs with Mamba Backbone](https://arxiv.org/abs/2511.15927)
- [Keller Jordan Muon Repository](https://github.com/KellerJordan/Muon)
