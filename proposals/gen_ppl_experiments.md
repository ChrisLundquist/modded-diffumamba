# Gen-PPL improvement experiments: fail-fast plan

Proposed after the research agent's survey (2026-04-19) identified three
techniques to directly improve gen_PPL independent of ELBO. Ranked by
expected (gen_PPL gain / GPU min). Reviewed and posted here so either
agent can pick them up when GPU time is available.

## 1. ReMDM remasking sampler — HIGHEST ROI (inference-only)

**Implementation status:** shipped in `model.py:sample()` via
`config.remdm_sigma_max` + `config.remdm_schedule`. CLI flags on
`train.py` and `probe_checkpoint.py`.

**MVE** (~12 min GPU, one existing checkpoint):

```bash
# Target: our best existing generative model (10L×640d @ 30k, gen_PPL 54.3)
for S in 0.05 0.10 0.20 0.30; do
  for SCHED in cap linear; do
    .venv/bin/python probe_checkpoint.py \
      --ckpt checkpoints/10L640d_50k_step30000.pt \
      --config quokka --n_layers 10 --d_model 640 \
      --n_samples 128 --seq_len 1024 --num_steps 1024 \
      --chunk 8 --top_k 50 \
      --remdm_sigma_max $S --remdm_schedule $SCHED \
      2>&1 | tee logs/remdm_10L_${SCHED}_s${S}.log
  done
done
```

8 probes × ~80s = ~11 min. Baseline (no ReMDM) is 54.3 from prior runs.

**Fail-fast (kill technique):**
- No σ × schedule combo improves gen_PPL by ≥1.5 PPL (relative <3%).
- All combinations tie baseline within n=128 seed noise (±1.2 PPL).

**Kill-switch:** rep_4 > 0.02 at the winning σ. ReMDM is "exploiting
the LM" rather than denoising — not a real win.

**If promising (<2 hr):** best (σ*, sched*) at n=512 + 3 sampling
seeds, cross-checkpoint on E_best_10k_s42 and mechanism-capture ckpts.

---

## 2. BD3LM clipped noise — single-seed smoke first

**Implementation status:** shipped via `config.clip_t_min` /
`clip_t_max`. CLI: `--clip_t_min --clip_t_max`. Val loop is bypass-safe
(clipping only active during self.training=True).

**MVE** (~30 min GPU):

```bash
.venv/bin/python train.py --config quokka \
  --optimizer muon --muon_variant vs --muon_lr 0.01 --adam_lr 3e-4 \
  --muon_out_proj --loss_weight minsnr --minsnr_gamma 1.5 \
  --data_dir data/fineweb-edu-10B \
  --batch_size 8 --max_steps 5000 --seed 42 --save_best \
  --clip_t_min 0.3 --clip_t_max 0.8 \
  --save_path checkpoints/bd3lm_smoke_s42.pt \
  --val_decomp --gen_probe --gen_probe_every 2 \
  --gen_probe_final --gen_probe_final_samples 64 \
  2>&1 | tee logs/bd3lm_smoke.log
```

Baseline comparator: Muon@0.10 seed-42 @ 5k (we have that from B).

**Fail-fast (kill technique):**
- val_loss worse by >0.1 nats AND gen_PPL worse/tied vs baseline.
- val_loss worse by >0.25 nats regardless of gen_PPL (ELBO collapse).

**Kill-switch:** training instability (loss spike, NaN), or wall-clock
regression >25%.

**If promising (<90 min):** 3 seeds × {unclipped, (0.3, 0.8), (0.1, 0.9)}
paired comparison.

---

## 3. SDTT self-distillation — deferred until 1 or 2 succeeds

**Implementation status:** NOT implemented. Would need teacher→student
reverse-KL distillation pipeline.

**MVE sketch** (~30 min if implemented):
- Teacher: `checkpoints/10L640d_50k_step30000.pt` (111.7M, gen_PPL 54.3)
- Student: quokka-from-scratch or existing quokka ckpt
- Generate 4k teacher trajectories offline (~15 min with ReMDM-optimal sampler)
- Student trains reverse-KL on (x_t, teacher_logits) for 2k steps (~12 min)

**Fail-fast:** student gen_PPL ≥ non-distilled quokka baseline (73),
or trajectory storage exceeds 8GB.

**If promising:** scale teacher-student to matched architecture, 10k
distill steps.

---

## Ordering

1. **Run ReMDM MVE immediately** when GPU frees (no training, uses
   existing ckpts).
2. **BD3LM smoke** after F resumes + completes (needs dedicated GPU time).
3. **SDTT implementation** only if #1 or #2 gives unambiguous positive
   signal justifying ~1-2 days of porting.

## Constraints

- Don't start ReMDM while F is running (would fragment GPU).
- ReMDM probes use the existing `probe_checkpoint.py` with `--chunk 8`
  for 1024-token samples (the Gumbel tensor at vocab=50304 needs
  smaller batches than 128-token probes).
- BD3LM clipped noise training uses our live train.py; no script
  changes needed.

## Known caveats

- **ReMDM "cap" schedule:** implementation uses σ_cap = t_next/t
  (Wang paper's bound for MDLM log-linear). Verified by math but not
  bit-compared to reference github.com/kuleshov-group/remdm.
- **BD3LM val_loss:** intentionally bypasses clipping during eval
  (model.py:_sample_t) so val_loss remains a proper ELBO and is
  comparable across clipped/unclipped runs.
- **ReMDM remasks all unmasked tokens (not just newly unmasked).**
  Wang paper allows both variants; this is the simpler one.
