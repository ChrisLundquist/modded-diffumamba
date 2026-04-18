#!/usr/bin/env bash
# Run the four objective-ablation fine-tunes sequentially.
#
# Each fine-tune resumes from checkpoint_40000.pt and runs +10k steps under a
# different MDLM objective variant. After each, the harness eval is queued so
# we get the leaderboard row immediately.
#
# Total compute: 4 × (3.5h fine-tune + 10 min eval) ≈ 14 hours. Designed to
# run overnight after Exp 4 (PAPL fine-tune) signals positive.
#
# Run from /mnt/d/code/gpt-slide (so muon_exp/outputs paths resolve).

set -euo pipefail

VENV=/mnt/d/code/gpt-slide/.venv/bin/python
TRAIN=/home/clundquist/modded-diffumamba/nvidia/training/finetune_objective_125m.py
HARNESS=/home/clundquist/modded-diffumamba/nvidia/eval/gen_harness/harness.py

LOGDIR=/tmp/obj_ablation_logs
mkdir -p $LOGDIR

run_one() {
  local obj=$1
  local outdir=$2
  local spec=$3
  local extra_args="${4:-}"
  local log=$LOGDIR/${obj}.log

  echo "===== $(date) — starting $obj ====="
  $VENV $TRAIN --objective $obj --extra-steps 10000 \
    --output-dir /mnt/d/code/gpt-slide/$outdir $extra_args 2>&1 | tee $log

  echo "===== $(date) — eval $obj ====="
  $VENV $HARNESS --models real d_modern_125m_40k d_modern_125m_50k $spec \
    rhysjones_gpt2_124m_fineweb_edu --batch-size 16 \
    --save-samples /home/clundquist/modded-diffumamba/nvidia/eval/gen_harness/samples_${obj}.pt \
    2>&1 | tee -a $log
}

# Order chosen so cheapest signal comes first: baseline (control) → t_curr →
# gamma_decay → papl_t_curr (combined). PAPL-alone is already covered by the
# separate finetune_papl_125m.py run.
run_one baseline       muon_exp/outputs/125m_baseline_finetune       d_modern_125m_baseline_50k
run_one t_curriculum   muon_exp/outputs/125m_t_curriculum_finetune   d_modern_125m_t_curr_50k
run_one gamma_decay    muon_exp/outputs/125m_gamma_decay_finetune    d_modern_125m_gamma_decay_50k
run_one papl_t_curr    muon_exp/outputs/125m_papl_t_curr_finetune    d_modern_125m_papl_t_curr_50k

echo "===== $(date) — all ablations done ====="
