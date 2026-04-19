#!/usr/bin/env bash
# 30M PAPL τ-sweep — 1 seed each, 4 τ values, ~3h total.
# Final eval compares all 4 PAPL endpoints + vanilla 55k + real + AR baseline.
#
# Run from /mnt/d/code/gpt-slide so muon_exp/outputs paths resolve.

set -euo pipefail

VENV=/mnt/d/code/gpt-slide/.venv/bin/python
TRAIN=/home/clundquist/modded-diffumamba/nvidia/training/finetune_papl_30m.py
HARNESS=/home/clundquist/modded-diffumamba/nvidia/eval/gen_harness/harness.py
OUTROOT=/mnt/d/code/gpt-slide/muon_exp/outputs

LOGDIR=/tmp/30m_papl_sweep
mkdir -p $LOGDIR

run_tau() {
  local tau=$1
  local label=$2
  local outdir=$OUTROOT/30m_papl_${label}
  echo "===== $(date) — PAPL τ=${tau} (label ${label}) ====="
  $VENV $TRAIN --alpha 1.0 --tau $tau --extra-steps 5000 \
    --resume $OUTROOT/transformer_converge_v3/checkpoint_50000.pt \
    --output-dir $outdir --seed 42 --genprobe-every 1000 \
    2>&1 | tee $LOGDIR/papl_${label}.log
}

run_tau 1.0   tau1
run_tau 0.3   tau03
run_tau 0.1   tau01
run_tau 0.03  tau003

echo "===== $(date) — final eval ====="
$VENV $HARNESS --models real \
  d_modern_30m_50k d_modern_30m_55k \
  d_modern_30m_papl_tau1 d_modern_30m_papl_tau03 \
  d_modern_30m_papl_tau01 d_modern_30m_papl_tau003 \
  rhysjones_gpt2_124m_fineweb_edu \
  --batch-size 16 \
  --save-samples /home/clundquist/modded-diffumamba/nvidia/eval/gen_harness/samples_30m_papl_sweep.pt \
  2>&1 | tee $LOGDIR/final_eval.log

echo "===== $(date) — sweep complete ====="
