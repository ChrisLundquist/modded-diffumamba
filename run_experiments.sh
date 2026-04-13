#!/bin/bash
# Run validation experiments: muon_flat vs adam_minsnr at 5000 steps
# Usage: bash run_experiments.sh
set -euo pipefail

export LD_PRELOAD=./librocprofiler_stub.so
# NOTE: expandable_segments crashes on RDNA4 (hipErrorInvalidValue)
# See ~/BUG_expandable_segments_rdna4.md

echo "=== Tier 1 Validation: 5000 steps ==="

# Run muon_flat (the winner from 1000-step grid)
echo ""
echo ">>> muon_flat_5k"
.venv/bin/python3 train.py --config quokka --batch_size 8 --max_steps 5000 \
    --val_every 500 --log_every 250 --warmup_steps 200 \
    --no_time_cond --lr_schedule cosine \
    --optimizer muon --loss_weight flat --muon_lr 0.02 --adam_lr 3e-4

# Run adam_minsnr (the best Adam config)
echo ""
echo ">>> adam_minsnr_5k"
.venv/bin/python3 train.py --config quokka --batch_size 8 --max_steps 5000 \
    --val_every 500 --log_every 250 --warmup_steps 200 \
    --no_time_cond --lr_schedule cosine \
    --optimizer adam --loss_weight minsnr --adam_lr 3e-4

# Tier 2: Adam LR sweep with flat (check if Adam just needs higher LR)
echo ""
echo ">>> adam_flat_lr1e3"
.venv/bin/python3 train.py --config quokka --batch_size 8 --max_steps 5000 \
    --val_every 500 --log_every 250 --warmup_steps 200 \
    --no_time_cond --lr_schedule cosine \
    --optimizer adam --loss_weight flat --adam_lr 1e-3

echo ""
echo "=== ALL DONE ==="
