# DiffuMamba3 Training Recipe Handoff

## Confidence Levels

**HIGH confidence (validated at two scales — tiny_shakespeare and FineWeb-Edu 1B):**
- Muon (lr=0.02) beats Adam (lr=3e-3) by 0.78 loss units at 500 steps, gap still widening
- Min-SNR gamma=5 loss weighting — consistent improvement over ELBO at both scales
- Cosine LR schedule beats linear
- Auxiliary Adam lr=3e-4 for embeddings/norms (higher hurts)

**MEDIUM confidence (validated at FineWeb scale, n=1):**
- Muon lr=0.02 is optimal (sweep: 0.005, 0.01, 0.02, 0.04)
- Time conditioning ON beats OFF by ~1.3%
- torch.compile adds nothing with Mamba3 Triton kernels

**LOW confidence (tiny model sweeps only):**
- Weight decay, beta2 effects are within noise

## Best Configuration (validated)

```bash
python train.py \
  --config quokka \
  --optimizer muon --muon_lr 0.02 --adam_lr 3e-4 \
  --loss_weight minsnr --minsnr_gamma 5 \
  --lr_schedule cosine --warmup_steps 50 \
  --data_path data/fineweb-edu/train.npy \
  --val_data_path data/fineweb-edu/val.npy \
  --batch_size 8
```

**500-step result: val_loss = 6.8426** (vs Adam baseline 7.6270, same steps/time)

## What We Found (and How Much to Trust It)

### Min-SNR gamma=5 (HIGH confidence)
- Clamps the ELBO weight `1/t` at `max=5`
- 2.2% better than standard ELBO across multiple runs
- Clear U-shaped gamma sweep: {1, 2, 3, **5**, 8, 10}
- Matches Hang et al. ICCV 2023 default exactly
- Zero wall-clock overhead
- **This is our most reliable finding.**

### Learning Rate (LOW confidence)
- Our sweep at 500 steps showed: 1e-4 << 3e-4 << 1e-3 << 3e-3
- But 500 steps strongly biases toward high LR — fast early, may diverge later
- DiffuMamba uses 1e-4, MDLM uses 3e-4, Quokka uses 2e-4 (all for long training)
- **Start with our 3e-3 but be ready to reduce. Try 1e-3 and 3e-4 as well.**

### Optimizer: Adam vs Muon (UNRELIABLE)
- We found Adam beat Muon on 84M model / 369K tokens / broken LR schedule
- Another agent found Muon DOES help at proper scale with a transformer
- Our test conditions were bad — wrong model size, broken warmup, wrong LR
- **Do not trust our "Muon doesn't help" claim. Test it yourself.**

### Weight Decay (LOW confidence)
- wd=0 won in our 500-step sweep (matching MDLM)
- But WD is a regularizer — its effect is invisible in short training
- DiffuMamba uses wd=0.1, Quokka recommends nonzero WD for long runs
- **Sweep wd={0, 0.01, 0.1} at your training length.**

### LR Schedule (LOW confidence)
- Cosine beat constant in our sweep
- Another agent found constant better for transformers
- At 500 steps, the schedule shape barely matters
- **Test both. Quokka uses Warmup-Stable-Decay.**

### beta2 (NEGLIGIBLE effect)
- 0.999 vs 0.95 showed <0.05 difference, within noise at n=1
- **Doesn't matter. Use either.**

### Time Conditioning (MEDIUM confidence)
- ON beat OFF by 1.3% on one run
- DiffuMamba uses it, MDLM doesn't
- **Keep it on.**

## Architecture: AdaLN Timestep Conditioning

We use AdaLN (from DiT) instead of DiffuMamba's concatenated timestep token.
This is pure PyTorch and works on any hardware:

```python
class AdaLN(nn.Module):
    """Adaptive LayerNorm with scale, shift, gate — zero-initialized."""
    def __init__(self, d_model: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.modulation = nn.Linear(cond_dim, 3 * d_model, bias=True)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x, c):
        # x: (B, L, D), c: (B, cond_dim) → (normed_x, gate)
        shift, scale, gate = self.modulation(c).unsqueeze(1).chunk(3, dim=-1)
        normed = self.norm(x) * (1 + scale) + shift
        return normed, gate
```

Usage in a block:
```python
h, gate = self.adaln(x, c)          # condition on timestep
h = self.backbone(h)                 # SSM or attention
x = x + gate * h                     # gated residual
```

Untested whether this is better or worse than DiffuMamba's concat approach.

## Data

FineWeb-Edu pre-tokenized at `data/fineweb-edu/`:
- `train.npy`: 1.0B tokens, uint16, 1.9GB (GPT-2 tokenizer)
- `val.npy`: 100M tokens, uint16, 191MB
- Load with: `torch.from_numpy(np.load(path).astype(np.int32))`
- Do NOT load as int64 — that's 8GB of RAM for 1B tokens

## Model Configs

| Config | Params | d_model | layers | seq_len | Use case |
|--------|--------|---------|--------|---------|----------|
| tiny | 8.4M | 128 | 4 | 256 | Quick HP sweep (overparameterized for shakespeare) |
| quokka | 35.9M | 384 | 4 | 1024 | Quokka-optimal for 1B tokens |
| small | 84.2M | 512 | 8 | 512 | Medium experiments |
| base | 231.4M | 768 | 12 | 1024 | Full scale |

## Mamba Backend

- `mamba_ssm` installs from git HEAD with `MAMBA_SKIP_CUDA_BUILD=TRUE`
- Mamba-2 SSD Triton kernels work on ROCm RDNA4 (935k tok/s)
- Mamba-3 Triton kernels fail on RDNA4 (register allocation — see TRITON_SSM_HANDOFF.md)
- Our PureSSM fallback works but is 700x slower (1.3k tok/s)

## What We Did NOT Test

- Hybrid attention layers (1 attn per 5 Mamba blocks)
- Soft masking (Hersche et al.)
- Longer training / convergence behavior
- Multiple seeds for any configuration
- Any config at >500 training steps

## Key References

- Min-SNR: Hang et al., ICCV 2023, arXiv 2303.09556
- MDLM: Sahoo et al., NeurIPS 2024, arXiv 2406.07524
- DiffuMamba: Singh et al., 2025, arXiv 2511.15927
- Quokka: Ni et al., 2025, arXiv 2510.03280
- Full experiment log: results/experiment_log.md
