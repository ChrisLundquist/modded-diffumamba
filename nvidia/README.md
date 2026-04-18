# NVIDIA (RTX 5090) Collaboration

Parallel experiments run on NVIDIA RTX 5090 (32GB) using the `mamba-ssm` built-from-source backend (not the Triton kernel used in the main repo). Focus: Muon for transformers, architecture ablations, and the ELBO-vs-generation-quality question.

## Summary of findings

### Validated by cross-agent replication
- **Muon config for Mamba3 SSMs is sensitive**: gamma ≤ 2 essential, param routing matters (out_proj helps when LR is right, doesn't when wrong), Muon LR 2x higher than AMD hardware. Cross-agent validation: "hardware-specific" Muon ranking was actually config sensitivity.
- **Muon NS precision (bf16/fp16/fp32) is irrelevant** — tested all three on mamba-ssm + Mamba3, identical within 0.002 nats. Not the cause of any cross-hardware differences.

### Transformer-specific (complementary to main repo's SSM work)
- **RoPE is 95% of the "modern transformer" advantage for MDLM** (3-seed ablation, std 0.007). SwiGLU alone is slightly harmful at 30M; AdaLN×SwiGLU interaction gives +0.03 nats at convergence.
- **Embedding LR is critical for mixed-optimizer setups**: AdamW lr=1.5e-4 (standard) vs Adam lr=1e-3 on embeddings differs by 0.20 nats. The "Mamba beats Transformer" headline at 30M was this artifact.
- **Muon beats AdamW by 0.077 nats on MDLM transformers at 30M** — but confounded with schedule (Muon+constant vs AdamW+cosine). Needs disentangling at 125M.
- **QK-norm + WSD schedule + 1500-step warmup** prevents divergence at 125M where vanilla Muon lr=0.02 diverged.

### The hard truth (2026-04-18)
- **ELBO gains don't translate to generation coherence.** 125M D_modern (best ELBO 4.499 MinSNR gamma=5) produces **worse text than a 4-gram Markov chain** on 10B tokens. Multi-seed (n=10) confirms: avg unique word ratio 29% at 125M vs 62% at 30M Mamba3.
- **Sampler bug** (now fixed) was inflating Mamba3's apparent generation advantage by biasing unmasking toward peaky positions. Under fixed top-k=50 sampling, 30M D_modern beats 30M Mamba3 on diversity, but both still produce fragmented output.

## Directory

```
nvidia/
├── src/
│   ├── gpt2.py              # GPT-2 transformer (baseline, causal=False for MDLM)
│   ├── transformer_v2.py    # D_modern: RoPE + SwiGLU + U-Net + 6-way AdaLN + QK-norm
│   ├── muon.py              # Muon optimizer with weight decay support
│   ├── hybrid_model.py      # DiffuMambaH: includes QuokkaBlock (matches main repo arch)
│   ├── adaln.py             # AdaLN primitives (zero-init modulation)
│   └── data.py              # TokenDataset, DataLoader helpers
├── training/
│   ├── train_dmodern_30m.py       # 30M D_modern, our best small-scale result
│   ├── train_dmodern_125m.py      # 125M D_modern on 10B (WSD + QK-norm, val 4.499)
│   ├── train_published_baseline.py # Faithful MDLM reproduction (AdamW+cosine)
│   ├── sweep_variants_3seeds.py   # A/B/C/D/E variant probe (3 seeds × 5 configs)
│   ├── sweep_muon_config.py       # Cross-agent Muon config match (5 configs)
│   ├── sweep_rope_swiglu_ablation.py  # 2×2 RoPE/SwiGLU ablation
│   ├── sweep_ns_precision.py      # bf16/fp16/fp32 NS test (all identical)
│   └── sweep_transformer_lr.py    # Transformer LR sensitivity study
├── probes/
│   ├── generation_multi_seed.py   # 10-seed generation + category analysis
│   ├── sampler_ablation.py        # Sampler comparison (greedy/nucleus/top-k)
│   └── cloze_basic.py             # Fill-in-the-blank factual probe
├── eval/
│   ├── standard_nll.py            # 3 metrics: proper ELBO, MinSNR γ=5, 1/t ELBO
│   ├── profile_vram.py            # Actual VRAM for scale planning
│   └── download_fineweb_10b.py    # 10B tokens with chunked .npy save (no OOM)
└── HANDOFF_nvidia.md              # Full NVIDIA-side experiment history
```

## Import convention

Training scripts expect `src/` to be on the Python path:
```python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
```

## Key checkpoints (not committed — local only)

| Model | Path (local) | Best val |
|-------|--------------|----------|
| 125M D_modern (10B, 1ep) | `outputs/125m_10b_dmodern/checkpoint_72479.pt` | 4.499 |
| 30M D_modern (1B, 7ep)   | `outputs/transformer_converge_v3/checkpoint_56000.pt` | 5.272 |
| 30M Mamba3 (1B, 7ep)     | `outputs/mamba3_converge/checkpoint_56000.pt` | 5.666 |
| 30M Published GELU+AdaLN | `outputs/published_baseline/checkpoint_56000.pt` | 5.349 |
| 30M Published SwiGLU+AdaLN | `outputs/published_swiglu/checkpoint_56000.pt` | 5.317 |

## Environment notes

- **mamba-ssm**: built from source at `/tmp/mamba_build`, installed to `.venv`
- **Patch**: `cpp_extension.py` CUDA version check softened (RuntimeError → warning) for CUDA 12.8 nvcc with PyTorch CUDA 13.0
- **GPU**: RTX 5090 32GB, sm_120 (Blackwell), WSL2 Ubuntu 24.04
- **Not usable on 5090**: MIMO kernels (shared-memory limit, works only on datacenter GPUs)

## Cross-agent disagreements resolved

1. **Muon vs Adam for Mamba3** — appeared to be hardware-specific, actually 4 config differences compounding. Main repo's Muon-VS + gamma=1.5 + in+out_proj + lr=0.02 + Adam 3e-4 transfers to NVIDIA hardware with minor retuning.
2. **NS step × gamma interaction** at 1K steps — we saw modest gamma effects but no reversals at 5K. Their 1-nat interactions at 1K were variance.

## What the main repo's backbone doesn't have (yet)

- Transformer baselines with matched training code path (for direct MDLM arch comparison)
- D_modern-style variants (U-Net skips, variable depth, QK-norm)
- 3-metric ELBO eval (proper ELBO + MinSNR + 1/t for cross-agent comparison)
- WSD schedule implementation (we use it for 125M single-epoch runs)
- QK-norm, for attention stability at scale
