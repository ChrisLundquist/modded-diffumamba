# DiffuMamba3: Masked Diffusion LM with Mamba-3 + Muon

## Project Goal

Build a nanogpt-clean, single-file implementation of a masked diffusion language model
using a bidirectional Mamba-3 backbone and Muon optimizer, targeting the AMD RX 9070 XT
(RDNA 4 / ROCm). Use Karpathy's autoresearch methodology to systematically find the
fastest training configuration.

## Key Research Question

Can Muon accelerate masked diffusion LM training? Standard Muon fails for image diffusion
(noise prediction loss), but masked diffusion uses cross-entropy over masked positions —
structurally more like MLM where Muon excels. This is an untested hypothesis.

## Architecture

- **Backbone:** Bidirectional Mamba-3 (forward + backward scan, additive merge)
  - Complex-valued states, MIMO formulation
  - Half the state size of Mamba-2 for same perplexity (VRAM-friendly)
  - Optional: hybrid with ~20% attention layers (DiffuMamba-H style)
- **Diffusion:** MDLM-style absorbing-state masked diffusion
  - Training = weighted mixture of MLM losses at different masking rates
  - Log-linear noise schedule with antithetic sampling
- **Optimizer:** Muon for 2D weights + AdamW for embeddings/biases/layernorm
- **Optional upgrades:** Soft masking (Hersche et al. ICLR 2026)

## Hardware Target

AMD RX 9070 XT (16GB VRAM, RDNA 4, gfx1201)
- ROCm 7.2+ required
- Flash attention via Triton backend: `FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"`
- PyTorch 2.8+ with ROCm
- Set `PYTORCH_TUNABLEOP_ENABLED=1` for kernel tuning

## Key Files

- `notes/papers.md` — annotated paper summaries with key technical details
- `notes/rocm-rdna4-rx9070xt-research.md` — ROCm/9070 XT setup guide
- `papers/mamba-bidirectional-research-survey.md` — comprehensive Mamba survey

## Baselines

- **DiffuMamba** (Singh et al. 2025): BiMamba-2 + MDLM + Adam. PPL 20.17 at 1.3B. 8.2x throughput.
- **LLaDA** (Nie et al. 2025): Masked diffusion at 8B scale, competitive with LLaMA3 8B.
- **nanogpt-speedrun**: GPT-2 124M in ~90sec on 8xH100 with Muon.

## Style

nanogpt-clean: minimal dependencies, single training script, readable code.
