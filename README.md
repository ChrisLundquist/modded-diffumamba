# DiffuMamba3: Autoresearch on Masked Diffusion LMs

A nanogpt-style research project exploring masked diffusion language models with
**Mamba-3** backbone and **Muon** optimizer, targeting AMD RX 9070 XT.

## Motivation

DiffuMamba (Singh et al., 2025) showed that replacing the Transformer backbone in
masked diffusion LMs with bidirectional Mamba-2 yields 8.2x throughput with no quality
loss. Two natural follow-ups:

1. **Mamba-3** (Gu et al., ICLR 2026) — complex-valued states, MIMO, half the state
   size of Mamba-2 for matching perplexity. Better fit for 16GB consumer GPUs.

2. **Muon optimizer** (Jordan et al.) — orthogonalized momentum that gave 1.35x speedup
   in the nanogpt speedrun. Never tested on masked diffusion LMs. The hypothesis: since
   MDLM training = cross-entropy (not noise prediction), Muon should transfer from LM
   training.

We combine these in a clean, single-file implementation and use **autoresearch** to
systematically search for the fastest training configuration.

## Architecture

```
Input tokens → Embed → [BiMamba3 Block × N] → Project → Logits
                              ↑
                        noise timestep t
```

Each BiMamba3 block:
- Forward Mamba-3 scan → h_fwd
- Backward Mamba-3 scan → h_bwd
- h = h_fwd + h_bwd (additive merge)
- MLP refinement + residual connection
- Noise conditioning via adaptive layernorm or embedding addition

Diffusion process (MDLM):
- Forward: progressively mask tokens with [MASK] at rate β(t)
- Reverse: predict original tokens from partially masked sequence
- Loss: weighted cross-entropy over masked positions

## Hardware

**AMD RX 9070 XT** (RDNA 4, 16GB VRAM, gfx1201)
- ROCm 7.2+ / PyTorch 2.8+
- Flash attention via Triton backend
- See `notes/rocm-rdna4-rx9070xt-research.md` for full setup guide

## Key Research Questions

1. Does Muon accelerate masked diffusion LM training? (Muon fails for image diffusion
   but masked diffusion loss = cross-entropy, not noise prediction)
2. Does Mamba-3's MIMO formulation help with bidirectional processing?
3. What's the optimal hybrid ratio (Mamba-3 vs attention layers)?
4. Does soft masking (Hersche et al., ICLR 2026) compound with Muon gains?
5. Can autoresearch find RDNA4-specific optimizations?

## Papers

See `notes/papers.md` for the full annotated bibliography. Key references:

| Paper | Key Contribution |
|-------|-----------------|
| MDLM (Sahoo et al., 2024) | Masked diffusion = mixture of MLM losses |
| SEDD (Lou et al., 2024) | Score entropy for discrete diffusion |
| DiffuMamba (Singh et al., 2025) | BiMamba-2 backbone for masked diffusion LMs |
| Mamba-3 (Lahoti et al., 2026) | Complex states, MIMO, half state size |
| Muon (Jordan et al., 2024) | Orthogonalized momentum, 1.35x nanogpt speedup |
| LLaDA (Nie et al., 2025) | Masked diffusion at 8B scale |
| Soft-Masked DLM (Hersche et al., 2026) | Soft masking improves masked diffusion |

## Project Structure

```
diffusion-lm-autoresearch/
├── CLAUDE.md           # Project context for Claude Code
├── README.md           # This file
├── notes/
│   ├── papers.md       # Annotated paper summaries
│   └── rocm-rdna4-rx9070xt-research.md  # Hardware setup guide
├── papers/
│   └── mamba-bidirectional-research-survey.md
└── src/                # Implementation (TBD)
    ├── train.py        # Single-file training script (nanogpt style)
    ├── model.py        # BiMamba3 diffusion model
    └── autoresearch.py # Automated experiment runner
```

## References

- [MDLM](https://arxiv.org/abs/2406.07524) | [SEDD](https://arxiv.org/abs/2310.16834) | [D3PM](https://arxiv.org/abs/2107.03006)
- [DiffuMamba](https://arxiv.org/abs/2511.15927) | [Mamba-3](https://arxiv.org/abs/2603.15569)
- [Muon](https://kellerjordan.github.io/posts/muon/) | [nanogpt-speedrun](https://github.com/KellerJordan/modded-nanogpt)
- [LLaDA](https://arxiv.org/abs/2502.09992) | [Soft-Masked DLM](https://arxiv.org/abs/2510.17206)
- [Karpathy autoresearch](https://github.com/karpathy/autoresearch) | [AMD fork](https://github.com/andyluo7/autoresearch)
