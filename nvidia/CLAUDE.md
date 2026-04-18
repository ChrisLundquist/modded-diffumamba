# nvidia/ subdirectory — context for agents

## What this is

Parallel MDLM experiments on NVIDIA RTX 5090 (32GB) using the `mamba-ssm` built-from-source backend. Complements the main repo's AMD RX 9070 XT work with the Triton backend. Focus: **transformer baselines**, **architecture ablations**, and **ELBO-vs-generation** stress-testing of the main repo's ELBO-based claims.

## Status when handed off (2026-04-18)

- All four convergence runs done at 30M/1B/7epoch scale (D_modern, published baselines GELU & SwiGLU, Mamba3)
- One 125M/10B/1epoch scale-up done (D_modern with RoPE+SwiGLU+QK-norm+WSD)
- Cross-agent Muon-config sensitivity findings resolved (config, not hardware)
- **The ELBO-vs-generation decoupling is confirmed and is the most important finding on either side**

## The honest story

Our headline evolved three times over this work:

1. **Round 1** (n=1, multi-epoch): "Mamba3 beats Transformer by 0.21 nats"
2. **Round 2** (after finding embedding LR bug): "Fixed Transformer beats Mamba3 by 0.22 nats"
3. **Round 3** (3-seed ablation): "RoPE accounts for 95% of the D_modern win — and RoPE is already standard in MDLM papers"
4. **Round 4** (multi-seed generation probe, 125M/10B): "Our best-ELBO model generates worse text than a 4-gram Markov chain"

Each round was discovered by scrutinizing our own claims. The sampler bug in the initial generation probe made Mamba3 look much better than it was — fixing it (pre-nucleus ordering, top-k=50) collapsed that advantage.

## What's defensible vs killed

| Claim | Status |
|---|---|
| Muon config sensitivity for SSMs (cross-agent) | **Solid** |
| NS precision (bf16/fp16/fp32) irrelevant | Solid negative result |
| Embedding LR matters enormously | Real, but scale-dependent |
| RoPE + SwiGLU + QK-norm + WSD stable at 125M | Solid engineering |
| "D_modern beats transformer baseline by X nats" | **All gain is RoPE** (killed by ablation) |
| "Mamba3 beats Transformer" | Killed (was embedding LR artifact) |
| "Muon beats AdamW by 0.08 nats" | n=1, confounded with schedule (reviewer will reject) |
| "SwiGLU is novel for MDLM" | Killed by our own 2x2 ablation |
| "Mamba3 > Transformer on generation" | Killed (sampler bug + multi-seed check) |
| "ELBO ≠ generation quality" at 125M | **Solid, multi-seed, n=10** |

## The collaboration

The **other agent** works in the main repo (`~/modded-diffumamba/`) on the AMD RX 9070 XT with the Triton Mamba3 backend. They have their own HANDOFF.md, CLAUDE.md, and experiment sweeps.

Cross-agent findings:
- Their "Muon+VS beats Adam by 0.45 nats on Mamba3" didn't appear to transfer to our hardware — but it was 4 config differences compounding, not hardware. Their config (in_proj only, lr=0.02, gamma=1.5, Adam 3e-4, wd=0.01) beats Adam by 0.011 nats on our NVIDIA hardware too.
- Their "sampler fix" on 2026-04-17 and our fix on 2026-04-18 were independent discoveries of the same class of bug (ordering by post-nucleus probs).
- Their gen-PPL metric (GPT-2-scored samples) and our MinSNR γ=5 ELBO disagreed sharply — we added 1/t ELBO to `eval/standard_nll.py` for direct cross-comparison.

When uncertain, check the main repo's HANDOFF.md and CLAUDE.md.

## Directory structure

```
nvidia/
├── README.md              # user-facing overview
├── CLAUDE.md              # this file — agent context
├── HANDOFF_nvidia.md      # detailed experiment log
├── src/                   # model code (import from '../src')
├── training/              # training scripts (one per experiment)
├── probes/                # cloze + generation probes
└── eval/                  # ELBO eval, VRAM profiler, data downloader
```

## Environment

- `.venv` at `/mnt/d/code/gpt-slide/.venv` (`--system-site-packages`)
- mamba-ssm built from source, patched `cpp_extension.py`
- 10B FineWeb-Edu at `/home/clundquist/muon_data/fineweb_10B.npy` (uint16, 19GB)
- 1B FineWeb-Edu at `/home/clundquist/muon_data/fineweb_1B.npy` (uint16, 2.1GB)
- Checkpoints live at `/mnt/d/code/gpt-slide/muon_exp/outputs/` (NOT in repo)

## What an incoming agent should probably do

1. **Read `HANDOFF_nvidia.md` sections "What We Found" and "Most Impactful Next Experiments"** — the full current state.
2. **Check the main repo's `HANDOFF.md`** — their current state + cross-agent findings.
3. **Before making any claim**: run it through the "killed vs defensible" table above. If the claim is n=1 with effect <0.15 nats, it's probably not real.
4. **Before any sampler-based claim**: verify ordering uses pre-nucleus probs, prefer top-k=50 over nucleus.
5. **Before any optimizer claim**: the 2x2 {Muon, AdamW} × {constant, cosine} at 125M/10B with 3 seeds is the single highest-value unrun experiment.

## What NOT to do

- Don't claim SwiGLU is novel for MDLM — our ablation killed that.
- Don't sell ELBO wins as generation wins — multi-seed shows they decouple.
- Don't chase Muon-vs-Adam as a headline without disentangling schedule.
- Don't add new baselines without matching embedding LR to the published recipe.
- Don't commit checkpoints, the 10B .npy, or anything else large. `.gitignore` covers `*.pt`, `*.npy`, `*.bin`, `nvidia/outputs/`, `nvidia/data_cache/`.
