# DiffuMamba3: Autoresearch on Masked Diffusion LMs

Masked diffusion language model with **bidirectional Mamba-3** backbone and
**Muon** optimizer, in the style of [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt).
Targets AMD RX 9070 XT (RDNA 4 / ROCm).

## Quick Start

```bash
# 1. Download pre-tokenized FineWeb-10B (1B tokens, ~2GB)
python data/get_data.py          # uses huggingface_hub
# OR
bash data/get_data.sh            # uses curl (no Python deps)

# 2. Train
python train.py --config small --max_steps 5000

# 3. Autoresearch (Muon vs Adam sweep)
python autoresearch.py --mode compare_optimizers --budget_steps 500
```

## Architecture

```
Input tokens -> Embed -> [BiMamba3 Block x N] -> AdaLN -> LM Head -> Logits
                              ^
                         sigma(t) timestep
```

Each BiMamba3 block (following [DiffuMamba](https://arxiv.org/abs/2511.15927)):
- AdaLN conditioning on noise timestep
- Forward Mamba scan + backward Mamba scan (additive merge)
- SwiGLU MLP with gated residual

Diffusion ([MDLM](https://arxiv.org/abs/2406.07524)):
- Absorbing-state masked diffusion (tokens -> [MASK] at rate governed by t)
- Log-linear noise schedule
- SUBS parameterization (kill mask logit, force unmasked to copy)

## Data

Pre-tokenized FineWeb-10B shards from [kjj0/fineweb10B-gpt2](https://huggingface.co/datasets/kjj0/fineweb10B-gpt2) — same data as modded-nanogpt. Each `.bin` shard is ~200MB (100M GPT-2 tokens).

Alternative: tokenize your own parquets with `python data/tokenize.py`.

## Model Configs

| Config | Params | d_model | layers | seq_len | Use case |
|--------|--------|---------|--------|---------|----------|
| tiny   | 8.4M   | 128     | 4      | 256     | Quick HP sweep |
| quokka | 35.9M  | 384     | 4      | 1024    | 1B token scale |
| small  | 84.2M  | 512     | 8      | 512     | Medium experiments |
| base   | 231M   | 768     | 12     | 1024    | Full scale |
| large  | 350M   | 1024    | 24     | 1024    | Large-scale |

## Files

```
train.py          # training loop, optimizer, data loading
model.py          # DiffuMamba3 architecture + MDLM diffusion
ssm.py            # PureSSM: pure-PyTorch selective scan (no custom kernels)
autoresearch.py   # automated experiment runner
data/
  get_data.py     # download pre-tokenized shards from HF Hub
  get_data.sh     # curl-based alternative
  tokenize.py     # tokenize raw parquet -> .bin shards
ref/              # reference implementations (muon, modded-nanogpt)
notes/            # research notes, paper summaries
```

## Key Findings

- **Min-SNR gamma=5** loss weighting consistently beats standard ELBO (HIGH confidence)
- **Flat ELBO weight + Muon + torch.compile** gave best val_loss in short runs
- See `HANDOFF.md` for full training recipe and confidence levels

## Hardware

AMD RX 9070 XT (16GB VRAM, RDNA 4, gfx1201) via ROCm.
See `setup_rocm.sh` for environment setup. On WSL2, use the `librocprofiler_stub.so`
workaround (see `ROCPROFILER_WSL2_BUG.md` in `~/rocm-libraries/`).

## References

- [MDLM](https://arxiv.org/abs/2406.07524) | [DiffuMamba](https://arxiv.org/abs/2511.15927) | [Mamba-3](https://arxiv.org/abs/2603.15569)
- [Muon](https://kellerjordan.github.io/posts/muon/) | [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)
- [LLaDA](https://arxiv.org/abs/2502.09992) | [Min-SNR](https://arxiv.org/abs/2303.09556)
