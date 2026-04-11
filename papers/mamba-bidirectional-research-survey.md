# Mamba & Bidirectional Mamba Variants: Research Survey

> Compiled: 2026-04-11

---

## Table of Contents

1. [Mamba (Original)](#1-mamba-original)
2. [Mamba-2 (SSD Framework)](#2-mamba-2-ssd-framework)
3. [NVIDIA Empirical Study of Mamba-based LMs](#3-nvidia-empirical-study-of-mamba-based-lms)
4. [Vision Mamba (Vim) -- Bidirectional Scanning](#4-vision-mamba-vim----bidirectional-scanning)
5. [Caduceus -- BiMamba for DNA Sequences](#5-caduceus----bimamba-for-dna-sequences)
6. [BiGS -- Pretraining Without Attention (SSM + MLM)](#6-bigs----pretraining-without-attention-ssm--mlm)
7. [MaBERT -- Hybrid Mamba-Transformer Encoder for MLM](#7-mabert----hybrid-mamba-transformer-encoder-for-mlm)
8. [MDLM -- Masked Diffusion Language Models](#8-mdlm----masked-diffusion-language-models)
9. [DiffuMamba -- Diffusion LMs with Mamba Backbone](#9-diffumamba----diffusion-lms-with-mamba-backbone)
10. [DiM -- Diffusion Mamba for Image Synthesis](#10-dim----diffusion-mamba-for-image-synthesis)
11. [Summary Table](#11-summary-table)
12. [Key Takeaways for Bidirectional Mamba + Diffusion LM Design](#12-key-takeaways)

---

## 1. Mamba (Original)

- **Full Title:** Mamba: Linear-Time Sequence Modeling with Selective State Spaces
- **Authors:** Albert Gu, Tri Dao
- **Year:** 2023 (December)
- **ArXiv:** [2312.00752](https://arxiv.org/abs/2312.00752)
- **Venue:** ICLR 2024 (rejected, but widely adopted); later revised
- **Code:** [github.com/state-spaces/mamba](https://github.com/state-spaces/mamba)

### Abstract Summary

Mamba addresses the key weakness of prior structured state space models (SSMs): their inability to perform content-based reasoning due to input-independent parameters. The core innovation is making SSM parameters (specifically the discretization step delta, and the B, C matrices) functions of the input, enabling the model to selectively propagate or forget information along the sequence length dimension depending on the current token. This is called the **selective scan** mechanism. Mamba removes attention and MLP blocks entirely, using only selective SSM blocks with a hardware-aware parallel scan algorithm.

### Architecture

- Selective SSM block: input-dependent parameters (delta, B, C) computed via linear projections from the input
- No attention, no MLP -- pure SSM architecture
- Each block: linear projection -> conv1d -> selective SSM -> linear projection (with gating via SiLU)
- Hardware-aware implementation: parallel scan in SRAM, avoiding materializing large states in HBM

### Computational Complexity

- **O(N)** in sequence length (linear time) for both training and inference
- Training: parallel scan algorithm (work-efficient)
- Inference: constant-time per step via recurrent mode (no KV cache needed)
- 5x higher throughput than Transformers at inference

### Relevance to Diffusion/MLM

- Mamba is **unidirectional/causal** by design -- it processes sequences left-to-right
- Not directly applicable to masked language modeling or diffusion (which require bidirectional context)
- However, its linear complexity makes it an attractive backbone if bidirectionality can be added

### Key Results

- Mamba-3B outperforms Transformers of the same size and matches Transformers 2x its size on language modeling
- State-of-the-art on language, audio, and genomics benchmarks
- Scales to million-length sequences

---

## 2. Mamba-2 (SSD Framework)

- **Full Title:** Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality
- **Authors:** Tri Dao, Albert Gu
- **Year:** 2024 (May)
- **ArXiv:** [2405.21060](https://arxiv.org/abs/2405.21060)
- **Venue:** ICML 2024
- **Code:** [github.com/state-spaces/mamba](https://github.com/state-spaces/mamba) (same repo, v2 models included)

### Abstract Summary

This paper establishes theoretical connections between SSMs and variants of attention through the **State Space Duality (SSD) framework**. The key insight is that both SSMs and (structured) attention can be expressed as operations on structured semiseparable matrices. This unification allows deriving a new architecture, Mamba-2, whose core layer is a refinement of Mamba's selective SSM that is 2-8x faster while remaining competitive with Transformers.

### Architecture

- SSD layer: a restricted form of linear attention that is equivalent to a particular SSM
- Larger head dimension (matching multihead attention structure) compared to Mamba-1
- More hardware-friendly: can leverage matrix multiply units (tensor cores) rather than just scan operations
- Supports chunked computation: sequences split into chunks, intra-chunk via matrix multiply, inter-chunk via recurrence

### Computational Complexity

- **O(N)** in sequence length (same as Mamba-1)
- 2-8x faster than Mamba-1 in wall-clock time due to better hardware utilization
- The SSD algorithm uses a chunk-wise approach: O(chunk_size^2) matmuls within chunks + O(N/chunk_size) recurrent steps between chunks

### Relevance to Diffusion/MLM

- Still fundamentally **causal/unidirectional** in the standard formulation
- However, the SSD framework's connection to attention makes it conceptually easier to design bidirectional variants
- DiffuMamba (see below) uses bidirectional Mamba-2 mixers for diffusion LMs

---

## 3. NVIDIA Empirical Study of Mamba-based LMs

- **Full Title:** An Empirical Study of Mamba-based Language Models
- **Authors:** Roger Waleffe, Wonmin Byeon, Duncan Riach, Brandon Norick, Vijay Korthikanti, Tri Dao, Albert Gu, Ali Hatamizadeh, Sudhakar Singh, Deepak Narayanan, Garvit Kulshreshtha, Vartika Singh, Jared Casper, Jan Kautz, Mohammad Shoeybi, Bryan Catanzaro
- **Year:** 2024 (June)
- **ArXiv:** [2406.07887](https://arxiv.org/abs/2406.07887)
- **Code/Models:** Released via NVIDIA Megatron-LM; pretrained models on [HuggingFace](https://huggingface.co/nvidia/mamba2-hybrid-8b-3t-32k)

### Abstract Summary

The first large-scale (8B parameter, 3.5T tokens) controlled comparison of Mamba, Mamba-2, and Transformer models. Also introduces **Mamba-2-Hybrid**: 43% Mamba-2 layers, 7% self-attention layers, 50% MLP layers.

### Key Findings

- Pure Mamba/Mamba-2 match Transformers on many tasks but lag on copying/in-context learning tasks (e.g., 5-shot MMLU, phonebook lookup)
- **Mamba-2-Hybrid exceeds Transformers** on all 12 standard benchmarks (+2.65 points average)
- Hybrid is predicted to be up to 8x faster at token generation
- Long-context (up to 128K): hybrid matches or exceeds Transformer on 23 long-context tasks

### Relevance

- Demonstrates that a small amount of attention (7%) combined with Mamba-2 yields the best of both worlds
- This hybrid strategy is directly relevant to designing bidirectional Mamba architectures: interleaving a few attention layers with Mamba layers

---

## 4. Vision Mamba (Vim) -- Bidirectional Scanning

- **Full Title:** Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model
- **Authors:** Lianghui Zhu, Bencheng Liao, Qian Zhang, Xinlong Wang, Wenyu Liu, Xinggang Wang
- **Year:** 2024 (January)
- **ArXiv:** [2401.09417](https://arxiv.org/abs/2401.09417)
- **Venue:** ICML 2024
- **Code:** [github.com/hustvl/Vim](https://github.com/hustvl/Vim)

### Abstract Summary

Vision Mamba (Vim) adapts Mamba for visual representation learning by introducing **bidirectional Mamba blocks**. The approach flattens images into sequences with position embeddings and processes them through bidirectional state space models. Vim achieves superior performance to DeiT (Vision Transformer) on ImageNet, COCO, and ADE20k while being 2.8x faster and saving 86.8% GPU memory on high-resolution inference.

### How Bidirectionality is Achieved

- **Two-pass forward/backward scanning:** Stacked blocks are paired where the first block processes the visual sequence in the forward direction and the second block processes in the backward direction
- Each Vim block contains a standard Mamba SSM but processes flattened visual sequences with simultaneous forward and backward SSMs to capture spatial context from both directions
- The forward and backward SSM outputs are combined (typically via addition or concatenation) before the output projection

### Computational Complexity

- **O(N)** in the number of image patches (linear in sequence length)
- Significant memory savings compared to ViT due to no quadratic attention

### Relevance to Diffusion/MLM

- Demonstrates that bidirectional Mamba is effective for non-causal tasks (image understanding)
- The two-pass scanning approach is a general strategy applicable to any domain needing bidirectional context
- Directly relevant to adapting Mamba for masked language modeling or diffusion

---

## 5. Caduceus -- BiMamba for DNA Sequences

- **Full Title:** Caduceus: Bi-Directional Equivariant Long-Range DNA Sequence Modeling
- **Authors:** Yair Schiff, Chia-Hsiang Kao, Aaron Gokaslan, Tri Dao, Albert Gu, Volodymyr Kuleshov
- **Year:** 2024 (March)
- **ArXiv:** [2403.03234](https://arxiv.org/abs/2403.03234)
- **Venue:** ICML 2024
- **Code:** [github.com/kuleshov-group/caduceus](https://github.com/kuleshov-group/caduceus)
- **Models:** [HuggingFace](https://huggingface.co/kuleshov-group)

### Abstract Summary

Caduceus is the first family of reverse-complement (RC) equivariant bidirectional long-range DNA language models. It introduces **BiMamba**, a parameter-efficient bidirectional extension of Mamba, and **MambaDNA**, which adds reverse complement equivariance. The models outperform larger unidirectional and Transformer-based models on variant effect prediction tasks.

### How Bidirectionality is Achieved

- **BiMamba:** Runs a Mamba module on both the original sequence and its reverse, with **in and out projection weights tied** between forward and backward passes
- This is parameter-efficient: no additional parameters compared to unidirectional Mamba (weight sharing)
- The forward and backward hidden states are combined before the output gate
- **MambaDNA** extends BiMamba with RC equivariance (specific to DNA: enforcing that a sequence and its reverse complement produce related representations)

### Computational Complexity

- **O(N)** -- two passes of the linear-time selective scan
- Memory-efficient: weight tying means no additional parameters over unidirectional Mamba
- Roughly 2x compute of unidirectional Mamba (two passes)

### Relevance to Diffusion/MLM

- **Directly relevant:** BiMamba is designed for non-causal sequence modeling (DNA does not have a natural left-to-right order)
- The weight-tying approach for bidirectionality is the most parameter-efficient strategy found in the literature
- The same authors (Schiff, Gokaslan, Kuleshov) also created MDLM (see below), suggesting a natural path from BiMamba to diffusion LMs

---

## 6. BiGS -- Pretraining Without Attention (SSM + MLM)

- **Full Title:** Pretraining Without Attention
- **Authors:** Junxiong Wang, Jing Nathan Yan, Albert Gu, Alexander M. Rush
- **Year:** 2022 (December)
- **ArXiv:** [2212.10544](https://arxiv.org/abs/2212.10544)
- **Code:** [github.com/jxiw/BiGS](https://github.com/jxiw/BiGS)
- **Models:** [HuggingFace](https://huggingface.co/JunxiongWang/BiGS_128)

### Abstract Summary

BiGS (Bidirectional Gated SSM) demonstrates that Transformer attention is not essential for BERT-style pretraining. BiGS combines SSM layers with a multiplicative gating architecture and achieves BERT-level performance on the GLUE benchmark -- the **first SSM-based model to do so** -- with subquadratic complexity.

### How Bidirectionality is Achieved

- **Two SSM layers:** one processing the sequence forward, one backward
- **Multiplicative gating:** the forward and backward SSM outputs are combined via element-wise multiplication (gating), inspired by gated linear units (GLU)
- This gating mechanism allows the model to selectively combine information from both directions
- Uses S4 (structured state space) as the base SSM layer (predates Mamba's selective scan)

### Computational Complexity

- **O(N log N)** using the S4 kernel (FFT-based convolution)
- Subquadratic in sequence length
- Supports long-form pretraining with 4096 tokens without approximation (unlike BERT's 512-token limit)

### Relevance to Diffusion/MLM

- **Foundational paper** for SSM-based masked language modeling
- Proves that bidirectional SSMs can match BERT on MLM pretraining and downstream GLUE tasks
- Directly establishes the feasibility of replacing Transformer attention with SSMs for encoder-style (bidirectional) language models
- Predates Mamba; using Mamba's selective scan instead of S4 would likely yield further improvements

---

## 7. MaBERT -- Hybrid Mamba-Transformer Encoder for MLM

- **Full Title:** MaBERT: A Padding Safe Interleaved Transformer Mamba Hybrid Encoder for Efficient Extended Context Masked Language Modeling
- **Authors:** Jinwoong Kim, Sangjin Park
- **Year:** 2026 (March)
- **ArXiv:** [2603.03001](https://arxiv.org/abs/2603.03001)
- **Code:** Not yet released (as of April 2026)

### Abstract Summary

MaBERT interleaves Transformer layers (for global dependency modeling) with Mamba layers (for linear-time state updates) in an encoder architecture trained with masked language modeling. Achieves strong GLUE results with 2.36x faster training and 2.43x faster inference when extending context from 512 to 4096 tokens.

### How Bidirectionality is Achieved

- **Interleaved hybrid:** Transformer layers provide full bidirectional attention, Mamba layers process in one direction
- **Padding-safe masking:** blocks state propagation through padded positions in Mamba layers (critical for variable-length batching in encoder models)
- **Mask-aware attention pooling:** aggregates information only from valid (non-padded) tokens

### Computational Complexity

- Subquadratic overall (fewer attention layers than standard BERT)
- Linear scaling from the Mamba layers; quadratic only in the interleaved attention layers
- Significant practical speedups at longer contexts (2.36-2.43x)

### Relevance to Diffusion/MLM

- **Directly targets MLM** as its pretraining objective
- Shows how to handle practical challenges of using Mamba in encoder/bidirectional settings (padding, masking)
- The hybrid approach (some attention + mostly Mamba) mirrors NVIDIA's findings that a small fraction of attention greatly helps

---

## 8. MDLM -- Masked Diffusion Language Models

- **Full Title:** Simple and Effective Masked Diffusion Language Models
- **Authors:** Subham Sekhar Sahoo, Marianne Arriola, Yair Schiff, Aaron Gokaslan, Edgar Marroquin, Justin T. Chiu, Alexander Rush, Volodymyr Kuleshov
- **Year:** 2024 (June)
- **ArXiv:** [2406.07524](https://arxiv.org/abs/2406.07524)
- **Venue:** NeurIPS 2024
- **Code:** [github.com/kuleshov-group/mdlm](https://github.com/kuleshov-group/mdlm)

### Abstract Summary

MDLM demonstrates that masked discrete diffusion performs much better than previously thought. The key contribution is a simplified, Rao-Blackwellized training objective that reduces to a **mixture of classical masked language modeling (MLM) losses**. This enables training encoder-only diffusion language models that support efficient sampling and semi-autoregressive generation.

### Architecture

- Uses a **Transformer encoder** backbone (not Mamba) in the original paper
- The diffusion process: tokens are progressively masked over time, and the model learns to unmask them
- The loss is a weighted mixture of standard MLM losses at different masking rates

### Computational Complexity

- Depends on backbone (Transformer: O(N^2), but could be swapped for Mamba: O(N))
- Sampling requires multiple denoising steps (typically 100-1000)

### Relevance to Diffusion/MLM

- **Core paper** for the diffusion LM approach that DiffuMamba builds on
- Achieves state-of-the-art perplexity among diffusion models on LM1B and OpenWebText
- The connection between masked diffusion and MLM is foundational: masked diffusion IS a generalization of MLM
- The same group (Kuleshov lab) created both Caduceus/BiMamba and MDLM -- DiffuMamba (below) is the natural synthesis
- Backbone is swappable: the original uses Transformer, DiffuMamba replaces it with bidirectional Mamba

---

## 9. DiffuMamba -- Diffusion LMs with Mamba Backbone

- **Full Title:** DiffuMamba: High-Throughput Diffusion LMs with Mamba Backbone
- **Authors:** Vaibhav Singh, Oleksiy Ostapenko, Pierre-Andre Noel, Eugene Belilovsky, Torsten Scholak
- **Year:** 2025 (November, arxiv; revised 2026)
- **ArXiv:** [2511.15927](https://arxiv.org/abs/2511.15927)
- **Code:** Not publicly released as of April 2026

### Abstract Summary

DiffuMamba replaces the Transformer backbone in masked diffusion language models with a **bidirectional Mamba/Mamba-2 backbone**. It also introduces DiffuMamba-H, a hybrid with interleaved attention. At scales up to 1.3B parameters, DiffuMamba matches Transformer-based diffusion LM performance while achieving up to **8.2x higher inference throughput** on long sequences.

### How Bidirectionality is Achieved

- **Bidirectional Mamba-2 mixers:** masked diffusion requires conditioning on both past and future context at each denoising step, so the standard causal Mamba is replaced with a bidirectional variant
- Likely uses the two-pass approach (forward + backward scan) similar to BiMamba/Vim
- By removing quadratic attention, per-step latency and memory pressure are reduced without altering the probabilistic semantics of masked diffusion
- **DiffuMamba-H** interleaves some attention layers for tasks where full bidirectional attention is beneficial

### Computational Complexity

- **O(N)** per denoising step (linear in sequence length)
- **Cache-efficient block diffusion with Mamba mixers** is identified as the only strategy that scales linearly with sequence length
- 8.2x throughput improvement over Transformers on full-sequence denoising
- 4.3x throughput improvement on the hybrid variant

### Relevance to Diffusion/MLM

- **The most directly relevant paper:** combines bidirectional Mamba with masked diffusion for language modeling
- Demonstrates that Mamba can serve as a drop-in replacement for Transformers in diffusion LMs
- The linear scaling makes it practical for long-sequence diffusion (where Transformer-based diffusion is prohibitively expensive)
- Validates the architecture path: BiMamba + MDLM-style training = efficient diffusion LM

---

## 10. DiM -- Diffusion Mamba for Image Synthesis

- **Full Title:** DiM: Diffusion Mamba for Efficient High-Resolution Image Synthesis
- **Authors:** Yao Teng, Yue Wu, Han Shi, Xuefei Ning, Guohao Dai, Yu Wang, Zhenguo Li, Xihui Liu
- **Year:** 2024 (May)
- **ArXiv:** [2405.14224](https://arxiv.org/abs/2405.14224)
- **Code:** [github.com/tyshiwo1/DiM-DiffusionMamba](https://github.com/tyshiwo1/DiM-DiffusionMamba)

### Abstract Summary

DiM replaces the Vision Transformer (DiT) backbone in image diffusion models with Mamba. To handle 2D image data, the authors introduce multi-directional scanning, learnable padding tokens, and lightweight local feature enhancement. Uses a weak-to-strong training strategy (pretrain at 256x256, fine-tune at 512x512, training-free upsampling to 1536x1536).

### How Bidirectionality/Multi-directionality is Achieved

- **Multi-directional scans:** processes the image in multiple scan directions (e.g., left-to-right, right-to-left, top-to-bottom, bottom-to-top) to capture spatial context from all directions
- **Learnable padding tokens** at row and column endpoints to handle 2D structure
- **Lightweight local feature enhancement** to compensate for Mamba's sequential inductive bias

### Computational Complexity

- **O(N)** in the number of image patches (linear, vs. O(N^2) for DiT)
- Enables efficient high-resolution synthesis where Transformers struggle

### Relevance to Diffusion/MLM

- Demonstrates that Mamba can serve as the diffusion model backbone for image generation
- The multi-directional scanning approach for 2D is analogous to bidirectional scanning for 1D text
- Validates Mamba's suitability for the iterative denoising process in diffusion models

---

## 11. Summary Table

| Paper | Year | ArXiv | Bidirectional? | How? | Complexity | Domain | Code |
|-------|------|-------|---------------|------|------------|--------|------|
| **Mamba** | 2023 | [2312.00752](https://arxiv.org/abs/2312.00752) | No (causal) | N/A | O(N) | Language, Audio, Genomics | [state-spaces/mamba](https://github.com/state-spaces/mamba) |
| **Mamba-2** | 2024 | [2405.21060](https://arxiv.org/abs/2405.21060) | No (causal) | N/A | O(N), 2-8x faster | Language | [state-spaces/mamba](https://github.com/state-spaces/mamba) |
| **NVIDIA Empirical** | 2024 | [2406.07887](https://arxiv.org/abs/2406.07887) | Hybrid | 7% attention + 43% Mamba-2 + 50% MLP | O(N) dominant | Language (8B scale) | NVIDIA Megatron-LM |
| **Vision Mamba** | 2024 | [2401.09417](https://arxiv.org/abs/2401.09417) | Yes | Forward+backward scan (paired blocks) | O(N) | Vision | [hustvl/Vim](https://github.com/hustvl/Vim) |
| **Caduceus/BiMamba** | 2024 | [2403.03234](https://arxiv.org/abs/2403.03234) | Yes | Forward+reverse scan, weight-tied | O(N) (2x cost) | DNA/Genomics | [kuleshov-group/caduceus](https://github.com/kuleshov-group/caduceus) |
| **BiGS** | 2022 | [2212.10544](https://arxiv.org/abs/2212.10544) | Yes | Two SSMs + multiplicative gating | O(N log N) | Language (MLM/GLUE) | [jxiw/BiGS](https://github.com/jxiw/BiGS) |
| **MaBERT** | 2026 | [2603.03001](https://arxiv.org/abs/2603.03001) | Hybrid | Interleaved Transformer + Mamba | Subquadratic | Language (MLM/GLUE) | Not released |
| **MDLM** | 2024 | [2406.07524](https://arxiv.org/abs/2406.07524) | Yes (encoder) | Transformer encoder backbone | O(N^2) backbone | Language (diffusion) | [kuleshov-group/mdlm](https://github.com/kuleshov-group/mdlm) |
| **DiffuMamba** | 2025 | [2511.15927](https://arxiv.org/abs/2511.15927) | Yes | Bidirectional Mamba-2 mixers | O(N) | Language (diffusion) | Not released |
| **DiM** | 2024 | [2405.14224](https://arxiv.org/abs/2405.14224) | Multi-dir | Multi-directional scanning (4 dirs) | O(N) | Image (diffusion) | [tyshiwo1/DiM-DiffusionMamba](https://github.com/tyshiwo1/DiM-DiffusionMamba) |

---

## 12. Key Takeaways for Bidirectional Mamba + Diffusion LM Design

### Strategies for Making Mamba Bidirectional

1. **Two-pass scanning (BiMamba/Caduceus style):** Run the SSM forward and backward on the sequence; combine outputs. Weight-tying keeps parameters equal to unidirectional. Compute cost is 2x. This is the **simplest and most parameter-efficient** approach.

2. **Paired block scanning (Vision Mamba style):** Alternate blocks process forward and backward. Each block is standard Mamba but direction alternates. Information propagates bidirectionally through the depth of the network.

3. **Hybrid with attention (NVIDIA/MaBERT style):** Use mostly Mamba layers with a few interleaved attention layers. The attention layers provide true bidirectional context. Even 7% attention layers significantly boost performance. Practical advantage: attention handles the tasks Mamba struggles with (copying, in-context learning).

4. **Multiplicative gating (BiGS style):** Combine forward and backward SSM representations via element-wise multiplication rather than addition. Potentially more expressive combination but less studied with Mamba (BiGS uses S4).

### The Path from Mamba to Diffusion LMs

The research trajectory is clear:
- **MDLM** established that masked diffusion = mixture of MLM losses, enabling encoder-only diffusion LMs
- **BiMamba/Caduceus** showed how to make Mamba bidirectional efficiently
- **DiffuMamba** combined these: bidirectional Mamba-2 as the backbone for MDLM-style masked diffusion
- Result: **O(N) per denoising step** instead of O(N^2), with 8.2x throughput gains

### Practical Recommendations

- For a **bidirectional Mamba diffusion LM**, the DiffuMamba approach (bidirectional Mamba-2 mixers + masked diffusion training) is the current state of the art
- Consider the **hybrid approach** (DiffuMamba-H) with a small fraction of attention layers for better quality
- For training, use the **MDLM objective** (Rao-Blackwellized mixture of MLM losses) which is simpler and more effective than earlier diffusion LM objectives
- The **causal convolution** in standard Mamba blocks should be replaced with **non-causal (standard) convolution** when used bidirectionally (noted in MaskMamba for images)
- **Padding-safe masking** (from MaBERT) is important for practical encoder training with variable-length sequences
