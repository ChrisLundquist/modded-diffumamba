"""Compute generative perplexity: sample from our model, score under GPT-2.

Standard metric in MDLM papers (Dream, LLaDA, Scaling Beyond MDLM).
Generates unconditional samples from each checkpoint, then scores them
under frozen GPT-2 small to get a reference-grounded quality measure.

Also reports: unigram entropy, repetition-4, distinct-4 for diagnostics.
"""
import sys
import copy
import math
from pathlib import Path
from collections import Counter

import torch
import torch.nn.functional as F
import tiktoken
from transformers import GPT2LMHeadModel

sys.path.insert(0, '.')
from model import DiffuMamba3, CONFIGS

CKPT_DIR = Path("checkpoints")
device = "cuda"
MASK_ID = 50257
enc = tiktoken.get_encoding("gpt2")

# Load reference LM once (GPT-2 small = 124M, fits easily alongside our 111M)
print("Loading GPT-2 small as reference LM...")
ref = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
for p in ref.parameters():
    p.requires_grad = False
print(f"  GPT-2 params: {sum(p.numel() for p in ref.parameters())/1e6:.1f}M")


@torch.no_grad()
def score_gen_ppl(token_ids, ref_model, batch_size=4):
    """Compute per-token NLL of a batch of sequences under GPT-2. Returns PPL."""
    # token_ids: (B, L), long, in GPT-2 vocab range
    nlls = []
    n_tokens = 0
    for i in range(0, token_ids.shape[0], batch_size):
        batch = token_ids[i:i+batch_size].to(device)
        outputs = ref_model(batch, labels=batch)
        # loss is mean NLL over non-pad tokens in the batch
        # multiply by tokens to get sum NLL
        b, L = batch.shape
        tokens_in_batch = b * (L - 1)  # GPT-2 shifts by 1
        nlls.append(outputs.loss.item() * tokens_in_batch)
        n_tokens += tokens_in_batch
    avg_nll = sum(nlls) / n_tokens
    return math.exp(avg_nll), avg_nll


def diagnostics(token_ids, vocab_size=50257):
    """Compute unigram entropy, distinct-4, repetition-4 on a batch of ids."""
    import math
    flat = token_ids.flatten().tolist()
    # Filter mask / OOV
    flat = [t for t in flat if t < vocab_size]
    if not flat:
        return {"unigram_H": 0, "distinct_4g": 0, "rep_4g": 0}

    # Unigram entropy
    counts = Counter(flat)
    total = len(flat)
    H = -sum((c/total) * math.log(c/total) for c in counts.values())

    # 4-gram diversity per sample
    distinct_4_list = []
    rep_4_list = []
    for row in token_ids:
        ids = [t.item() for t in row if t.item() < vocab_size]
        if len(ids) < 5:
            continue
        ngrams = [tuple(ids[i:i+4]) for i in range(len(ids)-3)]
        if not ngrams:
            continue
        distinct_4_list.append(len(set(ngrams)) / len(ngrams))
        counts_n = Counter(ngrams)
        rep_4_list.append(sum(c-1 for c in counts_n.values() if c > 1) / len(ngrams))

    return {
        "unigram_H": round(H, 3),
        "distinct_4g": round(sum(distinct_4_list)/len(distinct_4_list), 3) if distinct_4_list else 0,
        "rep_4g": round(sum(rep_4_list)/len(rep_4_list), 3) if rep_4_list else 0,
    }


def eval_checkpoint(ckpt_name, shape, n_samples=16, seq_len=128, num_steps=128,
                    temperature=1.0, top_p=1.0, top_k=0):
    """Sample from a checkpoint and score."""
    ckpt_path = CKPT_DIR / f"{ckpt_name}.pt"
    if not ckpt_path.exists():
        return None

    n_layers, d_model = shape
    cfg = copy.deepcopy(CONFIGS["quokka"])
    cfg.n_layers = n_layers
    cfg.d_model = d_model

    model = DiffuMamba3(cfg).to(device, dtype=torch.bfloat16)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    torch.manual_seed(0)
    samples = model.sample(batch_size=n_samples, seq_len=seq_len,
                           num_steps=num_steps, device=device,
                           temperature=temperature, top_p=top_p, top_k=top_k)

    # Clean to valid GPT-2 vocab range (GPT-2 has 50257 tokens)
    samples_clean = samples.clone()
    samples_clean[samples_clean >= 50257] = enc.encode(" ")[0]  # replace mask/OOV with space

    ppl, nll = score_gen_ppl(samples_clean, ref)
    diag = diagnostics(samples_clean)

    del model
    torch.cuda.empty_cache()

    return {
        "ckpt": ckpt_name,
        "gen_ppl": round(ppl, 1),
        "gen_nll": round(nll, 3),
        **diag,
        "sampler": f"T={temperature}, p={top_p}, k={top_k}",
    }


def main():
    checkpoints = [
        # Our best large model, progression through training
        ("10L640d_10k",                    (10, 640)),
        ("10L640d_50k_step10000",          (10, 640)),
        ("10L640d_50k_step30000",          (10, 640)),
        ("10L640d_50k",                    (10, 640)),
        # Quokka for comparison
        ("final10k_new_best_s42",          (4, 384)),
        ("final10k_old_best_s42",          (4, 384)),
        ("best10k_adam_s42",               (4, 384)),
    ]

    samplers = [
        {"temperature": 1.0, "top_p": 1.0, "top_k": 0},     # raw
        {"temperature": 1.0, "top_p": 1.0, "top_k": 50},    # top-k 50 (best from earlier)
        {"temperature": 0.9, "top_p": 0.9, "top_k": 0},     # moderate truncation
    ]

    print(f"\n{'='*100}")
    print(f"  {'Checkpoint':<28s} {'Sampler':<25s} {'gen-PPL':>8s} {'H1':>5s} {'dist4':>6s} {'rep4':>5s}")
    print(f"  {'-'*28} {'-'*25} {'-'*8} {'-'*5} {'-'*6} {'-'*5}")

    results = []
    for ckpt_name, shape in checkpoints:
        for sampler in samplers:
            r = eval_checkpoint(ckpt_name, shape, **sampler)
            if r is None:
                continue
            print(f"  {r['ckpt']:<28s} {r['sampler']:<25s} "
                  f"{r['gen_ppl']:>8.1f} {r['unigram_H']:>5.2f} "
                  f"{r['distinct_4g']:>6.3f} {r['rep_4g']:>5.3f}")
            results.append(r)

    # Save
    import json
    Path("results").mkdir(exist_ok=True)
    with open("results/gen_ppl.json", "w") as f:
        json.dump(results, f, indent=2)

    # Reference anchors from literature
    print(f"\n  Reference anchors (from MDLM/LLaDA papers):")
    print(f"    MDLM 169M (OWT, 1M steps):  gen-PPL ~82  (with top-p=0.9)")
    print(f"    LLaDA 1B (FineWeb, 30M+ steps): gen-PPL ~60-80")
    print(f"    Quokka 31M (~80M tokens):   our 5.07 ELBO → gen-PPL ~hundreds expected")
    print(f"    GPT-2 small on its own text: gen-PPL ~30 (lower bound for coherent English)")


if __name__ == "__main__":
    main()
