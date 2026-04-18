"""Cloze-style probing for MDLM models.

Tests:
1. Fill-in-the-blank: mask specific tokens, check top predictions
2. Iterative demasking: start from all [MASK], progressively unmask to generate text
"""

import sys
import os
import torch
import torch.nn.functional as F
import tiktoken

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from gpt2 import GPT2
from transformer_v2 import TransformerV2
from hybrid_model import DiffuMambaH

MASK_TOKEN = 50257
enc = tiktoken.get_encoding('gpt2')


def load_model(name, ckpt_path, device):
    """Load a model from checkpoint."""
    configs = {
        'd_modern': lambda: TransformerV2(
            vocab_size=50258, n_layer=6, n_head=6, n_embd=384,
            use_rope=True, use_swiglu=True),
        'mamba3': lambda: DiffuMambaH(
            n_embd=384, n_head=6, n_mamba=6, n_attn=0,
            attn_positions=set(), mamba_version=3, d_state=32),
        'transformer_v2': lambda: GPT2(
            vocab_size=50258, n_layer=6, n_head=6, n_embd=384),
    }
    model = configs[name]().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f'Loaded {name} from {ckpt_path} ({model.param_count()/1e6:.1f}M params)')
    del ckpt
    torch.cuda.empty_cache()
    return model


def cloze_probe(model, text, mask_words, device, top_k=5):
    """Mask specific words in text and show model predictions."""
    tokens = enc.encode(text)
    token_strs = [enc.decode([t]) for t in tokens]

    # Find positions to mask
    mask_positions = []
    for i, ts in enumerate(token_strs):
        for mw in mask_words:
            if mw.lower() in ts.lower():
                mask_positions.append(i)

    if not mask_positions:
        print(f'  Warning: none of {mask_words} found in tokens')
        return

    # Create masked input
    x = torch.tensor([tokens], device=device)
    x_masked = x.clone()
    for pos in mask_positions:
        x_masked[0, pos] = MASK_TOKEN

    # Show input
    masked_strs = []
    for i, ts in enumerate(token_strs):
        if i in mask_positions:
            masked_strs.append('[MASK]')
        else:
            masked_strs.append(ts)
    print(f'  Input:  {"".join(masked_strs)}')

    # Get predictions
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        logits = model(x_masked, causal=False)[..., :MASK_TOKEN]

    for pos in mask_positions:
        probs = F.softmax(logits[0, pos].float(), dim=-1)
        top_probs, top_ids = probs.topk(top_k)
        predictions = [(enc.decode([tid.item()]), tp.item()) for tid, tp in zip(top_ids, top_probs)]
        correct = token_strs[pos]
        rank = (probs.argsort(descending=True) == tokens[pos]).nonzero()
        rank = rank.item() + 1 if len(rank) > 0 else '>50257'
        print(f'  Pos {pos:3d} (correct: "{correct}", rank {rank}):')
        for pred_str, pred_p in predictions:
            marker = ' <--' if pred_str.strip() == correct.strip() else ''
            print(f'    {pred_p:6.2%} "{pred_str}"{marker}')


def iterative_demask(model, device, seq_len=64, n_steps=50):
    """Generate text by iterative demasking from all [MASK]."""
    x = torch.full((1, seq_len), MASK_TOKEN, device=device, dtype=torch.long)
    n_unmask_per_step = max(1, seq_len // n_steps)

    print(f'  Generating {seq_len} tokens in {n_steps} demasking steps...')

    for step in range(n_steps):
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(x, causal=False)[..., :MASK_TOKEN]

        # Find positions still masked
        masked_pos = (x[0] == MASK_TOKEN).nonzero(as_tuple=True)[0]
        if len(masked_pos) == 0:
            break

        # Get confidence at masked positions
        probs = F.softmax(logits[0, masked_pos].float(), dim=-1)
        max_probs, best_tokens = probs.max(dim=-1)

        # Unmask the most confident positions
        n_to_unmask = min(n_unmask_per_step, len(masked_pos))
        _, conf_order = max_probs.sort(descending=True)
        to_unmask = conf_order[:n_to_unmask]

        for idx in to_unmask:
            pos = masked_pos[idx]
            x[0, pos] = best_tokens[idx]

    text = enc.decode(x[0].tolist())
    return text


def main():
    device = 'cuda'
    torch.set_float32_matmul_precision('high')

    models_to_test = {
        'd_modern': 'muon_exp/outputs/transformer_converge_v3/checkpoint_56000.pt',
        'mamba3': 'muon_exp/outputs/mamba3_converge/checkpoint_56000.pt',
    }

    # Cloze tests
    cloze_tests = [
        ("The capital of France is Paris.", ["Paris"]),
        ("The capital of France is Paris.", ["France"]),
        ("Water freezes at zero degrees Celsius.", ["zero", "Celsius"]),
        ("The president of the United States lives in the White House.", ["White", "House"]),
        ("Machine learning models are trained using gradient descent.", ["gradient", "descent"]),
        ("The quick brown fox jumps over the lazy dog.", ["fox", "lazy"]),
        ("In 1969, Neil Armstrong walked on the moon.", ["Armstrong", "moon"]),
        ("Python is a popular programming language for data science.", ["Python", "programming"]),
    ]

    for model_name, ckpt_path in models_to_test.items():
        if not os.path.exists(ckpt_path):
            print(f'\nSkipping {model_name} (checkpoint not found)')
            continue

        model = load_model(model_name, ckpt_path, device)

        print(f'\n{"="*60}')
        print(f'  CLOZE PROBES: {model_name}')
        print(f'{"="*60}')

        for text, mask_words in cloze_tests:
            print()
            cloze_probe(model, text, mask_words, device)

        print(f'\n{"="*60}')
        print(f'  GENERATION: {model_name}')
        print(f'{"="*60}')

        for length in [32, 64, 128]:
            print(f'\n  --- {length} tokens ---')
            text = iterative_demask(model, device, seq_len=length, n_steps=length)
            print(f'  {text}')

        del model
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
