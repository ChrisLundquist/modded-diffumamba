"""PAPL fine-tune for 30M D_modern transformer (Exp 4 at smaller scale).

Mirrors finetune_papl_125m.py but matched to the 30M training recipe
(transformer_converge_v3): γ=5 Min-SNR, FineWeb-1B data, n_layer=6,
n_embd=384. Resume from any 30M checkpoint, sweep τ.

Vanilla comparison comes for free from the existing intermediate checkpoints
(checkpoint_50000.pt + 5k = checkpoint_55000.pt).
"""

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'eval', 'gen_harness'))
from transformer_v2 import TransformerV2  # noqa: E402
from muon import Muon  # noqa: E402
from data import TokenDataset  # noqa: E402
from samplers.mdlm_topk import demask_topk_prefix  # noqa: E402
from metrics.diversity import distinct_n  # noqa: E402
from metrics.repetition import rep_n, top_word_share  # noqa: E402

GENPROBE_PROMPTS_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'eval', 'gen_harness', 'prompts',
    'fineweb_edu_held.pt'
)

torch.set_float32_matmul_precision('high')

MASK_TOKEN = 50257
MICRO_BATCH = 16
GRAD_ACCUM = 8
SEQ_LEN = 1024
LOG_EVERY = 500
CHECKPOINT_EVERY = 5000

MUON_LR = 0.01
EMBED_LR = 1e-3
MIN_SNR_GAMMA = 5.0   # match transformer_converge_v3 recipe

DATA_PATH = '/home/clundquist/muon_data/fineweb_1B.npy'
DEFAULT_RESUME = '/mnt/d/code/gpt-slide/muon_exp/outputs/transformer_converge_v3/checkpoint_50000.pt'


def papl_mdlm_loss(model, x, alpha, tau, min_snr_gamma=MIN_SNR_GAMMA,
                   invert_planner=False):
    B, T = x.shape
    device = x.device
    t = torch.rand(B, 1, device=device) * 0.95 + 0.05
    k = 5.0
    alpha_t = 1.0 - torch.exp(-k * t)
    mask = torch.rand(B, T, device=device) < alpha_t
    mask_tokens = torch.full_like(x, MASK_TOKEN)
    x_t = torch.where(mask, mask_tokens, x)
    logits = model(x_t, causal=False)[..., :MASK_TOKEN]
    per_token_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), x.reshape(-1), reduction='none'
    ).reshape(B, T)

    if alpha > 0:
        with torch.no_grad():
            logprobs = F.log_softmax(logits, dim=-1)
            gt_logprob = logprobs.gather(-1, x.unsqueeze(-1)).squeeze(-1)
            # invert_planner=True flips the sign so HARD positions (low gt_logprob)
            # get high planner weight. Mechanistic control: tests whether PAPL's
            # specific direction matters or only that the loss is reweighted.
            score_input = -gt_logprob if invert_planner else gt_logprob
            scores = (score_input / tau).masked_fill(~mask, -1e4)
            w = F.softmax(scores, dim=1) * mask.float()
        papl_weight = 1.0 + alpha * w
    else:
        papl_weight = torch.ones_like(per_token_loss)

    mf = mask.float()
    weighted = per_token_loss * mf * papl_weight
    per_sample = weighted.sum(dim=1) / mf.sum(dim=1).clamp(min=1.0)
    elbo_weight = (k * torch.exp(-k * t) / (1.0 - torch.exp(-k * t))).clamp(max=min_snr_gamma)
    return (per_sample * elbo_weight.squeeze(1)).mean()


def eval_mdlm_decomp(model, val_x, alpha=1.0, tau=1.0):
    """Same as 125M decomp, but with γ=5 to match 30M training recipe."""
    model.eval()
    device = val_x.device
    n_val = val_x.shape[0]
    uniform_total = 0.0
    planner_total = 0.0
    n_points = 0
    k = 5.0
    eval_rng = torch.Generator(device=device); eval_rng.manual_seed(0)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for t_val in torch.linspace(0.05, 0.95, 20):
            uniform_b, planner_b = [], []
            for i in range(0, n_val, MICRO_BATCH):
                vx = val_x[i:i + MICRO_BATCH]
                bs = vx.shape[0]
                t = torch.full((bs, 1), t_val.item(), device=device)
                alpha_t = 1.0 - torch.exp(-k * t)
                mask = torch.rand(bs, vx.shape[1], device=device, generator=eval_rng) < alpha_t
                mask_tokens = torch.full_like(vx, MASK_TOKEN)
                x_t = torch.where(mask, mask_tokens, vx)
                logits = model(x_t, causal=False)[..., :MASK_TOKEN]
                per_token_loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), vx.reshape(-1), reduction='none'
                ).reshape(bs, -1)
                ew = (k * torch.exp(-k * t) / (1.0 - torch.exp(-k * t))).clamp(max=MIN_SNR_GAMMA)
                mf = mask.float()
                u_per = (per_token_loss * mf).sum(dim=1) / mf.sum(dim=1).clamp(min=1.0)
                uniform_b.append((u_per * ew.squeeze(1)).mean().item())
                logprobs = F.log_softmax(logits, dim=-1)
                gt_lp = logprobs.gather(-1, vx.unsqueeze(-1)).squeeze(-1)
                scores = (gt_lp / tau).masked_fill(~mask, -1e4)
                w = F.softmax(scores, dim=1) * mf
                pw = 1.0 + alpha * w
                p_per = (per_token_loss * mf * pw).sum(dim=1) / mf.sum(dim=1).clamp(min=1.0)
                planner_b.append((p_per * ew.squeeze(1)).mean().item())
            uniform_total += sum(uniform_b) / len(uniform_b)
            planner_total += sum(planner_b) / len(planner_b)
            n_points += 1
    return {
        'uniform_nll_minsnr': uniform_total / n_points,
        'planner_w_nll_minsnr': planner_total / n_points,
    }


@torch.no_grad()
def genprobe(model, prefix_ids, cont_len, prefix_len, top_k=50, n_steps=64,
             batch_size=16, device='cuda'):
    model.eval()
    samples = []
    for b in range(0, prefix_ids.shape[0], batch_size):
        chunk = prefix_ids[b:b + batch_size].to(device)
        out = demask_topk_prefix(model, chunk, cont_len, top_k=top_k,
                                 temperature=1.0, n_steps=n_steps, device=device)
        samples.append(out.cpu())
    full = torch.cat(samples, dim=0)
    completions = [row[prefix_len:].tolist() for row in full]
    return {
        'rep_2': rep_n(completions, 2), 'rep_4': rep_n(completions, 4),
        'rep_8': rep_n(completions, 8),
        'distinct_4': distinct_n(completions, 4),
        'top10_share': top_word_share(completions, 10),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--alpha', type=float, default=1.0)
    ap.add_argument('--tau', type=float, default=1.0)
    ap.add_argument('--extra-steps', type=int, default=5000,
                    help='Number of training steps. With --from-scratch this is total steps.')
    ap.add_argument('--resume', type=str, default=DEFAULT_RESUME)
    ap.add_argument('--from-scratch', action='store_true',
                    help='Train from random init instead of resuming from --resume. '
                         'Matches the recipe of transformer_converge_v3 for direct '
                         'comparison against checkpoint_5000.pt etc.')
    ap.add_argument('--output-dir', type=str, required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--genprobe-every', type=int, default=1000)
    ap.add_argument('--genprobe-n-prompts', type=int, default=64)
    ap.add_argument('--invert-planner', action='store_true',
                    help='Flip planner score sign — upweight HARD positions instead '
                         'of easy ones. Mechanistic control for PAPL direction.')
    args = ap.parse_args()

    device = 'cuda'
    os.makedirs(args.output_dir, exist_ok=True)

    print(f'Loading data (mmap)...')
    all_tokens = np.load(DATA_PATH, mmap_mode='r')
    train_tokens = all_tokens[:1_000_000_000]
    val_tokens = all_tokens[1_000_000_000:]

    torch.manual_seed(args.seed)
    model = TransformerV2(vocab_size=50258, n_layer=6, n_head=6, n_embd=384,
                          use_rope=True, use_swiglu=True).to(device)
    print(f'Params: {model.param_count()/1e6:.1f}M')

    muon_params, adamw_params = model.param_groups()
    adam_opt = torch.optim.Adam(
        [{'params': adamw_params, 'lr': EMBED_LR, 'betas': (0.9, 0.999)}])
    muon_opt = Muon([{'params': muon_params}], lr=MUON_LR, momentum=0.95, ns_steps=5)

    if args.from_scratch:
        print('From-scratch training (no resume).')
        start_step = 0
        baseline_val = None
    else:
        print(f'Resuming from {args.resume}...')
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        if 'adam_state_dict' in ckpt:
            adam_opt.load_state_dict(ckpt['adam_state_dict'])
            muon_opt.load_state_dict(ckpt['muon_state_dict'])
        elif 'optimizer_state_dict' in ckpt:
            print('  Note: ckpt has single optimizer_state_dict; starting opts fresh')
        start_step = ckpt['step']
        baseline_val = ckpt.get('val_loss', None)
        print(f'  resumed at step {start_step}, ckpt val_loss {baseline_val}')
        del ckpt
        torch.cuda.empty_cache()

    n_val = min(512, len(val_tokens) // SEQ_LEN)
    val_x = torch.from_numpy(
        val_tokens[:n_val * SEQ_LEN].astype(np.int64).reshape(n_val, SEQ_LEN)
    ).to(device)

    ds = TokenDataset(train_tokens, seq_len=SEQ_LEN)
    g = torch.Generator(); g.manual_seed(args.seed * 1000 + 7)
    loader = DataLoader(ds, batch_size=MICRO_BATCH, shuffle=True, num_workers=4,
                        pin_memory=True, drop_last=True, persistent_workers=True,
                        generator=g)
    data_iter = iter(loader)

    decomp0 = eval_mdlm_decomp(model, val_x, alpha=args.alpha, tau=args.tau)
    print(f'Baseline val: uniform {decomp0["uniform_nll_minsnr"]:.4f}  '
          f'papl_w {decomp0["planner_w_nll_minsnr"]:.4f}')

    prompts = torch.load(GENPROBE_PROMPTS_PATH, weights_only=False)
    probe_prefix = prompts['prefix_ids'][:args.genprobe_n_prompts]
    probe_cont_len = int(prompts['cont_len'])
    probe_prefix_len = int(prompts['prefix_len'])
    m0 = genprobe(model, probe_prefix, probe_cont_len, probe_prefix_len, device=device)
    print(f'Genprobe baseline (n={args.genprobe_n_prompts}): '
          f'rep_2={m0["rep_2"]:.4f} rep_4={m0["rep_4"]:.4f} '
          f'rep_8={m0["rep_8"]:.4f} dist_4={m0["distinct_4"]:.4f}')

    print(f'\nPAPL 30M fine-tune: alpha={args.alpha}, tau={args.tau}')
    print(f'Steps {start_step} → {start_step + args.extra_steps} (constant LR, γ={MIN_SNR_GAMMA})\n')

    for pg in adam_opt.param_groups: pg['lr'] = EMBED_LR
    for pg in muon_opt.param_groups: pg['lr'] = MUON_LR

    best_val = decomp0['uniform_nll_minsnr']
    t0 = time.time()
    tokens_per_step = MICRO_BATCH * GRAD_ACCUM * SEQ_LEN

    for local_step in range(args.extra_steps):
        step = start_step + local_step
        model.train()
        for opt in [adam_opt, muon_opt]: opt.zero_grad()
        total_loss = 0.0
        for _ in range(GRAD_ACCUM):
            try:
                x, _ = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x, _ = next(data_iter)
            x = x.to(device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = papl_mdlm_loss(model, x, alpha=args.alpha, tau=args.tau,
                                       invert_planner=args.invert_planner) / GRAD_ACCUM
            loss.backward()
            total_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(adamw_params, 1.0)
        for opt in [adam_opt, muon_opt]: opt.step()

        is_log = (local_step % LOG_EVERY == 0) or (local_step == args.extra_steps - 1)
        is_probe = (args.genprobe_every > 0 and
                    ((local_step + 1) % args.genprobe_every == 0 or
                     local_step == args.extra_steps - 1))
        if is_log or is_probe:
            decomp = eval_mdlm_decomp(model, val_x, alpha=args.alpha, tau=args.tau)
            vloss = decomp['uniform_nll_minsnr']
            vloss_papl = decomp['planner_w_nll_minsnr']
            elapsed = time.time() - t0
            tps = (local_step + 1) * tokens_per_step / elapsed if elapsed > 0 else 0
            if vloss < best_val: best_val = vloss
            extra = ''
            if is_probe:
                m = genprobe(model, probe_prefix, probe_cont_len, probe_prefix_len,
                             device=device)
                extra = (f' | rep2={m["rep_2"]:.3f} rep4={m["rep_4"]:.3f} '
                         f'rep8={m["rep_8"]:.3f} dist4={m["distinct_4"]:.3f}')
            print(f'  step {step:6d} (+{local_step+1:5d}) | train {total_loss:.4f} | '
                  f'val_uniform {vloss:.4f} | val_papl {vloss_papl:.4f} | '
                  f'best_uniform {best_val:.4f} | {tps/1e3:.0f}K tok/s | '
                  f'{elapsed:.0f}s{extra}', flush=True)

        if (local_step + 1) % CHECKPOINT_EVERY == 0 or local_step == args.extra_steps - 1:
            ckpt_path = os.path.join(args.output_dir, f'checkpoint_{step+1}.pt')
            torch.save({
                'step': step + 1,
                'model_state_dict': model.state_dict(),
                'adam_state_dict': adam_opt.state_dict(),
                'muon_state_dict': muon_opt.state_dict(),
                'val_loss': vloss if is_log else best_val,
                'papl_alpha': args.alpha, 'papl_tau': args.tau,
                'resumed_from': args.resume, 'seed': args.seed,
            }, ckpt_path)
            print(f'  Checkpoint: {ckpt_path}', flush=True)

    print(f'\nDone. Best val (γ={MIN_SNR_GAMMA}): {best_val:.4f}  '
          f'(baseline at start: {decomp0["uniform_nll_minsnr"]:.4f})')
    print(f'Total time: {(time.time()-t0)/3600:.2f} hours')


if __name__ == '__main__':
    main()
