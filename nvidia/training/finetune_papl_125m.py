"""Fine-tune 125M D_modern for 10k steps with PAPL loss (Peng et al. 2025).

Signal test: resume from checkpoint_40000.pt (diversity peak under std MDLM
training — distinct_4=0.87, rep_4=0.131) and train 10k more steps with the
planner-aware path-learning loss instead of vanilla MinSNR-ELBO.

Hypothesis: std MDLM at step 50k regresses on generation metrics
(distinct_4 0.87→0.85, rep_4 0.13→0.16) while ELBO keeps dropping.
If PAPL fixes the train/sampler mismatch, the fine-tuned model at equivalent
step 50k should retain or improve on the 40k generation metrics.

PAPL loss: L = − Σ_i (1/M) · (1 + α · w_i) · log p(x_0^i | x_t)
where w_i = softmax_{masked positions}(max_logprob_i / τ)
— i.e., reweight each masked position by the self-planner's probability
of unmasking it next.

α=0 recovers vanilla MDLM. Paper defaults α=1.0, τ=1.0. We match Min-SNR γ=1.5
from the original recipe so the only change is the per-position reweighting.
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
MIN_SNR_GAMMA = 1.5

N_LAYER = 12
N_EMBD = 768
N_HEAD = 12
VOCAB_SIZE = 50258

DATA_PATH = '/home/clundquist/muon_data/fineweb_10B.npy'
RESUME_CKPT = '/mnt/d/code/gpt-slide/muon_exp/outputs/125m_10b_dmodern/checkpoint_40000.pt'
OUTPUT_DIR = '/mnt/d/code/gpt-slide/muon_exp/outputs/125m_papl_finetune'


def papl_mdlm_loss(model, x, alpha, tau, min_snr_gamma=MIN_SNR_GAMMA):
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
        # PAPL self-planner weight (Peng 2025 §3.4):
        #   w_i = softmax_{masked positions}( log p(x_0^i | x_t) / tau )
        # Planner concentration (entropy, top-1 share) measured separately by
        # papl_diagnostics.py to avoid per-microbatch overhead.
        with torch.no_grad():
            logprobs = F.log_softmax(logits, dim=-1)
            gt_logprob = logprobs.gather(-1, x.unsqueeze(-1)).squeeze(-1)  # [B, T]
            scores = gt_logprob / tau
            scores = scores.masked_fill(~mask, -1e4)
            w = F.softmax(scores, dim=1) * mask.float()
        papl_weight = 1.0 + alpha * w
    else:
        papl_weight = torch.ones_like(per_token_loss)

    mask_float = mask.float()
    weighted = per_token_loss * mask_float * papl_weight
    per_sample = weighted.sum(dim=1) / mask_float.sum(dim=1).clamp(min=1.0)
    elbo_weight = (k * torch.exp(-k * t) / (1.0 - torch.exp(-k * t))).clamp(max=min_snr_gamma)
    return (per_sample * elbo_weight.squeeze(1)).mean()


@torch.no_grad()
def genprobe(model, prefix_ids, cont_len, prefix_len, top_k=50, n_steps=64,
             batch_size=16, device='cuda'):
    """Fast generation probe: rep_n at multiple n + distinct_4 + top10_share.

    Used during training to give live signal on the metrics that actually matter
    for PAPL (which intentionally does not optimize standard val_loss).
    Reviewer's note: tracking rep_n at multiple n separates "scale-localized
    fix" (only rep_4 moves) from "global degeneracy fix" (rep_2..16 all move).
    """
    model.eval()
    N = prefix_ids.shape[0]
    samples = []
    for b in range(0, N, batch_size):
        chunk = prefix_ids[b:b + batch_size].to(device)
        out = demask_topk_prefix(model, chunk, cont_len, top_k=top_k,
                                 temperature=1.0, n_steps=n_steps, device=device)
        samples.append(out.cpu())
    full = torch.cat(samples, dim=0)
    completions = [row[prefix_len:].tolist() for row in full]
    return {
        'rep_2': rep_n(completions, 2),
        'rep_4': rep_n(completions, 4),
        'rep_8': rep_n(completions, 8),
        'distinct_4': distinct_n(completions, 4),
        'top10_share': top_word_share(completions, 10),
    }


def eval_mdlm_decomp(model, val_x, alpha=1.0, tau=1.0):
    """Decomposed val eval: uniform-MDLM-NLL and PAPL-planner-weighted-NLL.

    Both terms share the same masking, t-sampling, and logits (one forward
    per batch) so they're directly comparable. Min-SNR γ=1.5 clamp applied
    to both. The two numbers are the diagnostic the Peng paper cares about:

      - uniform: standard MDLM ELBO. Drops slowly (or even rises slightly)
        under PAPL by construction — PAPL deallocates capacity from
        positions the planner rarely visits. Slow movement is expected.
      - planner_w: PAPL's own training objective on val. Should drop
        meaningfully under PAPL — that's what successful sampler-aware
        training looks like in the loss space.

    Reviewer's framing: a decrease in planner_w alongside a small increase
    in uniform is the predicted signature of successful PAPL training.
    """
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
            t_batch = torch.full((min(MICRO_BATCH, n_val), 1), t_val.item(), device=device)
            uniform_b, planner_b = [], []
            for i in range(0, n_val, MICRO_BATCH):
                vx = val_x[i:i + MICRO_BATCH]
                bs = vx.shape[0]
                t = t_batch[:bs]
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

                # Uniform (vanilla MDLM)
                u_per_sample = (per_token_loss * mf).sum(dim=1) / mf.sum(dim=1).clamp(min=1.0)
                uniform_b.append((u_per_sample * ew.squeeze(1)).mean().item())

                # Planner-weighted (PAPL): same gt-logprob planner used at train
                logprobs = F.log_softmax(logits, dim=-1)
                gt_lp = logprobs.gather(-1, vx.unsqueeze(-1)).squeeze(-1)
                scores = (gt_lp / tau).masked_fill(~mask, -1e4)
                w = F.softmax(scores, dim=1) * mf
                papl_w = 1.0 + alpha * w
                p_per_sample = (per_token_loss * mf * papl_w).sum(dim=1) / mf.sum(dim=1).clamp(min=1.0)
                planner_b.append((p_per_sample * ew.squeeze(1)).mean().item())

            uniform_total += sum(uniform_b) / len(uniform_b)
            planner_total += sum(planner_b) / len(planner_b)
            n_points += 1
    return {
        'uniform_nll_minsnr': uniform_total / n_points,
        'planner_w_nll_minsnr': planner_total / n_points,
    }


def eval_mdlm_standard(model, val_x):
    """Backwards-compatible single-number eval (uniform-only). Used for callers
    that don't need the decomposition. Prefer eval_mdlm_decomp for new code."""
    return eval_mdlm_decomp(model, val_x, alpha=0.0)['uniform_nll_minsnr']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--alpha', type=float, default=1.0, help='PAPL reweight strength')
    ap.add_argument('--tau', type=float, default=1.0, help='PAPL planner temperature')
    ap.add_argument('--extra-steps', type=int, default=10000)
    ap.add_argument('--resume', type=str, default=RESUME_CKPT)
    ap.add_argument('--output-dir', type=str, default=OUTPUT_DIR)
    ap.add_argument('--genprobe-every', type=int, default=0,
                    help='Run a fast generation probe (rep_4, distinct_4) every N steps. '
                         '0 disables. Recommended 1000 for resume runs.')
    ap.add_argument('--genprobe-n-prompts', type=int, default=32,
                    help='Number of held-out prompts for the live gen probe.')
    args = ap.parse_args()

    device = 'cuda'
    os.makedirs(args.output_dir, exist_ok=True)

    print(f'Loading data (mmap)...')
    all_tokens = np.load(DATA_PATH, mmap_mode='r')
    n_total = len(all_tokens)
    val_size = 500_000_000
    train_tokens = all_tokens[:n_total - val_size]
    val_tokens = all_tokens[n_total - val_size:]

    torch.manual_seed(42)
    model = TransformerV2(vocab_size=VOCAB_SIZE, n_layer=N_LAYER, n_head=N_HEAD,
                          n_embd=N_EMBD, use_rope=True, use_swiglu=True,
                          use_qk_norm=True).to(device)
    print(f'Params: {model.param_count()/1e6:.1f}M')

    muon_params, adamw_params = model.param_groups()

    adam_opt = torch.optim.Adam(
        [{'params': adamw_params, 'lr': EMBED_LR, 'betas': (0.9, 0.999)}])
    muon_opt = Muon([{'params': muon_params}], lr=MUON_LR, momentum=0.95, ns_steps=5)

    print(f'Resuming from {args.resume}...')
    ckpt = torch.load(args.resume, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    adam_opt.load_state_dict(ckpt['adam_state_dict'])
    muon_opt.load_state_dict(ckpt['muon_state_dict'])
    start_step = ckpt['step']
    baseline_val = ckpt.get('val_loss', None)
    print(f'  resumed at step {start_step}, val_loss {baseline_val}')
    del ckpt
    torch.cuda.empty_cache()

    n_val = min(1024, len(val_tokens) // SEQ_LEN)
    val_x = torch.from_numpy(
        val_tokens[:n_val * SEQ_LEN].astype(np.int64).reshape(n_val, SEQ_LEN)
    ).to(device)

    ds = TokenDataset(train_tokens, seq_len=SEQ_LEN)
    g = torch.Generator(); g.manual_seed(12345)   # different seed vs original so batches differ
    loader = DataLoader(ds, batch_size=MICRO_BATCH, shuffle=True, num_workers=4,
                        pin_memory=True, drop_last=True, persistent_workers=True,
                        generator=g)
    data_iter = iter(loader)

    # Baseline val with both terms (uniform + planner-weighted)
    decomp0 = eval_mdlm_decomp(model, val_x, alpha=args.alpha, tau=args.tau)
    vloss0 = decomp0['uniform_nll_minsnr']
    vloss0_papl = decomp0['planner_w_nll_minsnr']
    print(f'Baseline val: uniform {vloss0:.4f} | papl_w {vloss0_papl:.4f}  '
          f'(ckpt reported {baseline_val})')

    # Load gen probe prompts if requested
    probe_prefix = None
    if args.genprobe_every > 0:
        prompts = torch.load(GENPROBE_PROMPTS_PATH, weights_only=False)
        probe_prefix = prompts['prefix_ids'][:args.genprobe_n_prompts]
        probe_cont_len = int(prompts['cont_len'])
        probe_prefix_len = int(prompts['prefix_len'])
        m0 = genprobe(model, probe_prefix, probe_cont_len, probe_prefix_len,
                      device=device)
        print(f'Genprobe baseline (n={args.genprobe_n_prompts}): '
              f'rep_4={m0["rep_4"]:.4f} distinct_4={m0["distinct_4"]:.4f} '
              f'top10={m0["top10_share"]:.4f}')

    print(f'\nPAPL fine-tune: alpha={args.alpha}, tau={args.tau}')
    print(f'Steps {start_step} → {start_step + args.extra_steps} (constant LR)')

    # Constant LR (no WSD decay for this 10k stretch)
    for pg in adam_opt.param_groups:
        pg['lr'] = EMBED_LR
    for pg in muon_opt.param_groups:
        pg['lr'] = MUON_LR

    best_val = vloss0
    t0 = time.time()
    tokens_per_step = MICRO_BATCH * GRAD_ACCUM * SEQ_LEN

    for local_step in range(args.extra_steps):
        step = start_step + local_step

        model.train()
        for opt in [adam_opt, muon_opt]:
            opt.zero_grad()
        total_loss = 0.0
        for _ in range(GRAD_ACCUM):
            try:
                x, _ = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x, _ = next(data_iter)
            x = x.to(device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = papl_mdlm_loss(model, x, alpha=args.alpha, tau=args.tau) / GRAD_ACCUM
            loss.backward()
            total_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(adamw_params, 1.0)
        for opt in [adam_opt, muon_opt]:
            opt.step()

        # Decide whether to log this step: at LOG_EVERY boundary, at genprobe
        # boundary, or at end of run. Avoids the bug where genprobe_every and
        # LOG_EVERY have no common divisor and genprobe never fires.
        is_log_step = (local_step % LOG_EVERY == 0) or \
                      (local_step == args.extra_steps - 1)
        is_genprobe_step = (args.genprobe_every > 0 and
                            ((local_step + 1) % args.genprobe_every == 0 or
                             local_step == args.extra_steps - 1))
        if is_log_step or is_genprobe_step:
            decomp = eval_mdlm_decomp(model, val_x, alpha=args.alpha, tau=args.tau)
            vloss = decomp['uniform_nll_minsnr']
            vloss_papl = decomp['planner_w_nll_minsnr']
            elapsed = time.time() - t0
            tps = (local_step + 1) * tokens_per_step / elapsed if elapsed > 0 else 0
            if vloss < best_val:
                best_val = vloss
            extra = ''
            if is_genprobe_step and probe_prefix is not None:
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
                'val_loss': vloss if local_step % LOG_EVERY == 0 else best_val,
                'papl_alpha': args.alpha,
                'papl_tau': args.tau,
                'resumed_from': args.resume,
            }, ckpt_path)
            print(f'  Checkpoint: {ckpt_path}', flush=True)

    print(f'\nDone. Best val (std Min-SNR ELBO): {best_val:.4f}  '
          f'(baseline at start: {vloss0:.4f})')
    print(f'Total time: {(time.time()-t0)/3600:.2f} hours')


if __name__ == '__main__':
    main()
