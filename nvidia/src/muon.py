"""Muon optimizer with Newton-Schulz orthogonalization.

Adapted from Keller Jordan's nanoGPT / Moonlight reference implementation.
"""

import torch
from torch.optim import Optimizer


def newton_schulz_orthogonalize(G, ns_steps=5):
    """Newton-Schulz iteration for zeroth power / orthogonalization of G.

    Uses the "cursed quintic" coefficients from Keller Jordan's Muon that
    maximize slope at zero. Singular values converge to ~[0.68, 1.13] range
    after 5 steps — this is intentional and empirically sufficient.

    Canonical reference: github.com/KellerJordan/Muon
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    # Normalize per-matrix spectral norm to at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(ns_steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


class Muon(Optimizer):
    """Muon optimizer: Nesterov momentum with Newton-Schulz orthogonalization.

    Args:
        params: Parameters to optimize (should be 2D weight matrices)
        lr: Learning rate (default 0.02, much higher than AdamW because
            updates are orthogonalized to unit spectral norm)
        momentum: Momentum factor (default 0.95)
        ns_steps: Number of Newton-Schulz iterations (default 5, 0 to disable)
        normalize_grad: If True and ns_steps=0, normalize gradient by Frobenius
            norm before stepping (for ablation: isolates norm control from
            orthogonal structure)
    """
    def __init__(self, params, lr=0.02, momentum=0.95, ns_steps=5,
                 normalize_grad=False):
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps,
                        normalize_grad=normalize_grad)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            ns_steps = group['ns_steps']
            normalize_grad = group['normalize_grad']

            for p in group['params']:
                if p.grad is None:
                    continue

                g = p.grad
                if g.dim() != 2:
                    # Fallback: just do SGD+momentum for non-2D params
                    # (shouldn't happen if param_groups are set up correctly)
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(g)
                    p.add_(buf, alpha=-lr)
                    continue

                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)

                buf = state['momentum_buffer']

                # Standard Nesterov momentum (canonical Muon formulation):
                # buf = mu * buf + g
                # update = NS(buf + mu * (buf - buf_prev))
                # Simplified: use mu * buf + g as the Nesterov lookahead
                buf_prev = buf.clone()
                buf.mul_(momentum).add_(g)
                # Nesterov look-ahead: buf + mu * (buf - buf_prev)
                g_nesterov = buf + momentum * (buf - buf_prev)

                if ns_steps > 0:
                    # Full Muon: orthogonalize the update
                    update = newton_schulz_orthogonalize(g_nesterov, ns_steps)
                    # Aspect ratio scaling for non-square matrices
                    update = update * max(1, g.size(-2) / g.size(-1)) ** 0.5
                elif normalize_grad:
                    # Ablation: normalize by Frobenius norm (cheap norm control)
                    update = g_nesterov / (g_nesterov.norm() + 1e-7)
                    update = update * max(1, g.size(-2) / g.size(-1)) ** 0.5
                else:
                    # Raw momentum (will likely need very different lr)
                    update = g_nesterov

                wd = group.get('weight_decay', 0)
                if wd > 0:
                    p.mul_(1 - lr * wd)
                p.add_(update.to(p.dtype), alpha=-lr)
