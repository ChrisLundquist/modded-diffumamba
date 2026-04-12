"""
train.py — nanogpt-style training script for DiffuMamba3

Single-file trainer with:
  - Muon optimizer for 2D hidden weights, AdamW for embeddings/biases/norms
  - FineWeb10B data loading (pre-tokenized .bin shards, modded-nanogpt format)
  - Linear cooldown LR schedule (from modded-nanogpt)
  - bfloat16 mixed precision
  - Wandb logging (optional)
  - Designed for single AMD RX 9070 XT (16GB VRAM)

Data setup:
    python data/get_data.py          # download 1B pre-tokenized tokens
    bash data/get_data.sh            # alternative: curl-based download

Usage:
    python train.py --config small --max_steps 5000
    python train.py --config base --optimizer adam  # baseline without Muon
"""

import os
import sys
import time
import math
import glob
import argparse
import functools
from pathlib import Path

# Unbuffered print for background/piped execution
print = functools.partial(print, flush=True)

import numpy as np
import torch
import torch.nn.functional as F

from model import DiffuMamba3, DiffuMamba3Config, CONFIGS, count_parameters

# ---------------------------------------------------------------------------
# Data loading (modded-nanogpt .bin shard format)
# ---------------------------------------------------------------------------

SHARD_MAGIC = 20240520

def read_shard(path: str) -> torch.Tensor:
    """Read a .bin shard → flat int64 tensor. Uses memmap (no full copy)."""
    header = np.fromfile(path, dtype=np.int32, count=256)
    assert header[0] == SHARD_MAGIC, f"Bad magic in {path}: {header[0]}"
    n = int(header[2])
    tokens = np.memmap(path, dtype=np.uint16, mode="r",
                       offset=256 * 4, shape=(n,))
    return torch.from_numpy(tokens.astype(np.int64))


def load_data(path: str, split: str = "train",
              max_tokens: int = None) -> torch.Tensor:
    """Load tokens from .bin shard directory, single .npy, or single .bin file.

    Supports:
      - Directory of .bin shards (modded-nanogpt format): loads all matching shards
      - Single .npy file (uint16): loads directly
      - Single .bin file: reads shard header + body
    """
    if path and os.path.isdir(path):
        # Load all .bin shards for this split
        pattern = os.path.join(path, f"fineweb_{split}_*.bin")
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(
                f"No {split} shards in {path}/\n"
                f"  Run: python data/get_data.py  (or bash data/get_data.sh)")
        chunks = []
        total = 0
        for f in files:
            t = read_shard(f)
            chunks.append(t)
            total += len(t)
            if max_tokens and total >= max_tokens:
                break
        tokens = torch.cat(chunks)
        if max_tokens:
            tokens = tokens[:max_tokens]
        print(f"  Loaded {len(tokens)/1e6:.1f}M {split} tokens "
              f"from {len(files)} shards in {path}/")
        return tokens

    if path and os.path.isfile(path):
        if path.endswith(".npy"):
            tokens = torch.from_numpy(np.load(path).astype(np.int32))
        else:
            tokens = read_shard(path)
        if max_tokens:
            tokens = tokens[:max_tokens]
        print(f"  Loaded {len(tokens)/1e6:.1f}M tokens from {path}")
        return tokens

    # Fallback: tiny_shakespeare (download + tokenize on first use)
    return _prepare_tiny_shakespeare(split, max_tokens)


def _prepare_tiny_shakespeare(split: str, max_tokens: int = None) -> torch.Tensor:
    """Download tiny_shakespeare once, tokenize with GPT-2, cache as .bin shards."""
    cache = Path("data")
    cache.mkdir(parents=True, exist_ok=True)
    train_path = cache / "tiny_shakespeare_train.bin"
    val_path = cache / "tiny_shakespeare_val.bin"

    if not train_path.exists() or not val_path.exists():
        print("  Downloading tiny_shakespeare (one-time fallback)...")
        import tiktoken
        from datasets import load_dataset

        enc = tiktoken.get_encoding("gpt2")
        ds = load_dataset("Trelis/tiny-shakespeare")
        text_key = "Text" if "Text" in ds["train"].column_names else "text"

        for s, p in [("train", train_path), ("test", val_path)]:
            all_tokens = []
            for example in ds[s]:
                all_tokens.extend(enc.encode_ordinary(example[text_key]))
            _write_shard(str(p), np.array(all_tokens))

    path = str(train_path if split == "train" else val_path)
    tokens = read_shard(path)
    if max_tokens:
        tokens = tokens[:max_tokens]
    print(f"  Loaded {len(tokens)/1e6:.3f}M tokens from {path} (tiny_shakespeare)")
    return tokens


def _write_shard(path: str, tokens: np.ndarray):
    """Write tokens to a .bin shard file (modded-nanogpt format)."""
    tokens = tokens.astype(np.uint16)
    header = np.zeros(256, dtype=np.int32)
    header[0] = SHARD_MAGIC
    header[1] = 1
    header[2] = len(tokens)
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens.tobytes())


class DataLoader:
    """Simple random-chunk data loader from a flat token tensor."""
    def __init__(self, tokens: torch.Tensor, seq_len: int, batch_size: int):
        self.tokens = tokens
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.n = len(tokens)
        assert self.n > seq_len, \
            f"Need more tokens ({self.n}) than seq_len ({seq_len})"

    def __iter__(self):
        return self

    def __next__(self) -> torch.Tensor:
        starts = torch.randint(0, self.n - self.seq_len, (self.batch_size,))
        return torch.stack([self.tokens[s:s + self.seq_len] for s in starts])


# ---------------------------------------------------------------------------
# Newton-Schulz orthogonalization (from Keller Jordan's Muon)
# ---------------------------------------------------------------------------

def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Orthogonalize gradient matrix via Newton-Schulz iteration.

    Computes approximate UV^T from the SVD of G.
    Quintic iteration with coefficients maximizing slope at zero.
    """
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


# ---------------------------------------------------------------------------
# Muon + AdamW hybrid optimizer
# ---------------------------------------------------------------------------

class MuonAdamW(torch.optim.Optimizer):
    """Hybrid optimizer: Muon for 2D hidden weights, AdamW for everything else.

    Based on SingleDeviceMuonWithAuxAdam from Keller Jordan's Muon repo.
    """
    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group.setdefault("lr", 0.02)
                group.setdefault("momentum", 0.95)
                group.setdefault("weight_decay", 0.0)
                group.setdefault("ns_steps", 5)
            else:
                group.setdefault("lr", 3e-4)
                group.setdefault("betas", (0.9, 0.95))
                group.setdefault("eps", 1e-10)
                group.setdefault("weight_decay", 0.0)
        super().__init__(param_groups, {})

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group["use_muon"]:
                self._muon_step(group)
            else:
                self._adam_step(group)

    def _muon_step(self, group):
        lr = group["lr"]
        beta = group["momentum"]
        wd = group["weight_decay"]
        ns_steps = group["ns_steps"]

        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad

            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(g)
            buf = state["momentum_buffer"]

            # Nesterov momentum
            buf.lerp_(g, 1 - beta)
            update = g.lerp_(buf, beta)

            # Flatten if needed (conv filters)
            orig_shape = update.shape
            if update.ndim > 2:
                update = update.view(update.size(0), -1)

            # Newton-Schulz orthogonalization
            update = zeropower_via_newtonschulz5(update, steps=ns_steps)
            update *= max(1, update.size(-2) / update.size(-1)) ** 0.5

            update = update.view(orig_shape)

            # Weight decay + update
            p.mul_(1 - lr * wd)
            p.add_(update, alpha=-lr)

    def _adam_step(self, group):
        lr = group["lr"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        wd = group["weight_decay"]

        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad

            state = self.state[p]
            if "step" not in state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)

            state["step"] += 1
            s = state["step"]

            state["exp_avg"].lerp_(g, 1 - beta1)
            state["exp_avg_sq"].lerp_(g * g, 1 - beta2)

            # Bias correction
            bc1 = 1 - beta1 ** s
            bc2 = 1 - beta2 ** s

            # Weight decay (decoupled)
            p.mul_(1 - lr * wd)

            # Adam update
            p.addcdiv_(state["exp_avg"] / bc1,
                        (state["exp_avg_sq"] / bc2).sqrt() + eps,
                        value=-lr)


# ---------------------------------------------------------------------------
# Learning rate schedule: warmup + cooldown
# ---------------------------------------------------------------------------

def get_lr_multiplier(step: int, warmup_steps: int, max_steps: int,
                       cooldown_frac: float = 0.6, min_lr_frac: float = 0.1,
                       schedule: str = "linear") -> float:
    """Linear warmup then cooldown. schedule: 'linear' or 'cosine'."""
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    if schedule == "cosine":
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return min_lr_frac + 0.5 * (1 - min_lr_frac) * (1 + math.cos(math.pi * progress))
    else:  # linear cooldown
        cd_start = int(max_steps * (1 - cooldown_frac))
        if step >= cd_start:
            t = min(1.0, (step - cd_start) / max(1, max_steps - cd_start))
            return 1.0 * (1 - t) + min_lr_frac * t
        return 1.0


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def build_optimizer(model: DiffuMamba3, args) -> torch.optim.Optimizer:
    """Build Muon+AdamW hybrid optimizer, or plain AdamW."""

    if args.optimizer == "muon":
        muon_params = []
        adam_params = []

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            # Muon for 2D weights in mamba blocks (not AdaLN, norms, or out_proj)
            is_hidden_2d = (
                p.ndim >= 2
                and "blocks" in name
                and "adaln" not in name
                and "norm" not in name
                and "out_proj" not in name
            )
            if is_hidden_2d:
                muon_params.append(p)
            else:
                adam_params.append(p)

        print(f"  Muon params: {sum(p.numel() for p in muon_params)/1e6:.1f}M "
              f"({len(muon_params)} tensors)")
        print(f"  Adam params: {sum(p.numel() for p in adam_params)/1e6:.1f}M "
              f"({len(adam_params)} tensors)")

        param_groups = [
            dict(params=muon_params, use_muon=True,
                 lr=args.muon_lr, momentum=args.muon_momentum,
                 weight_decay=args.muon_wd),
            dict(params=adam_params, use_muon=False,
                 lr=args.adam_lr, betas=(0.9, args.adam_beta2),
                 weight_decay=args.adam_wd),
        ]
        return MuonAdamW(param_groups)

    else:  # plain AdamW
        return torch.optim.AdamW(
            model.parameters(), lr=args.adam_lr,
            betas=(0.9, 0.95), weight_decay=args.adam_wd,
        )


def train(args):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Model config
    if args.config in CONFIGS:
        config = CONFIGS[args.config]
    else:
        raise ValueError(f"Unknown config: {args.config}. "
                         f"Choose from: {list(CONFIGS.keys())}")

    # Override config from args
    if args.n_layers is not None:
        config.n_layers = args.n_layers
    if args.d_model is not None:
        config.d_model = args.d_model
    if args.seq_len is not None:
        config.max_seq_len = args.seq_len
    config.time_conditioning = not args.no_time_cond
    config.loss_weight = args.loss_weight
    config.minsnr_gamma = args.minsnr_gamma

    # Build model
    model = DiffuMamba3(config).to(device)
    if device.type == "cuda" and config.dtype == torch.bfloat16:
        model = model.to(dtype=torch.bfloat16)
    n_params = count_parameters(model)

    if args.compile:
        print("  Compiling model with torch.compile...")
        model = torch.compile(model)

    print(f"\nModel: {args.config} ({n_params/1e6:.1f}M params)")
    print(f"  d_model={config.d_model}, n_layers={config.n_layers}, "
          f"seq_len={config.max_seq_len}, compile={args.compile}")
    lw_str = config.loss_weight
    if config.loss_weight == "minsnr":
        lw_str += f"(gamma={config.minsnr_gamma})"
    print(f"  time_conditioning={config.time_conditioning}, "
          f"loss_weight={lw_str}, lr_schedule={args.lr_schedule}")

    # Data
    print(f"\nLoading data...")
    data_src = args.data_path or args.data_dir
    tokens = load_data(data_src, split="train", max_tokens=args.max_data_tokens)
    loader = DataLoader(tokens, config.max_seq_len, args.batch_size)

    val_src = args.val_data_path or data_src
    val_tokens = load_data(val_src, split="val")
    val_loader = DataLoader(val_tokens, config.max_seq_len, args.batch_size)

    # Optimizer
    print(f"\nOptimizer: {args.optimizer}")
    optimizer = build_optimizer(model, args)

    # Wandb
    if args.wandb:
        import wandb
        wandb.init(project="diffumamba3", config=vars(args))
        wandb.watch(model, log_freq=100)

    # Training loop
    print(f"\nTraining for {args.max_steps} steps, batch_size={args.batch_size}")
    print(f"  Effective tokens/step: {args.batch_size * config.max_seq_len:,}")
    print("=" * 60)

    model.train()
    best_val_loss = float("inf")
    t0 = time.perf_counter()

    for step in range(args.max_steps + 1):
        # ---- Validation ----
        if step % args.val_every == 0:
            model.eval()
            # Always use standard ELBO weighting for val (comparable across configs)
            saved_lw = config.loss_weight
            config.loss_weight = "elbo"
            val_losses = []
            with torch.no_grad():
                for _ in range(args.val_steps):
                    x_0 = next(val_loader).to(device)
                    loss, _ = model.compute_loss(x_0)
                    val_losses.append(loss.item())
            config.loss_weight = saved_lw
            val_loss = sum(val_losses) / len(val_losses)
            elapsed = time.perf_counter() - t0

            tokens_seen = step * args.batch_size * config.max_seq_len
            is_best = val_loss < best_val_loss
            best_val_loss = min(best_val_loss, val_loss)
            marker = " *" if is_best else ""

            print(f"step {step:>6d}/{args.max_steps} | "
                  f"val_loss {val_loss:.4f}{marker} | "
                  f"tokens {tokens_seen/1e6:.1f}M | "
                  f"time {elapsed:.0f}s")

            if args.wandb:
                import wandb
                wandb.log({
                    "val/loss": val_loss,
                    "val/best_loss": best_val_loss,
                    "tokens": tokens_seen,
                    "time": elapsed,
                }, step=step)

            if is_best and args.save_best:
                torch.save(model.state_dict(), args.save_path)

            model.train()

        if step == args.max_steps:
            break

        # ---- Training step ----
        x_0 = next(loader).to(device)

        # LR schedule
        lr_mult = get_lr_multiplier(step, args.warmup_steps, args.max_steps,
                                     schedule=args.lr_schedule)
        for group in optimizer.param_groups:
            if group.get("use_muon", False):
                group["lr"] = args.muon_lr * lr_mult
            else:
                group["lr"] = args.adam_lr * lr_mult

        # Forward + loss
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, metrics = model.compute_loss(x_0)
        else:
            loss, metrics = model.compute_loss(x_0)

        # Backward
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient clipping
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()

        # Logging
        if step % args.log_every == 0 and step > 0:
            elapsed = time.perf_counter() - t0
            tokens_per_sec = (step * args.batch_size * config.max_seq_len) / elapsed
            base_lr = args.muon_lr if args.optimizer == "muon" else args.adam_lr
            print(f"step {step:>6d} | loss {metrics['loss']:.4f} | "
                  f"mask {metrics['mask_rate']:.2f} | "
                  f"lr {lr_mult * base_lr:.2e} | "
                  f"{tokens_per_sec/1e3:.1f}k tok/s")

            if args.wandb:
                import wandb
                wandb.log({
                    "train/loss": metrics["loss"],
                    "train/mask_rate": metrics["mask_rate"],
                    "train/lr": lr_mult * base_lr,
                    "train/tokens_per_sec": tokens_per_sec,
                }, step=step)

    # Done
    total_time = time.perf_counter() - t0
    total_tokens = args.max_steps * args.batch_size * config.max_seq_len
    print("=" * 60)
    print(f"Training complete: {total_time:.0f}s, "
          f"{total_tokens/1e6:.0f}M tokens, "
          f"best val_loss={best_val_loss:.4f}")

    return best_val_loss


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="DiffuMamba3 Training")

    # Model
    p.add_argument("--config", type=str, default="quokka",
                   choices=list(CONFIGS.keys()), help="Model config")
    p.add_argument("--n_layers", type=int, default=None)
    p.add_argument("--d_model", type=int, default=None)
    p.add_argument("--seq_len", type=int, default=None)
    p.add_argument("--no_time_cond", action="store_true",
                   help="Disable timestep conditioning (MDLM default)")
    p.add_argument("--loss_weight", type=str, default="minsnr",
                   choices=["elbo", "flat", "minsnr"],
                   help="Loss weighting: elbo (1/t), flat (1), minsnr (clamped 1/t)")
    p.add_argument("--minsnr_gamma", type=float, default=5.0,
                   help="Clamp value for Min-SNR weighting (default 5)")

    # Optimizer
    p.add_argument("--optimizer", type=str, default="muon",
                   choices=["muon", "adam"],
                   help="Optimizer (muon = Muon+AdamW hybrid)")
    p.add_argument("--muon_lr", type=float, default=0.02)
    p.add_argument("--muon_momentum", type=float, default=0.95)
    p.add_argument("--muon_wd", type=float, default=0.01)
    p.add_argument("--adam_lr", type=float, default=3e-4)
    p.add_argument("--adam_wd", type=float, default=0.01)
    p.add_argument("--adam_beta2", type=float, default=0.999,
                   help="Adam beta2 (MDLM=0.999, DiffuMamba=0.95)")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--warmup_steps", type=int, default=50)
    p.add_argument("--lr_schedule", type=str, default="cosine",
                   choices=["cosine", "linear"],
                   help="LR schedule: cosine (DiffuMamba) or linear cooldown")

    # Training
    p.add_argument("--max_steps", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--compile", action="store_true",
                   help="Use torch.compile (Triton backend on ROCm)")

    # Data
    p.add_argument("--data_dir", type=str, default="data/fineweb10B",
                   help="Directory of .bin shard files (modded-nanogpt format)")
    p.add_argument("--data_path", type=str, default=None,
                   help="Single .npy or .bin file (overrides --data_dir)")
    p.add_argument("--val_data_path", type=str, default=None,
                   help="Explicit val data path (defaults to --data_dir val shards)")
    p.add_argument("--max_data_tokens", type=int, default=None,
                   help="Limit training tokens loaded into memory")

    # Eval
    p.add_argument("--val_every", type=int, default=250)
    p.add_argument("--val_steps", type=int, default=10)
    p.add_argument("--log_every", type=int, default=10)

    # Saving
    p.add_argument("--save_best", action="store_true")
    p.add_argument("--save_path", type=str, default="best_model.pt")

    # Logging
    p.add_argument("--wandb", action="store_true")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
