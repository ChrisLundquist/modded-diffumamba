"""Weight-geometric analysis of DiffuMamba3 checkpoints.

For each 2D weight matrix in a state_dict this computes:
  - stable rank = ||W||_F^2 / ||W||_2^2  (dimensionless, cheap)
  - SVD (normalized) entropy on p_i = sigma_i^2 / sum sigma_i^2
  - sigma_max, sigma_min (non-zero), condition number
  - PL-alpha (Martin-Mahoney heavy-tailed fit on tail of sigma^2, d >= 500 only)
  - Frobenius / spectral norms

Layout notes (Mamba3 non-MIMO, as trained here):
  in_proj.weight  shape (D_in_proj, D_model) with concatenation order:
    [z(d_inner), x(d_inner), B(d_state*nbc_heads), C(d_state*nbc_heads),
     dd_dt(nheads), dd_A(nheads), trap(nheads), angle(num_rope_angles)]
  ngroups=nbc_heads=1, mimo_rank=1, rope_fraction=0.5,
  so num_rope_angles = d_state // 4 (d_state=32 -> 8).

The in_proj matrix is heterogeneous; we slice it into its homogeneous pieces
and compute metrics for the z/x/B/C/dt/A/trap/angle sub-blocks individually.

All SVDs are computed in fp32 on CPU via torch.linalg.svdvals.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent
CKPT_DIR = REPO_ROOT / "checkpoints"
OUT_DIR = REPO_ROOT / "results" / "geometry"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_svdvals(W: torch.Tensor) -> np.ndarray:
    """Return singular values of W as fp32 numpy (descending)."""
    if W.dim() != 2:
        raise ValueError(f"Need 2D, got {W.shape}")
    W32 = W.detach().to(torch.float32).cpu()
    # torch.linalg.svdvals: returns values in descending order
    sv = torch.linalg.svdvals(W32)
    return sv.numpy()


def stable_rank(sv: np.ndarray) -> float:
    smax = float(sv[0]) if len(sv) else 0.0
    if smax <= 0:
        return 0.0
    return float(np.sum(sv * sv) / (smax * smax))


def svd_entropy(sv: np.ndarray, normalize: bool = True) -> float:
    """Shannon entropy of p_i = sigma_i^2 / sum sigma_i^2.
    Returns nats. If normalize=True, divides by log(len(sv)) to [0,1]."""
    if len(sv) == 0:
        return 0.0
    p = sv * sv
    s = float(p.sum())
    if s <= 0:
        return 0.0
    p = p / s
    eps = 1e-30
    H = float(-np.sum(p * np.log(p + eps)))
    if normalize:
        Hmax = math.log(len(sv))
        return H / Hmax if Hmax > 0 else 0.0
    return H


def condition_number(sv: np.ndarray) -> float:
    sv_nz = sv[sv > 0]
    if len(sv_nz) < 2:
        return float("inf")
    return float(sv_nz[0] / sv_nz[-1])


def pl_alpha(sv: np.ndarray, min_tail_frac: float = 0.5) -> Optional[float]:
    """Martin-Mahoney heavy-tailed self-regularization: fit PL exponent to tail
    of eigenvalue spectrum lambda_i = sigma_i^2 via maximum likelihood.

    alpha-hat = 1 + n / sum_i log(lambda_i / lambda_min_tail).
    We use the simple fixed-tail approach (min_tail_frac of the largest eigenvalues).

    Returns None if d < 500 or if fit degenerates.
    """
    d = len(sv)
    if d < 500:
        return None
    lam = sv * sv
    lam = lam[lam > 1e-30]
    if len(lam) < 20:
        return None
    n_tail = max(10, int(len(lam) * min_tail_frac))
    tail = lam[:n_tail]  # sv already descending -> tail = largest eigenvalues
    lam_min = float(tail[-1])
    if lam_min <= 0:
        return None
    logs = np.log(tail / lam_min + 1e-30)
    s = float(logs.sum())
    if s <= 0:
        return None
    return 1.0 + len(tail) / s


def matrix_metrics(W: torch.Tensor, compute_alpha: bool = True) -> Dict:
    """Full metric dict for a 2D matrix."""
    shape = tuple(W.shape)
    sv = compute_svdvals(W)
    fro = float(np.sqrt(float(np.sum(sv * sv))))
    spec = float(sv[0]) if len(sv) else 0.0
    m = min(shape)
    out = {
        "shape": list(shape),
        "min_dim": int(m),
        "fro_norm": fro,
        "spec_norm": spec,
        "stable_rank": stable_rank(sv),
        "stable_rank_norm": stable_rank(sv) / float(m),
        "svd_entropy": svd_entropy(sv, normalize=True),
        "cond_number": condition_number(sv),
        "sigma_max": float(sv[0]) if len(sv) else 0.0,
        "sigma_min": float(sv[sv > 0][-1]) if (sv > 0).any() else 0.0,
        "sigma_top10": [float(x) for x in sv[:10].tolist()],
    }
    if compute_alpha and m >= 500:
        a = pl_alpha(sv)
        out["pl_alpha"] = a
    # Keep the spectrum around for CDF plotting at a coarse resolution:
    n_keep = min(256, len(sv))
    idx = np.linspace(0, len(sv) - 1, n_keep).astype(int)
    out["sv_samples"] = sv[idx].tolist()
    out["sv_len"] = int(len(sv))
    return out


# ---------------------------------------------------------------------------
# In-proj slicing
# ---------------------------------------------------------------------------

@dataclass
class Mamba3Shape:
    d_model: int
    d_inner: int
    d_state: int = 32
    nheads: int = 0
    num_bc_heads: int = 1
    mimo_rank: int = 1
    rope_fraction: float = 0.5

    @property
    def num_rope_angles(self) -> int:
        split = int(self.d_state * self.rope_fraction)
        if split % 2 != 0:
            split -= 1
        return split // 2

    def slice_in_proj(self, W: torch.Tensor) -> Dict[str, torch.Tensor]:
        """W: (D_in_proj, d_model). Returns named sub-matrices.

        Order per Mamba3 source: [z, x, B, C, dd_dt, dd_A, trap, angle]."""
        bc_dim = self.d_state * self.num_bc_heads * self.mimo_rank
        offsets = [
            ("z", self.d_inner),
            ("x", self.d_inner),
            ("B", bc_dim),
            ("C", bc_dim),
            ("dd_dt", self.nheads),
            ("dd_A", self.nheads),
            ("trap", self.nheads),
            ("angle", self.num_rope_angles),
        ]
        total = sum(s for _, s in offsets)
        if total != W.shape[0]:
            raise ValueError(
                f"in_proj slice total {total} != W.shape[0] {W.shape[0]}; "
                f"offsets={offsets}, d_inner={self.d_inner}, nheads={self.nheads}"
            )
        out = {}
        pos = 0
        for name, n in offsets:
            out[name] = W[pos: pos + n, :]
            pos += n
        return out


def infer_shape_from_state_dict(sd: Dict[str, torch.Tensor]) -> Mamba3Shape:
    """Infer d_model/d_inner/d_state/nheads from checkpoint tensor shapes."""
    d_model = sd["tok_emb.weight"].shape[1]
    # out_proj is (d_model, d_inner)
    d_inner = sd["blocks.0.mamba_fwd.out_proj.weight"].shape[1]
    nheads = sd["blocks.0.mamba_fwd.D"].shape[0]
    # B_bias shape (nheads, mimo_rank, d_state)
    b_bias = sd["blocks.0.mamba_fwd.B_bias"]
    mimo_rank = b_bias.shape[1]
    d_state = b_bias.shape[2]
    sh = Mamba3Shape(d_model=d_model, d_inner=d_inner, d_state=d_state,
                     nheads=nheads, num_bc_heads=1, mimo_rank=mimo_rank,
                     rope_fraction=0.5)
    return sh


# ---------------------------------------------------------------------------
# Checkpoint analysis
# ---------------------------------------------------------------------------

def state_dict_from_ckpt(path: Path) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"]
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    return ckpt


# Classification of each matrix into (layer_type, optimizer_routing)
# Routing depends on the run (muon_out_proj flag); caller supplies 'muon_out_proj'.
def classify_matrix(name: str, muon_out_proj: bool) -> Tuple[str, str]:
    """Return (layer_type, routing) where routing in {'muon','adam','na'}.

    layer_type: 'emb', 'pos_emb', 'sigma_map', 'mlp_w1', 'mlp_w2', 'mlp_w3',
                'mamba_in_proj', 'mamba_out_proj', 'adaln', 'merge_gate', 'other'.
    """
    routing = "na"
    lt = "other"
    if name == "tok_emb.weight":
        lt, routing = "emb", "adam"
    elif name == "pos_emb.weight":
        lt, routing = "pos_emb", "adam"
    elif name.startswith("sigma_map."):
        lt, routing = "sigma_map", "adam"
    elif name.startswith("blocks.") and ".adaln_" in name:
        lt, routing = "adaln", "adam"
    elif name.startswith("blocks.") and ".merge_gate" in name:
        lt, routing = "merge_gate", "adam"
    elif name.startswith("blocks.") and ".mlp.w1" in name:
        lt, routing = "mlp_w1", "muon"
    elif name.startswith("blocks.") and ".mlp.w2" in name:
        lt, routing = "mlp_w2", "muon"
    elif name.startswith("blocks.") and ".mlp.w3" in name:
        lt, routing = "mlp_w3", "muon"
    elif ".in_proj.weight" in name:
        lt, routing = "mamba_in_proj", "muon"
    elif ".out_proj.weight" in name:
        lt, routing = "mamba_out_proj", "muon" if muon_out_proj else "adam"
    return lt, routing


def analyze_checkpoint(
    path: Path,
    muon_out_proj: bool,
    slice_in_proj: bool = True,
    skip_emb: bool = False,
) -> Dict:
    sd = state_dict_from_ckpt(path)
    shape = infer_shape_from_state_dict(sd)

    results: Dict[str, Dict] = {}

    n_blocks = 0
    for k in sd.keys():
        if k.startswith("blocks."):
            n_blocks = max(n_blocks, int(k.split(".")[1]) + 1)

    for name, tensor in sd.items():
        if not torch.is_tensor(tensor):
            continue
        if tensor.dim() < 2:
            continue
        lt, routing = classify_matrix(name, muon_out_proj)

        # Special handling of in_proj: slice into heterogeneous blocks.
        if lt == "mamba_in_proj":
            if slice_in_proj:
                subs = shape.slice_in_proj(tensor)
                # Only run SVD on the parts that are 2D (>=2 rows).
                for sub_name, sub_W in subs.items():
                    if sub_W.dim() != 2 or sub_W.shape[0] < 2:
                        continue
                    key = f"{name}::{sub_name}"
                    results[key] = {
                        "layer_type": f"in_proj_{sub_name}",
                        "routing": routing,
                        "metrics": matrix_metrics(sub_W),
                    }
                continue
            # Else fall through and analyze the full fused matrix.

        # Size/embedding handling
        if skip_emb and lt in ("emb", "pos_emb"):
            continue

        # Skip 3D+ tensors (D,bias etc are 1D and skipped above)
        if tensor.dim() != 2:
            continue

        try:
            m = matrix_metrics(tensor)
        except Exception as e:
            m = {"error": str(e), "shape": list(tensor.shape)}
        results[name] = {
            "layer_type": lt,
            "routing": routing,
            "metrics": m,
        }

    summary = {
        "checkpoint": str(path),
        "shape": asdict(shape),
        "n_blocks": n_blocks,
        "muon_out_proj": muon_out_proj,
        "matrices": results,
    }
    return summary


# ---------------------------------------------------------------------------
# Orchestration: run the three comparisons
# ---------------------------------------------------------------------------

# Characterize which sweeps used --muon_out_proj.
# opt10k_*: NO (default)
# 10L640d_*: YES (train_large.py sets it)
# best10k_vs_outproj_*: YES
# best10k_vs_no_outproj_*: NO
# best10k_adam_*: N/A (all adam)
# final10k_{new,old}_best_*: YES
def ckpt_uses_muon_out_proj(name: str) -> bool:
    if "10L640d" in name:
        return True
    if "vs_outproj" in name:
        return True
    if "vs_no_outproj" in name:
        return False
    if "final10k_new_best" in name or "final10k_old_best" in name:
        return True
    if name.startswith("opt10k_muon"):
        return False
    if name.startswith("best10k_adam") or name.startswith("opt10k_adam"):
        return False
    # Conservative default
    return False


def task_trajectory(out_dir: Path) -> None:
    """Analyze the 10L640d trajectory checkpoints."""
    print("\n" + "=" * 70)
    print("TASK 1: temporal trajectory on 10L640d (steps 10k/20k/30k/40k/50k)")
    print("=" * 70)
    names = [
        "10L640d_50k_step10000.pt",
        "10L640d_50k_step20000.pt",
        "10L640d_50k_step30000.pt",
        "10L640d_50k_step40000.pt",
        "10L640d_50k_step50000.pt",
    ]
    all_results = []
    for n in names:
        p = CKPT_DIR / n
        if not p.exists():
            print(f"  MISSING {p}")
            continue
        print(f"  analyzing {n} ...")
        res = analyze_checkpoint(p, muon_out_proj=True, slice_in_proj=True)
        res["step"] = int(n.split("step")[1].split(".")[0])
        all_results.append(res)
        with open(out_dir / f"trajectory_{n.replace('.pt','.json')}", "w") as f:
            json.dump(res, f, indent=2)
    with open(out_dir / "trajectory_all.json", "w") as f:
        json.dump({"runs": all_results}, f, indent=2)
    print(f"  wrote trajectory_all.json ({len(all_results)} checkpoints)")


def task_optimizer_comparison(out_dir: Path) -> None:
    """Analyze the 4-optimizer paired comparison at 10k, across 3 seeds."""
    print("\n" + "=" * 70)
    print("TASK 2: optimizer paired comparison at 10k (3 seeds)")
    print("=" * 70)
    optimizers = ["muon", "muon_vs", "mousse", "adam"]
    seeds = [42, 137, 2024]
    all_results = []
    for opt in optimizers:
        for s in seeds:
            name = f"opt10k_{opt}_s{s}.pt"
            p = CKPT_DIR / name
            if not p.exists():
                print(f"  MISSING {p}")
                continue
            # opt10k is trained WITHOUT --muon_out_proj
            muon_op = False if opt != "adam" else False
            print(f"  analyzing {name} (muon_out_proj={muon_op}) ...")
            res = analyze_checkpoint(p, muon_out_proj=muon_op, slice_in_proj=True)
            res["optimizer"] = opt
            res["seed"] = s
            all_results.append(res)
    with open(out_dir / "optimizer_all.json", "w") as f:
        json.dump({"runs": all_results}, f, indent=2)
    print(f"  wrote optimizer_all.json ({len(all_results)} checkpoints)")


def task_embedding_vs_block(out_dir: Path) -> None:
    """Within the SAME model, compare Adam-trained embedding vs Muon-trained blocks.

    Uses the final 10L640d step50000 checkpoint (our strongest model).
    """
    print("\n" + "=" * 70)
    print("TASK 3: embedding (Adam) vs block (Muon) within the same model")
    print("=" * 70)
    # We can reuse trajectory step50000 output if it's already saved.
    p = CKPT_DIR / "10L640d_50k_step50000.pt"
    if not p.exists():
        print(f"  MISSING {p}")
        return
    # Persist the per-matrix metrics for this single checkpoint
    res = analyze_checkpoint(p, muon_out_proj=True, slice_in_proj=True,
                             skip_emb=False)
    res["label"] = "10L640d_step50000"
    with open(out_dir / "embedding_vs_block.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"  wrote embedding_vs_block.json")


def task_init_reference(out_dir: Path) -> None:
    """Build an approximate freshly-initialized 10L640d state_dict (seed=42).

    We do NOT construct the full DiffuMamba3 model because Mamba3's CUDA/Triton
    backend interferes with CPU analysis. Instead we mirror _init_weights as
    best we can with the shapes read from the trained checkpoint. This is a
    proxy, not the actual trained init. Flagged clearly in the report.
    """
    print("\n" + "=" * 70)
    print("TASK 4: approximate fresh-init reference (seed=42)")
    print("=" * 70)
    try:
        import sys
        sys.path.insert(0, str(REPO_ROOT))
        torch.manual_seed(42)
        # Derive shapes from an existing 10L640d ckpt
        p = CKPT_DIR / "10L640d_50k_step10000.pt"
        template_sd = state_dict_from_ckpt(p)
        sd = {}
        for name, tensor in template_sd.items():
            if not torch.is_tensor(tensor):
                sd[name] = tensor
                continue
            shape = tensor.shape
            dtype = torch.float32
            if tensor.dim() >= 2:
                if ".adaln_" in name or "merge_gate" in name:
                    # AdaLN is zero-initialized in _init_weights
                    sd[name] = torch.zeros(shape, dtype=dtype)
                else:
                    # Standard Linear init: normal_(std=0.02)
                    sd[name] = torch.randn(shape, dtype=dtype) * 0.02
            elif tensor.dim() == 1:
                # biases (adaln, B_bias, C_bias, D) — keep as zeros / ones trivially
                # D is ones, B_bias/C_bias are ones + zeros
                if "bias" in name and ".adaln_" in name:
                    sd[name] = torch.zeros(shape, dtype=dtype)
                elif name.endswith(".D"):
                    sd[name] = torch.ones(shape, dtype=dtype)
                elif "_norm.weight" in name:
                    sd[name] = torch.ones(shape, dtype=dtype)
                else:
                    # dt_bias / B_bias / C_bias: random-ish; zeros is a reasonable proxy
                    sd[name] = torch.zeros(shape, dtype=dtype)
            else:
                sd[name] = torch.zeros_like(tensor, dtype=dtype)
        # Save metrics only
        init_metrics = {}
        muon_op_default = True
        shape = infer_shape_from_state_dict(sd)
        for name, tensor in sd.items():
            if not torch.is_tensor(tensor) or tensor.dim() != 2:
                continue
            lt, routing = classify_matrix(name, muon_op_default)
            if lt == "mamba_in_proj":
                subs = shape.slice_in_proj(tensor)
                for sub_name, sub_W in subs.items():
                    if sub_W.dim() != 2 or sub_W.shape[0] < 2:
                        continue
                    key = f"{name}::{sub_name}"
                    init_metrics[key] = {
                        "layer_type": f"in_proj_{sub_name}",
                        "routing": routing,
                        "metrics": matrix_metrics(sub_W),
                    }
                continue
            init_metrics[name] = {
                "layer_type": lt,
                "routing": routing,
                "metrics": matrix_metrics(tensor),
            }
        # Also save a state_dict of the init for change-magnitude calculations
        init_sd_path = out_dir / "init_10L640d_s42.pt"
        torch.save(sd, init_sd_path)
        with open(out_dir / "init_metrics.json", "w") as f:
            json.dump({"label": "init_10L640d_s42", "matrices": init_metrics}, f, indent=2)
        print(f"  wrote init_metrics.json and init_10L640d_s42.pt")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  FAILED to build init reference: {e}")


def task_change_magnitudes(out_dir: Path) -> None:
    """Compute ||W_t - W_init||_F / ||W_init||_F for the 10L640d trajectory.

    Reference "init" = the freshly-initialized model from task_init_reference().
    Also computes ||W_t - W_10k||_F relative change as a second reference.
    """
    print("\n" + "=" * 70)
    print("TASK 5: per-layer change magnitude over training")
    print("=" * 70)
    init_sd_path = out_dir / "init_10L640d_s42.pt"
    init_sd = torch.load(init_sd_path, map_location="cpu", weights_only=False) if init_sd_path.exists() else None
    if init_sd is None:
        print("  skipping (no init_sd found)")
        return

    ckpt_names = [
        "10L640d_50k_step10000.pt",
        "10L640d_50k_step20000.pt",
        "10L640d_50k_step30000.pt",
        "10L640d_50k_step40000.pt",
        "10L640d_50k_step50000.pt",
    ]
    ref_sd = None
    results = []
    for nm in ckpt_names:
        p = CKPT_DIR / nm
        if not p.exists():
            continue
        sd = state_dict_from_ckpt(p)
        step = int(nm.split("step")[1].split(".")[0])
        row = {"step": step, "layers": {}}
        for name, tensor in sd.items():
            if not torch.is_tensor(tensor) or tensor.dim() < 2:
                continue
            if name not in init_sd:
                continue
            W = tensor.to(torch.float32)
            W0 = init_sd[name].to(torch.float32)
            if W.shape != W0.shape:
                continue
            denom = float(torch.linalg.norm(W0)) + 1e-12
            row["layers"][name] = {
                "abs_change_from_init": float(torch.linalg.norm(W - W0)),
                "rel_change_from_init": float(torch.linalg.norm(W - W0) / denom),
            }
            if ref_sd is not None and name in ref_sd:
                Wref = ref_sd[name].to(torch.float32)
                if Wref.shape == W.shape:
                    denom2 = float(torch.linalg.norm(Wref)) + 1e-12
                    row["layers"][name]["abs_change_from_10k"] = float(torch.linalg.norm(W - Wref))
                    row["layers"][name]["rel_change_from_10k"] = float(torch.linalg.norm(W - Wref) / denom2)
        results.append(row)
        if ref_sd is None:
            ref_sd = sd  # step 10k is "early reference"
    with open(out_dir / "change_magnitudes.json", "w") as f:
        json.dump({"runs": results}, f, indent=2)
    print(f"  wrote change_magnitudes.json ({len(results)} steps)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=None,
                   help="single checkpoint path; if given, analyze that only")
    p.add_argument("--muon_out_proj", action="store_true",
                   help="override: treat out_proj as Muon-routed")
    p.add_argument("--task", choices=["trajectory", "optim", "emb_vs_block", "init",
                                       "change", "all"], default="all")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.ckpt is not None:
        path = Path(args.ckpt)
        muon_op = args.muon_out_proj or ckpt_uses_muon_out_proj(path.stem)
        res = analyze_checkpoint(path, muon_out_proj=muon_op, slice_in_proj=True)
        out = OUT_DIR / f"single_{path.stem}.json"
        with open(out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"wrote {out}")
        return

    if args.task in ("trajectory", "all"):
        task_trajectory(OUT_DIR)
    if args.task in ("init", "all"):
        task_init_reference(OUT_DIR)
    if args.task in ("change", "all"):
        task_change_magnitudes(OUT_DIR)
    if args.task in ("optim", "all"):
        task_optimizer_comparison(OUT_DIR)
    if args.task in ("emb_vs_block", "all"):
        task_embedding_vs_block(OUT_DIR)


if __name__ == "__main__":
    main()
