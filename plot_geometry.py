"""Plot weight-geometric analysis results.

Produces:
  - stable_rank_vs_step.png         : stable-rank trajectory per layer-type
  - svd_entropy_vs_step.png         : SVD entropy trajectory per layer-type
  - sigma_max_vs_step.png           : spectral norm trajectory per layer-type
  - optimizer_entropy_violin.png    : per-matrix SVD entropy by optimizer
  - optimizer_stable_rank.png       : per-matrix stable rank by optimizer
  - sigma_cdf_muon_vs_adam.png      : sigma CDF on matched matrices
  - change_magnitude_heatmap.png    : per-layer |W_t - W_init|/|W_init| heatmap
  - emb_vs_block_spectra.png        : Adam-routed vs Muon-routed within one model
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent
OUT_DIR = REPO_ROOT / "results" / "geometry"

# Consistent colors
COLORS = {
    "muon": "#1f77b4",
    "muon_vs": "#2ca02c",
    "mousse": "#d62728",
    "adam": "#ff7f0e",
    "mlp_w1": "#1f77b4",
    "mlp_w2": "#2ca02c",
    "mlp_w3": "#d62728",
    "mamba_out_proj": "#9467bd",
    "mamba_in_proj": "#8c564b",
    "in_proj_z": "#17becf",
    "in_proj_x": "#bcbd22",
    "in_proj_B": "#e377c2",
    "in_proj_C": "#7f7f7f",
    "in_proj_dd_dt": "#8c564b",
    "in_proj_dd_A": "#9467bd",
    "in_proj_trap": "#d62728",
    "in_proj_angle": "#2ca02c",
    "emb": "#ff7f0e",
    "pos_emb": "#000000",
    "sigma_map": "#888888",
    "adaln": "#cccccc",
}


def load_trajectory() -> List[Dict]:
    return json.load(open(OUT_DIR / "trajectory_all.json"))["runs"]


def load_optimizer() -> List[Dict]:
    return json.load(open(OUT_DIR / "optimizer_all.json"))["runs"]


def load_change() -> List[Dict]:
    return json.load(open(OUT_DIR / "change_magnitudes.json"))["runs"]


def group_by_layertype(run: Dict, key: str) -> Dict[str, List[float]]:
    """Collect metric values, keyed by layer_type, across all matrices in a run."""
    g = defaultdict(list)
    for name, mat in run["matrices"].items():
        v = mat["metrics"].get(key)
        if v is None or (isinstance(v, float) and math.isinf(v)):
            continue
        g[mat["layer_type"]].append(float(v))
    return g


# ---------------------------------------------------------------------------
# Trajectory plots
# ---------------------------------------------------------------------------

def plot_trajectory_metric(metric_key: str, ylabel: str, filename: str,
                            log_y: bool = False,
                            layer_types: List[str] = None) -> None:
    traj = load_trajectory()
    traj.sort(key=lambda r: r["step"])
    steps = [r["step"] for r in traj]

    # gather per-layer-type mean + std at each step
    if layer_types is None:
        layer_types = ["mlp_w1", "mlp_w2", "mlp_w3",
                       "mamba_out_proj",
                       "in_proj_z", "in_proj_x",
                       "in_proj_B", "in_proj_C",
                       "in_proj_dd_dt", "in_proj_trap",
                       "emb", "pos_emb"]

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for lt in layer_types:
        means = []
        stds = []
        for r in traj:
            g = group_by_layertype(r, metric_key)
            vals = g.get(lt, [])
            if not vals:
                means.append(np.nan)
                stds.append(0.0)
                continue
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals)))
        means = np.array(means)
        stds = np.array(stds)
        if np.isnan(means).all():
            continue
        color = COLORS.get(lt, None)
        ax.plot(steps, means, "-o", label=lt, color=color, lw=1.5)
        ax.fill_between(steps, means - stds, means + stds, alpha=0.15, color=color)

    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel(ylabel)
    ax.set_title(f"10L640d trajectory: {ylabel}")
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=120)
    plt.close(fig)
    print(f"  wrote {filename}")


# ---------------------------------------------------------------------------
# Optimizer comparison plots
# ---------------------------------------------------------------------------

def plot_optimizer_metric_box(metric_key: str, ylabel: str, filename: str,
                                 log_y: bool = False,
                                 layer_filter=None) -> None:
    """For each optimizer, plot a box/violin of `metric_key` across all matrices
    in all seed checkpoints, optionally filtered by layer_type."""
    runs = load_optimizer()
    if layer_filter is None:
        layer_filter = lambda lt, routing: routing == "muon"

    # Aggregate: optimizer -> list of values
    by_opt = defaultdict(list)
    for r in runs:
        opt = r["optimizer"]
        for name, mat in r["matrices"].items():
            if not layer_filter(mat["layer_type"], mat["routing"]):
                continue
            v = mat["metrics"].get(metric_key)
            if v is None or (isinstance(v, float) and math.isinf(v)):
                continue
            by_opt[opt].append(float(v))

    opts = ["muon", "muon_vs", "mousse", "adam"]
    data = [by_opt.get(o, []) for o in opts]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    bp = ax.boxplot(data, tick_labels=opts, patch_artist=True,
                    showmeans=True, meanprops={"marker": "x", "markeredgecolor": "k"})
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(COLORS.get(opts[i], "#cccccc"))
        patch.set_alpha(0.6)
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("optimizer")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} across all Muon-routed matrices (31.5M @ 10k, 3 seeds pooled)")
    ax.grid(True, axis="y", alpha=0.3)
    # Annotate medians
    for i, d in enumerate(data):
        if d:
            ax.text(i + 1, np.median(d), f"n={len(d)}", ha="center", va="bottom",
                    fontsize=7, color="black")
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=120)
    plt.close(fig)
    print(f"  wrote {filename}")


def plot_optimizer_by_layertype(metric_key: str, ylabel: str, filename: str,
                                  log_y: bool = False,
                                  ylim: Tuple[float, float] = None) -> None:
    """Group by (layer_type, optimizer); median + IQR."""
    runs = load_optimizer()
    layer_types = ["mlp_w1", "mlp_w2", "mlp_w3",
                   "mamba_out_proj",
                   "in_proj_z", "in_proj_x",
                   "in_proj_B", "in_proj_C",
                   "in_proj_dd_dt", "in_proj_trap",
                   "in_proj_angle", "in_proj_dd_A"]
    opts = ["muon", "muon_vs", "mousse", "adam"]

    # (layer_type, opt) -> list of values
    agg = defaultdict(list)
    for r in runs:
        opt = r["optimizer"]
        for name, mat in r["matrices"].items():
            v = mat["metrics"].get(metric_key)
            if v is None or (isinstance(v, float) and math.isinf(v)):
                continue
            agg[(mat["layer_type"], opt)].append(float(v))

    fig, ax = plt.subplots(figsize=(11, 5.6))
    x = np.arange(len(layer_types))
    w = 0.2
    for i, opt in enumerate(opts):
        means = []
        stds = []
        for lt in layer_types:
            vals = agg.get((lt, opt), [])
            if vals:
                means.append(float(np.mean(vals)))
                stds.append(float(np.std(vals)))
            else:
                means.append(np.nan)
                stds.append(0)
        ax.bar(x + i * w, means, w, yerr=stds, capsize=2,
               label=opt, color=COLORS.get(opt, None), alpha=0.8)
    if log_y:
        ax.set_yscale("log")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xticks(x + 1.5 * w)
    ax.set_xticklabels(layer_types, rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} by layer type × optimizer (31.5M @ 10k, mean ± std over 3 seeds × blocks)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=120)
    plt.close(fig)
    print(f"  wrote {filename}")


# ---------------------------------------------------------------------------
# Per-layer seed variance check (for the final claims)
# ---------------------------------------------------------------------------

def plot_optimizer_seed_variance(metric_key: str, ylabel: str, filename: str) -> None:
    runs = load_optimizer()
    layer_types = ["mlp_w1", "mamba_out_proj", "in_proj_z", "in_proj_x"]
    opts = ["muon", "muon_vs", "mousse", "adam"]
    seeds = [42, 137, 2024]

    fig, axes = plt.subplots(1, len(layer_types), figsize=(4 * len(layer_types), 4.2),
                              sharey=True)
    for ax, lt in zip(axes, layer_types):
        # (opt, seed) -> list of values across blocks
        for i, opt in enumerate(opts):
            series = []
            for s in seeds:
                r = next((x for x in runs if x["optimizer"] == opt and x["seed"] == s), None)
                if r is None:
                    continue
                vals = [m["metrics"].get(metric_key)
                        for m in r["matrices"].values()
                        if m["layer_type"] == lt]
                vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isinf(v))]
                if vals:
                    series.append((s, float(np.mean(vals))))
            xs = [p[0] for p in series]
            ys = [p[1] for p in series]
            ax.plot(xs, ys, "-o", color=COLORS.get(opt), label=opt, markersize=6)
        ax.set_title(lt)
        ax.set_xlabel("seed")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(ylabel)
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle(f"Per-seed mean of {ylabel} by optimizer (within-block mean)", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {filename}")


# ---------------------------------------------------------------------------
# Sigma-CDF comparison on matched weight matrix
# ---------------------------------------------------------------------------

def plot_sigma_cdf_matched(filename: str,
                            target_name: str = "blocks.0.mlp.w1.weight") -> None:
    """Overlay sigma CDFs for the same weight matrix across 4 optimizers, seed=42."""
    runs = load_optimizer()
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for r in runs:
        if r["seed"] != 42:
            continue
        mat = r["matrices"].get(target_name)
        if mat is None:
            continue
        sv = np.array(mat["metrics"]["sv_samples"])
        sv = np.sort(sv)[::-1]  # descending
        # plot CDF: x = sigma, y = fraction of sv >= sigma (standard complementary CDF)
        frac = np.arange(1, len(sv) + 1) / len(sv)
        ax.plot(sv / sv.max(), frac, label=r["optimizer"],
                color=COLORS.get(r["optimizer"]), lw=2)
    ax.set_xscale("log")
    ax.set_xlabel("sigma / sigma_max (log)")
    ax.set_ylabel("CDF")
    ax.set_title(f"Normalized sigma CDF on {target_name} (seed=42)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=120)
    plt.close(fig)
    print(f"  wrote {filename}")


def plot_sigma_histogram_matched(filename: str,
                                   target_name: str = "blocks.0.mlp.w1.weight") -> None:
    """Overlay sigma PDF estimates for matched matrix, 4 optimizers, seed=42."""
    runs = load_optimizer()
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for r in runs:
        if r["seed"] != 42:
            continue
        mat = r["matrices"].get(target_name)
        if mat is None:
            continue
        sv = np.array(mat["metrics"]["sv_samples"])
        sv = sv / sv.max()  # normalize
        ax.hist(sv, bins=40, histtype="step",
                label=r["optimizer"], color=COLORS.get(r["optimizer"]),
                density=True, lw=2)
    ax.set_xlabel("sigma / sigma_max")
    ax.set_ylabel("density (normalized)")
    ax.set_title(f"sigma distribution on {target_name} (seed=42)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=120)
    plt.close(fig)
    print(f"  wrote {filename}")


# ---------------------------------------------------------------------------
# Change magnitude heatmap
# ---------------------------------------------------------------------------

def plot_change_magnitude_heatmap(filename: str) -> None:
    changes = load_change()
    changes.sort(key=lambda r: r["step"])
    steps = [r["step"] for r in changes]

    # Get sorted layer order; we'll pick interesting slices
    layer_names = sorted(list(changes[0]["layers"].keys()))
    # Filter to 2D Muon-routed weights + embedding for context
    interesting = [n for n in layer_names
                   if (".in_proj.weight" in n or ".out_proj.weight" in n
                       or ".mlp.w" in n or n in ("tok_emb.weight",))]
    # Keep reasonable ordering
    def sort_key(n):
        if n == "tok_emb.weight":
            return (-1, 0, n)
        if not n.startswith("blocks."):
            return (100, 0, n)
        idx = int(n.split(".")[1])
        if ".mamba_fwd" in n:
            sub = 0
        elif ".mamba_bwd" in n:
            sub = 1
        else:
            sub = 2
        return (idx, sub, n)
    interesting.sort(key=sort_key)

    # Two panels: rel_change_from_init (approx init) and rel_change_from_10k
    for metric_key, title, fname_suffix in [
        ("rel_change_from_init",
         "Per-layer ||W - W_init||_F / ||W_init||_F (approx-init proxy)",
         "_from_init"),
        ("rel_change_from_10k",
         "Per-layer ||W - W_10k||_F / ||W_10k||_F  (step-10k reference)",
         "_from_10k"),
    ]:
        mat = np.zeros((len(interesting), len(steps)))
        for j, r in enumerate(changes):
            for i, n in enumerate(interesting):
                mat[i, j] = r["layers"].get(n, {}).get(metric_key, np.nan)

        fig, ax = plt.subplots(figsize=(7.5, 10))
        im = ax.imshow(mat, aspect="auto", cmap="viridis",
                       interpolation="nearest")
        ax.set_xticks(range(len(steps)))
        ax.set_xticklabels([str(s) for s in steps])
        ax.set_yticks(range(len(interesting)))
        short = [n.replace("blocks.", "b").replace(".weight", "")
                    .replace("mamba_fwd", "mf").replace("mamba_bwd", "mb")
                    .replace("in_proj", "ip").replace("out_proj", "op")
                  for n in interesting]
        ax.set_yticklabels(short, fontsize=6)
        cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
        cb.set_label(metric_key)
        ax.set_xlabel("step")
        ax.set_title(title)
        fig.tight_layout()
        fname = filename.replace(".png", f"{fname_suffix}.png")
        fig.savefig(OUT_DIR / fname, dpi=120)
        plt.close(fig)
        print(f"  wrote {fname}")


# ---------------------------------------------------------------------------
# Embedding vs block
# ---------------------------------------------------------------------------

def plot_emb_vs_block_spectra(filename: str) -> None:
    d = json.load(open(OUT_DIR / "embedding_vs_block.json"))
    mats = d["matrices"]

    pick = {
        "tok_emb.weight": "tok_emb (Adam)",
        "pos_emb.weight": "pos_emb (Adam)",
        "blocks.5.mlp.w1.weight": "block 5 mlp.w1 (Muon)",
        "blocks.5.mlp.w3.weight": "block 5 mlp.w3 (Muon)",
        "blocks.5.mamba_fwd.out_proj.weight": "block 5 mamba_fwd.out_proj (Muon)",
        "blocks.5.mamba_fwd.in_proj.weight::x": "block 5 in_proj x-slice (Muon)",
        "blocks.5.mamba_fwd.in_proj.weight::z": "block 5 in_proj z-slice (Muon)",
    }
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for name, label in pick.items():
        if name not in mats:
            continue
        sv = np.array(mats[name]["metrics"]["sv_samples"])
        sv = sv / sv[0]
        frac = np.arange(1, len(sv) + 1) / len(sv)
        # Adam-routed layers dashed, Muon solid
        routing = mats[name]["routing"]
        ls = "--" if routing == "adam" else "-"
        ax.plot(sv, frac, ls, label=label, lw=2)
    ax.set_xscale("log")
    ax.set_xlabel("sigma / sigma_max (log)")
    ax.set_ylabel("CDF")
    ax.set_title("Normalized sigma CDF: Adam-routed (dashed) vs Muon-routed (solid)\n"
                 "within the same 10L640d model @ step 50k")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=120)
    plt.close(fig)
    print(f"  wrote {filename}")


# ---------------------------------------------------------------------------
# Summary stats table as JSON
# ---------------------------------------------------------------------------

def summarize_optimizer() -> None:
    runs = load_optimizer()
    # metric -> optimizer -> list
    metrics = ["svd_entropy", "stable_rank_norm", "sigma_max", "cond_number"]
    summary = {}
    for metric in metrics:
        summary[metric] = {}
        for opt in ["muon", "muon_vs", "mousse", "adam"]:
            vals = []
            for r in runs:
                if r["optimizer"] != opt:
                    continue
                for name, mat in r["matrices"].items():
                    if mat["routing"] != "muon":
                        continue
                    v = mat["metrics"].get(metric)
                    if v is None or (isinstance(v, float) and math.isinf(v)):
                        continue
                    vals.append(float(v))
            if vals:
                summary[metric][opt] = {
                    "n": len(vals),
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "median": float(np.median(vals)),
                    "q25": float(np.percentile(vals, 25)),
                    "q75": float(np.percentile(vals, 75)),
                }
    # Per-seed means for the top 3 metrics (so variance across seeds is exposed)
    seed_breakdown = {}
    for metric in ["svd_entropy", "stable_rank_norm", "sigma_max"]:
        seed_breakdown[metric] = {}
        for opt in ["muon", "muon_vs", "mousse", "adam"]:
            seed_breakdown[metric][opt] = {}
            for s in [42, 137, 2024]:
                vals = []
                for r in runs:
                    if r["optimizer"] != opt or r["seed"] != s:
                        continue
                    for name, mat in r["matrices"].items():
                        if mat["routing"] != "muon":
                            continue
                        v = mat["metrics"].get(metric)
                        if v is None or (isinstance(v, float) and math.isinf(v)):
                            continue
                        vals.append(float(v))
                seed_breakdown[metric][opt][f"s{s}"] = float(np.mean(vals)) if vals else None

    # Paired test: Muon vs Adam per-matrix deltas across 3 seeds
    pair_delta = {}
    for metric in ["svd_entropy", "stable_rank_norm", "sigma_max"]:
        deltas_all = []
        # Build matrix-level aligned vectors by (seed, matrix_name) for muon and adam
        muon_by_key = {}
        adam_by_key = {}
        for r in runs:
            for name, mat in r["matrices"].items():
                if mat["routing"] != "muon":
                    continue
                v = mat["metrics"].get(metric)
                if v is None or (isinstance(v, float) and math.isinf(v)):
                    continue
                key = (r["seed"], name)
                if r["optimizer"] == "muon":
                    muon_by_key[key] = float(v)
                elif r["optimizer"] == "adam":
                    adam_by_key[key] = float(v)
        for key, mv in muon_by_key.items():
            if key in adam_by_key:
                deltas_all.append(mv - adam_by_key[key])
        pair_delta[metric] = {
            "n_pairs": len(deltas_all),
            "mean_delta": float(np.mean(deltas_all)) if deltas_all else None,
            "std_delta": float(np.std(deltas_all)) if deltas_all else None,
            "frac_muon_greater": float(np.mean([d > 0 for d in deltas_all])) if deltas_all else None,
        }

    with open(OUT_DIR / "optimizer_summary.json", "w") as f:
        json.dump({"pooled": summary,
                   "per_seed_mean": seed_breakdown,
                   "paired_muon_minus_adam": pair_delta}, f, indent=2)
    print("  wrote optimizer_summary.json")


def summarize_trajectory() -> None:
    traj = load_trajectory()
    traj.sort(key=lambda r: r["step"])
    out = {"runs": []}
    for r in traj:
        row = {"step": r["step"], "by_layer_type": {}}
        for lt_key in ["mlp_w1", "mlp_w2", "mlp_w3", "mamba_out_proj",
                        "in_proj_z", "in_proj_x", "in_proj_B", "in_proj_C",
                        "in_proj_dd_dt", "in_proj_dd_A", "in_proj_trap",
                        "in_proj_angle", "emb", "pos_emb"]:
            for met in ["svd_entropy", "stable_rank_norm", "sigma_max"]:
                vals = [m["metrics"].get(met) for m in r["matrices"].values()
                        if m["layer_type"] == lt_key]
                vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isinf(v))]
                if not vals:
                    continue
                row["by_layer_type"].setdefault(lt_key, {})[met] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "n": len(vals),
                }
        out["runs"].append(row)
    with open(OUT_DIR / "trajectory_summary.json", "w") as f:
        json.dump(out, f, indent=2)
    print("  wrote trajectory_summary.json")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Trajectory
    plot_trajectory_metric("svd_entropy", "normalized SVD entropy (nats)",
                            "svd_entropy_vs_step.png")
    plot_trajectory_metric("stable_rank_norm", "stable rank / min(m,n)",
                            "stable_rank_vs_step.png")
    plot_trajectory_metric("sigma_max", "sigma_max  (spectral norm)",
                            "sigma_max_vs_step.png", log_y=True)
    plot_trajectory_metric("sigma_min", "sigma_min",
                            "sigma_min_vs_step.png", log_y=True)

    # Optimizer: pooled across all Muon-routed matrices
    plot_optimizer_metric_box("svd_entropy", "normalized SVD entropy",
                                "optimizer_entropy_box.png")
    plot_optimizer_metric_box("stable_rank_norm", "stable rank / min(m,n)",
                                "optimizer_stable_rank_box.png")
    plot_optimizer_metric_box("sigma_max", "sigma_max",
                                "optimizer_sigma_max_box.png", log_y=True)

    # Optimizer: broken out by layer_type
    plot_optimizer_by_layertype("svd_entropy", "SVD entropy",
                                  "optimizer_entropy_by_layertype.png",
                                  ylim=(0.80, 1.0))
    plot_optimizer_by_layertype("stable_rank_norm", "stable rank / min(m,n)",
                                  "optimizer_stable_rank_by_layertype.png")
    plot_optimizer_by_layertype("sigma_max", "sigma_max",
                                  "optimizer_sigma_max_by_layertype.png", log_y=True)

    # Seed variance diagnostic
    plot_optimizer_seed_variance("svd_entropy", "mean SVD entropy (within layer type)",
                                   "optimizer_seed_variance.png")

    # Single-matrix spectra overlay
    plot_sigma_cdf_matched("sigma_cdf_mlp_w1.png",
                            target_name="blocks.0.mlp.w1.weight")
    plot_sigma_cdf_matched("sigma_cdf_mlp_w3.png",
                            target_name="blocks.0.mlp.w3.weight")
    plot_sigma_histogram_matched("sigma_hist_mlp_w1.png",
                                    target_name="blocks.0.mlp.w1.weight")

    # Change magnitude heatmap
    plot_change_magnitude_heatmap("change_magnitude_heatmap.png")

    # Embedding-vs-block within one model
    plot_emb_vs_block_spectra("emb_vs_block_spectra.png")

    # Summary JSONs
    summarize_optimizer()
    summarize_trajectory()


if __name__ == "__main__":
    main()
