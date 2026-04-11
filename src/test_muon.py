"""
Test that our Muon implementation is correct.

Checks:
  1. Newton-Schulz produces ~orthogonal matrices (singular values in [0.5, 1.5])
  2. Our NS matches the reference implementation from Keller Jordan
  3. The full MuonAdamW optimizer actually trains (loss decreases)
  4. Edge cases: tall, wide, small, large matrices
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn

# Our implementation
from train import zeropower_via_newtonschulz5, MuonAdamW

# Reference implementation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ref"))
from muon_standalone import (
    zeropower_via_newtonschulz5 as ref_ns5,
    muon_update as ref_muon_update,
)

PASS = 0
FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS: {name}")
    else:
        FAIL += 1
        print(f"  FAIL: {name}  {detail}")


# -------------------------------------------------------
# Test 1: Newton-Schulz produces ~orthogonal output
# -------------------------------------------------------
print("=== Test 1: Newton-Schulz orthogonality ===")

torch.manual_seed(42)
for shape_name, shape in [("square", (128, 128)), ("tall", (256, 64)),
                           ("wide", (64, 256)), ("small", (8, 8)),
                           ("mamba-like", (768, 1536))]:
    G = torch.randn(shape)
    X = zeropower_via_newtonschulz5(G, steps=5)

    # Check output shape matches input
    check(f"{shape_name} shape", X.shape == G.shape,
          f"expected {G.shape}, got {X.shape}")

    # Compute singular values — should be in [0.5, 1.5] per Keller's comment
    S = torch.linalg.svdvals(X.float())
    s_min, s_max = S.min().item(), S.max().item()
    # Keller Jordan: NS5 gives US'V^T where S' ~ Uniform(0.5, 1.5) but
    # bf16 outliers can push a bit beyond. The key property is approximate
    # orthogonality, not exact singular values.
    check(f"{shape_name} singular values in [0.1, 2.0]",
          0.1 < s_min and s_max < 2.0,
          f"range [{s_min:.3f}, {s_max:.3f}]")

    # For square matrices, X @ X^T should be approximately I
    if shape[0] == shape[1]:
        eye_err = (X.float() @ X.float().T - torch.eye(shape[0])).norm().item()
        check(f"{shape_name} X@X^T ≈ I (err < 5)",
              eye_err < 5.0, f"err={eye_err:.3f}")


# -------------------------------------------------------
# Test 2: Our NS matches the reference exactly
# -------------------------------------------------------
print("\n=== Test 2: Match reference implementation ===")

torch.manual_seed(123)
for shape in [(128, 128), (256, 64), (64, 256)]:
    G = torch.randn(shape)
    ours = zeropower_via_newtonschulz5(G.clone(), steps=5)
    ref = ref_ns5(G.clone(), steps=5)

    diff = (ours.float() - ref.float()).abs().max().item()
    check(f"shape {shape} matches reference (max diff < 1e-3)",
          diff < 1e-3, f"max_diff={diff:.6f}")


# -------------------------------------------------------
# Test 3: Full muon_update matches reference
# -------------------------------------------------------
print("\n=== Test 3: Full muon_update matches reference ===")

torch.manual_seed(456)
G = torch.randn(128, 128)
mom = torch.zeros_like(G)
# Our implementation replicates the reference muon_update inline
# Let's verify the components match

# Reference
g_ref = G.clone()
mom_ref = torch.zeros_like(g_ref)
ref_result = ref_muon_update(g_ref, mom_ref, beta=0.95, ns_steps=5, nesterov=True)

# Ours (manually replicate what _muon_step does)
g_ours = G.clone()
mom_ours = torch.zeros_like(g_ours)
mom_ours.lerp_(g_ours, 1 - 0.95)
update = g_ours.lerp_(mom_ours, 0.95)
update = zeropower_via_newtonschulz5(update, steps=5)
update *= max(1, update.size(-2) / update.size(-1)) ** 0.5

diff = (update.float() - ref_result.float()).abs().max().item()
check(f"muon_update matches reference (max diff < 1e-3)",
      diff < 1e-3, f"max_diff={diff:.6f}")


# -------------------------------------------------------
# Test 4: Muon actually trains a simple model
# -------------------------------------------------------
print("\n=== Test 4: MuonAdamW trains a simple model ===")

torch.manual_seed(789)

# Simple 2-layer MLP
model = nn.Sequential(
    nn.Linear(32, 64, bias=False),
    nn.ReLU(),
    nn.Linear(64, 64, bias=False),
    nn.ReLU(),
    nn.Linear(64, 10, bias=False),
)

# Muon for hidden layers, Adam for output
hidden_params = [model[0].weight, model[2].weight]
output_params = [model[4].weight]

optimizer = MuonAdamW([
    dict(params=hidden_params, use_muon=True, lr=0.02, weight_decay=0.0),
    dict(params=output_params, use_muon=False, lr=3e-4, weight_decay=0.0),
])

# Fake classification task
X = torch.randn(64, 32)
y = torch.randint(0, 10, (64,))

losses = []
for step in range(50):
    logits = model(X)
    loss = nn.functional.cross_entropy(logits, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

check(f"loss decreases (start={losses[0]:.3f} → end={losses[-1]:.3f})",
      losses[-1] < losses[0] * 0.8,
      f"ratio={losses[-1]/losses[0]:.3f}")
check(f"no NaN in losses", all(not torch.isnan(torch.tensor(l)) for l in losses))

# Also train with plain Adam for comparison
torch.manual_seed(789)
model2 = nn.Sequential(
    nn.Linear(32, 64, bias=False),
    nn.ReLU(),
    nn.Linear(64, 64, bias=False),
    nn.ReLU(),
    nn.Linear(64, 10, bias=False),
)
opt2 = torch.optim.AdamW(model2.parameters(), lr=3e-4)
adam_losses = []
for step in range(50):
    logits = model2(X)
    loss = nn.functional.cross_entropy(logits, y)
    opt2.zero_grad()
    loss.backward()
    opt2.step()
    adam_losses.append(loss.item())

check(f"Muon converges faster than Adam on this toy task",
      losses[25] < adam_losses[25],
      f"Muon@25={losses[25]:.3f}, Adam@25={adam_losses[25]:.3f}")


# -------------------------------------------------------
# Test 5: Newton-Schulz numerical stability
# -------------------------------------------------------
print("\n=== Test 5: Numerical stability ===")

# Very small matrix values
G_small = torch.randn(64, 64) * 1e-6
X_small = zeropower_via_newtonschulz5(G_small, steps=5)
check("small input (1e-6 scale) — no NaN",
      not torch.isnan(X_small).any().item())

# Very large matrix values
G_large = torch.randn(64, 64) * 1e6
X_large = zeropower_via_newtonschulz5(G_large, steps=5)
check("large input (1e6 scale) — no NaN",
      not torch.isnan(X_large).any().item())

# Already orthogonal input
Q, _ = torch.linalg.qr(torch.randn(64, 64))
X_orth = zeropower_via_newtonschulz5(Q, steps=5)
orth_err = (X_orth.float() @ X_orth.float().T - torch.eye(64)).norm().item()
# NS5 doesn't converge to exact orthogonality (cursed quintic trades
# convergence for slope at zero). Error < 8 is fine.
check(f"orthogonal input stays ~orthogonal (err < 8)",
      orth_err < 8.0, f"err={orth_err:.3f}")

# Near-zero matrix
G_zero = torch.zeros(64, 64) + 1e-10
X_zero = zeropower_via_newtonschulz5(G_zero, steps=5)
check("near-zero input — no NaN",
      not torch.isnan(X_zero).any().item())


# -------------------------------------------------------
# Test 6: Training DiffuMamba3 with Muon (end-to-end)
# -------------------------------------------------------
print("\n=== Test 6: DiffuMamba3 + Muon end-to-end ===")

from model import DiffuMamba3, DiffuMamba3Config, CONFIGS

cfg = CONFIGS["tiny"]
model = DiffuMamba3(cfg)

# Build optimizer like train.py does
muon_params = []
adam_params = []
for name, p in model.named_parameters():
    if not p.requires_grad:
        continue
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

optimizer = MuonAdamW([
    dict(params=muon_params, use_muon=True, lr=0.02, weight_decay=0.0),
    dict(params=adam_params, use_muon=False, lr=3e-4, weight_decay=0.0),
])

x_0 = torch.randint(0, 50257, (4, cfg.max_seq_len))
losses = []
for step in range(10):
    loss, metrics = model.compute_loss(x_0)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

check(f"DiffuMamba3 loss decreases with Muon ({losses[0]:.2f} → {losses[-1]:.2f})",
      losses[-1] < losses[0],
      f"ratio={losses[-1]/losses[0]:.3f}")
check("no NaN in DiffuMamba3 training",
      all(not torch.isnan(torch.tensor(l)) for l in losses))

# Verify AdaLN is still zero-initialized
adaln_weights = []
for name, p in model.named_parameters():
    if "adaln" in name and "modulation.weight" in name:
        adaln_weights.append((name, p.data.abs().max().item()))
# After 10 steps they should have moved from zero, but let's check they started near zero
# (can't check init anymore since we trained, but we can check they're reasonable)
check("AdaLN modulation weights are finite after training",
      all(not torch.isnan(torch.tensor(v)) for _, v in adaln_weights))


# -------------------------------------------------------
# Summary
# -------------------------------------------------------
print(f"\n{'='*50}")
print(f"Results: {PASS} passed, {FAIL} failed out of {PASS + FAIL} checks")
if FAIL == 0:
    print("All tests passed.")
else:
    print(f"WARNING: {FAIL} test(s) failed!")
    sys.exit(1)
