"""Sample from large-model checkpoints and compare to quokka."""
import sys
import copy
import time
from pathlib import Path
import torch
import tiktoken
sys.path.insert(0, '.')
from model import DiffuMamba3, CONFIGS

CKPT_DIR = Path("checkpoints")
enc = tiktoken.get_encoding("gpt2")
MASK_ID = 50257

device = "cuda"

CHECKPOINTS = [
    # Large model checkpoints (10L×640d)
    ("10L640d_10k",                    (10, 640), "10L×640d Phase 1 (10k steps)"),
    ("10L640d_50k_step10000",          (10, 640), "10L×640d Phase 2 @ step 10k"),
    ("10L640d_50k_step30000",          (10, 640), "10L×640d Phase 2 @ step 30k"),
    ("10L640d_50k",                    (10, 640), "10L×640d Phase 2 BEST (50k done)"),
    ("10L640d_100k_step10000",         (10, 640), "10L×640d Phase 3 @ step 10k"),
    ("10L640d_100k",                   (10, 640), "10L×640d Phase 3 partial best"),
    # Quokka reference
    ("final10k_new_best_s42",          (4, 384),  "quokka 10k (new_best s42)"),
]

for ckpt_name, (n_layers, d_model), desc in CHECKPOINTS:
    ckpt_path = CKPT_DIR / f"{ckpt_name}.pt"
    if not ckpt_path.exists():
        print(f"[SKIP] {ckpt_name}")
        continue

    cfg = copy.deepcopy(CONFIGS["quokka"])
    cfg.n_layers = n_layers
    cfg.d_model = d_model

    model = DiffuMamba3(cfg).to(device, dtype=torch.bfloat16)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    torch.manual_seed(0)
    tokens = model.sample(batch_size=3, seq_len=128, num_steps=128, device=device, temperature=0.8)

    print(f"\n{'='*70}")
    print(f"  {ckpt_name}")
    print(f"  {desc}")
    print(f"{'='*70}")
    for i in range(3):
        ids = [t for t in tokens[i].cpu().tolist() if t != MASK_ID and t < 50257]
        text = enc.decode(ids).replace("\n", " ")[:280]
        print(f"\n  [{i+1}] {text}")

    del model
    torch.cuda.empty_cache()
