"""Compare temperature × top-p grid on the 10L×640d 50k checkpoint."""
import sys
import copy
import torch
import tiktoken
from pathlib import Path
sys.path.insert(0, '.')
from model import DiffuMamba3, CONFIGS

enc = tiktoken.get_encoding("gpt2")
MASK_ID = 50257
device = "cuda"

cfg = copy.deepcopy(CONFIGS["quokka"])
cfg.n_layers = 10
cfg.d_model = 640

model = DiffuMamba3(cfg).to(device, dtype=torch.bfloat16)
model.load_state_dict(torch.load("checkpoints/10L640d_50k.pt", map_location=device, weights_only=True))
model.eval()

configs = [
    (1.0, 1.0, 0),    # no truncation (baseline after bug fix)
    (0.9, 1.0, 0),    # lower temperature
    (1.0, 0.9, 0),    # top-p 0.9
    (1.0, 0.95, 0),   # top-p 0.95
    (0.9, 0.9, 0),    # both
    (1.0, 1.0, 50),   # top-k 50
]

for t, p, k in configs:
    torch.manual_seed(0)
    tokens = model.sample(batch_size=3, seq_len=128, num_steps=128,
                          device=device, temperature=t, top_p=p, top_k=k)
    print(f"\n{'='*70}")
    print(f"  temperature={t}, top_p={p}, top_k={k}")
    print(f"{'='*70}")
    for i in range(3):
        ids = [tok for tok in tokens[i].cpu().tolist() if tok != MASK_ID and tok < 50257]
        text = enc.decode(ids).replace("\n", " ")[:220]
        print(f"\n  [{i+1}] {text}")
