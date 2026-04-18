"""AR baseline adapter.

Wraps an HuggingFace causal LM as a unified generate(prefix_ids, cont_len) API
compatible with the MDLM harness. Handles the different tokenizer (GPT-2 both
here and in our MDLM, so prefix_ids transfer directly).

Supported specs:
  rhysjones_gpt2_124m_fineweb_edu  — 124M GPT-2, FineWeb-Edu 10B, llm.c-trained
  gpt2                              — GPT-2 small (WebText), reference only
"""

import os
import sys
import time
import torch
import torch.nn.functional as F

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

AR_SPECS = {
    'rhysjones_gpt2_124m_fineweb_edu': dict(
        path='/home/clundquist/muon_data/hf_models/rhysjones_gpt2-124M',
        hf_id='rhysjones/gpt2-124M-edu-fineweb-10B',
    ),
    'gpt2_small_webtext': dict(
        path=None,  # hub fallback
        hf_id='gpt2',
    ),
}


class ARAdapter:
    def __init__(self, name, model, device='cuda'):
        self.name = name
        self.model = model
        self.device = device

    @classmethod
    def from_spec(cls, name, device='cuda', dtype=torch.float32):
        if name not in AR_SPECS:
            raise KeyError(f'Unknown AR spec: {name}. Known: {list(AR_SPECS)}')
        spec = AR_SPECS[name]
        from transformers import GPT2LMHeadModel
        path = spec['path'] if spec['path'] and os.path.isdir(spec['path']) else spec['hf_id']
        model = GPT2LMHeadModel.from_pretrained(path, torch_dtype=dtype).to(device).eval()
        return cls(name, model, device)

    @torch.no_grad()
    def generate(self, prefix_ids, cont_len=128, top_k=50, temperature=1.0, n_steps=None):
        """Causal AR continuation. n_steps ignored (AR is strictly sequential).

        Returns LongTensor [B, P + cont_len] on CPU with prefix preserved,
        cont_len new tokens sampled one at a time with top-k + temperature.
        """
        B, P = prefix_ids.shape
        device = self.device
        ids = prefix_ids.to(device).long().clone()

        # KV-cache-free greedy loop (simple + robust; enough for 128 tokens)
        for _ in range(cont_len):
            logits = self.model(ids).logits                           # [B, T, V]
            next_logits = logits[:, -1, :].float()                    # [B, V]
            # temperature + top-k
            scaled = next_logits / temperature
            if top_k is not None and top_k > 0:
                tv, ti = scaled.topk(top_k, dim=-1)
                filtered = torch.full_like(scaled, -float('inf'))
                filtered.scatter_(-1, ti, tv)
                scaled = filtered
            probs = F.softmax(scaled, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)       # [B, 1]
            ids = torch.cat([ids, next_tok], dim=1)

        return ids.cpu()

    def unload(self):
        del self.model
        torch.cuda.empty_cache()
