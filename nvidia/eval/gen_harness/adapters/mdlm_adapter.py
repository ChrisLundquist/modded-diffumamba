"""Unified MDLM adapter for TransformerV2 and DiffuMambaH.

Both model families output [B, T, vocab_size] from forward(idx, causal=False);
the sampler handles the prefix-conditioned demasking.

Usage:
    adapter = MDLMAdapter.from_spec('d_modern_125m', device='cuda')
    samples = adapter.generate(prefix_ids, cont_len=128)  # LongTensor [B, P+C]
"""

import os
import sys
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(THIS_DIR, '..', '..', '..', 'src'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from transformer_v2 import TransformerV2      # noqa: E402
from hybrid_model import DiffuMambaH           # noqa: E402

HARNESS_DIR = os.path.abspath(os.path.join(THIS_DIR, '..'))
sys.path.insert(0, HARNESS_DIR)
from samplers.mdlm_topk import demask_topk_prefix  # noqa: E402
from samplers.mdlm_maskgit import maskgit_prefix  # noqa: E402

SAMPLERS = {
    'topk': demask_topk_prefix,
    'maskgit': maskgit_prefix,
}


# Model zoo: (factory, checkpoint_path). Paths relative to /mnt/d/code/gpt-slide/
# which is CWD for existing scripts.
CHECKPOINT_ROOT = '/mnt/d/code/gpt-slide'

MODEL_SPECS = {
    'd_modern_125m': dict(
        family='transformer_v2',
        ckpt='muon_exp/outputs/125m_10b_dmodern/checkpoint_72479.pt',
        kwargs=dict(vocab_size=50258, n_layer=12, n_head=12, n_embd=768,
                    use_rope=True, use_swiglu=True, use_qk_norm=True),
    ),
    'd_modern_30m': dict(
        family='transformer_v2',
        ckpt='muon_exp/outputs/transformer_converge_v3/checkpoint_56000.pt',
        kwargs=dict(vocab_size=50258, n_layer=6, n_head=6, n_embd=384,
                    use_rope=True, use_swiglu=True),
    ),
    'd_modern_125m_papl_45k': dict(
        family='transformer_v2',
        ckpt='muon_exp/outputs/125m_papl_finetune/checkpoint_45000.pt',
        kwargs=dict(vocab_size=50258, n_layer=12, n_head=12, n_embd=768,
                    use_rope=True, use_swiglu=True, use_qk_norm=True),
    ),
    'd_modern_125m_papl_50k': dict(
        family='transformer_v2',
        ckpt='muon_exp/outputs/125m_papl_finetune/checkpoint_50000.pt',
        kwargs=dict(vocab_size=50258, n_layer=12, n_head=12, n_embd=768,
                    use_rope=True, use_swiglu=True, use_qk_norm=True),
    ),
    'd_modern_125m_30k': dict(   # vanilla MDLM at step 30k — generation peak (NLL min)
        family='transformer_v2',
        ckpt='muon_exp/outputs/125m_10b_dmodern/checkpoint_30000.pt',
        kwargs=dict(vocab_size=50258, n_layer=12, n_head=12, n_embd=768,
                    use_rope=True, use_swiglu=True, use_qk_norm=True),
    ),
    'd_modern_125m_40k': dict(   # vanilla MDLM at step 40k — distinct_4 peak
        family='transformer_v2',
        ckpt='muon_exp/outputs/125m_10b_dmodern/checkpoint_40000.pt',
        kwargs=dict(vocab_size=50258, n_layer=12, n_head=12, n_embd=768,
                    use_rope=True, use_swiglu=True, use_qk_norm=True),
    ),
    'd_modern_125m_50k': dict(   # vanilla MDLM at step 50k — regression baseline
        family='transformer_v2',
        ckpt='muon_exp/outputs/125m_10b_dmodern/checkpoint_50000.pt',
        kwargs=dict(vocab_size=50258, n_layer=12, n_head=12, n_embd=768,
                    use_rope=True, use_swiglu=True, use_qk_norm=True),
    ),
    # Objective-ablation fine-tunes (Exps 1-3 in experiments_plan.md)
    'd_modern_125m_baseline_50k': dict(
        family='transformer_v2',
        ckpt='muon_exp/outputs/125m_baseline_finetune/checkpoint_50000.pt',
        kwargs=dict(vocab_size=50258, n_layer=12, n_head=12, n_embd=768,
                    use_rope=True, use_swiglu=True, use_qk_norm=True),
    ),
    'd_modern_125m_gamma_decay_50k': dict(
        family='transformer_v2',
        ckpt='muon_exp/outputs/125m_gamma_decay_finetune/checkpoint_50000.pt',
        kwargs=dict(vocab_size=50258, n_layer=12, n_head=12, n_embd=768,
                    use_rope=True, use_swiglu=True, use_qk_norm=True),
    ),
    'd_modern_125m_t_curr_50k': dict(
        family='transformer_v2',
        ckpt='muon_exp/outputs/125m_t_curriculum_finetune/checkpoint_50000.pt',
        kwargs=dict(vocab_size=50258, n_layer=12, n_head=12, n_embd=768,
                    use_rope=True, use_swiglu=True, use_qk_norm=True),
    ),
    'd_modern_125m_papl_t_curr_50k': dict(
        family='transformer_v2',
        ckpt='muon_exp/outputs/125m_papl_t_curr_finetune/checkpoint_50000.pt',
        kwargs=dict(vocab_size=50258, n_layer=12, n_head=12, n_embd=768,
                    use_rope=True, use_swiglu=True, use_qk_norm=True),
    ),
    'mamba3_30m': dict(
        family='diffumamba_h',
        ckpt='muon_exp/outputs/mamba3_converge/checkpoint_56000.pt',
        kwargs=dict(n_embd=384, n_head=6, n_mamba=6, n_attn=0,
                    attn_positions=set(), mamba_version=3, d_state=32),
    ),
}


class MDLMAdapter:
    def __init__(self, name, model, device='cuda'):
        self.name = name
        self.model = model
        self.device = device

    @classmethod
    def from_spec(cls, name, device='cuda'):
        if name not in MODEL_SPECS:
            raise KeyError(f'Unknown model spec: {name}. Known: {list(MODEL_SPECS)}')
        spec = MODEL_SPECS[name]
        if spec['family'] == 'transformer_v2':
            model = TransformerV2(**spec['kwargs']).to(device)
        elif spec['family'] == 'diffumamba_h':
            model = DiffuMambaH(**spec['kwargs']).to(device)
        else:
            raise ValueError(f'Unknown family: {spec["family"]}')
        ckpt_path = os.path.join(CHECKPOINT_ROOT, spec['ckpt'])
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        del ckpt
        torch.cuda.empty_cache()
        return cls(name, model, device)

    @torch.no_grad()
    def generate(self, prefix_ids, cont_len=128, top_k=50, temperature=1.0,
                 n_steps=None, sampler='topk', **sampler_kwargs):
        """prefix_ids: LongTensor [B, P]. Returns LongTensor [B, P + cont_len].

        sampler: 'topk' (default, pre-nucleus ordering, one-shot) or 'maskgit'
            (iterative confidence-based remasking).
        """
        if sampler not in SAMPLERS:
            raise KeyError(f'Unknown sampler: {sampler}. Available: {list(SAMPLERS)}')
        fn = SAMPLERS[sampler]
        kwargs = dict(top_k=top_k, temperature=temperature, device=self.device)
        # Forward n_steps consistently: only set when the caller actually
        # passes a value, so each sampler's own default takes effect when
        # n_steps is None (mdlm_topk → cont_len//2; maskgit → 12).
        if n_steps is not None:
            kwargs['n_steps'] = n_steps
        # sampler_kwargs takes precedence over the above defaults — explicit
        # caller intent wins (e.g. confidence='gumbel' for MaskGIT).
        kwargs.update(sampler_kwargs)
        return fn(self.model, prefix_ids, cont_len, **kwargs)

    def unload(self):
        del self.model
        torch.cuda.empty_cache()
