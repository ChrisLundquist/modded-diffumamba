#!/bin/bash
# Setup script for AMD RX 9070 XT (RDNA 4 / gfx1201)
# Requires: ROCm 7.2+, Linux kernel 6.13.5+
# Uses uv for fast dependency management

set -e

echo "=== DiffuMamba3 ROCm Setup ==="

# Check ROCm
if ! command -v rocminfo &> /dev/null; then
    echo "ERROR: ROCm not found. Install ROCm 7.2+ first."
    echo "  https://rocm.docs.amd.com/projects/install-on-linux/en/latest/"
    exit 1
fi

echo "ROCm GPU info:"
rocminfo | grep -E "gfx|Name" | head -4

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "=== Installing uv ==="
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.local/bin/env" 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"
fi

echo "uv: $(uv --version)"

# Environment variables for RDNA 4
export FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"
export PYTORCH_TUNABLEOP_ENABLED="1"
export PYTORCH_TUNABLEOP_TUNING_DURATION="short"

# Create venv with uv
echo "=== Creating venv ==="
uv venv --python 3.12

# Detect ROCm version
ROCM_VER=$(cat /opt/rocm/.info/version 2>/dev/null | cut -d. -f1,2 || echo "")
echo "Detected ROCm: ${ROCM_VER:-unknown}"

# Install PyTorch with ROCm
echo "=== Installing PyTorch with ROCm ==="
if [[ "$ROCM_VER" == "7."* ]]; then
    echo "Using PyTorch nightly for ROCm 7.x"
    uv pip install --pre torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/nightly/rocm7.2
else
    echo "Using PyTorch stable for ROCm 6.3"
    uv pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/rocm6.3
fi

# Install other deps (fast with uv)
echo "=== Installing dependencies ==="
uv pip install tiktoken datasets huggingface_hub wandb numpy

# Install Mamba-3 from source (needs build)
echo "=== Installing Mamba-3 from source (this takes a while) ==="
MAMBA_FORCE_BUILD=TRUE uv pip install --no-cache --reinstall \
    "mamba_ssm @ git+https://github.com/state-spaces/mamba.git" \
    --no-build-isolation

# Verify
echo "=== Verification ==="
uv run python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'ROCm available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    x = torch.randn(512, 512, device='cuda', dtype=torch.bfloat16)
    y = x @ x.T
    print(f'BF16 matmul test: OK ({y.shape})')

try:
    from mamba_ssm.modules.mamba3 import Mamba3
    m = Mamba3(d_model=256, d_state=64, headdim=32, dtype=torch.bfloat16)
    print(f'Mamba-3 import: OK')
except Exception as e:
    print(f'Mamba-3 import: FAILED ({e})')
    print(f'  Falling back to SimpleSSM for testing')
"

echo ""
echo "=== Setup complete ==="
echo "Activate with: source .venv/bin/activate"
echo "Or run directly: uv run python3 src/train.py --config small"
echo ""
echo "Add to your shell profile:"
echo '  export FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"'
echo '  export PYTORCH_TUNABLEOP_ENABLED="1"'
