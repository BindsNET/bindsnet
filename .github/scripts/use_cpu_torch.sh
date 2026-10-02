#!/usr/bin/env bash
# CI runners have no GPU. The CUDA torch build pinned in poetry.lock sometimes
# segfaults while preloading its CUDA libraries on these runners, so swap in the
# CPU build of the same torch/torchvision versions after `poetry install`.
# Usage: use_cpu_torch.sh <python command>, e.g. "python" or "poetry run python".
set -euo pipefail
PY=${1:-python}

# Read versions from package metadata; importing torch here could crash.
pins=$($PY - <<'EOF'
from importlib.metadata import version
print(" ".join(f"{p}=={version(p).split('+')[0]}" for p in ("torch", "torchvision")))
EOF
)
$PY -m pip install --no-deps --force-reinstall \
    --index-url https://download.pytorch.org/whl/cpu $pins
$PY -c "import torch; assert torch.version.cuda is None, torch.__version__; print('torch', torch.__version__)"
