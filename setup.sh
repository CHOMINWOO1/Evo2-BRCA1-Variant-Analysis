#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source ./env.sh
if [[ ! -x .tools/bin/uv ]]; then
    python3 -m pip install --target "$PWD/.tools" uv==0.12.11
fi
if [[ ! -x .venv/bin/python ]]; then
    .tools/bin/uv --system-certs venv --python 3.12 .venv
fi
.tools/bin/uv --system-certs pip install --python .venv/bin/python torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
.tools/bin/uv --system-certs pip install --python .venv/bin/python evo2==0.6.0 packaging ninja psutil setuptools wheel \
  'https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2%2Bcu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl'
