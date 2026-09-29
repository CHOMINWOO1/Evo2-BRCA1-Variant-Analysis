#!/usr/bin/env bash
# Source this file before installing packages or running an example.
export EVO2_WORKSPACE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export XDG_CACHE_HOME="$EVO2_WORKSPACE/.cache"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export UV_CACHE_DIR="$XDG_CACHE_HOME/uv"
export UV_PYTHON_INSTALL_DIR="$XDG_CACHE_HOME/python"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_XET_CACHE="$HF_HOME/xet"
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch_extensions"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export TMPDIR="$EVO2_WORKSPACE/.tmp"
export PYTHONPYCACHEPREFIX="$XDG_CACHE_HOME/pycache"
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=4
export MAX_JOBS=4
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
if [[ -f "$EVO2_WORKSPACE/.venv/bin/activate" ]]; then
    source "$EVO2_WORKSPACE/.venv/bin/activate"
fi
