#!/usr/bin/env bash
# Swap the CPU `paddlepaddle` wheel for the CUDA build so PaddleOCR 3.x runs on the GPU.
#
# Why a script: paddlepaddle-gpu pins exact nvidia-cudnn/cublas versions that conflict with
# the torch CUDA 12.8 wheels in requirements.txt, so pip/uv would resolve to an ancient
# 2.6.x release. Installing the wheel with --no-deps lets Paddle reuse torch's CUDA libraries
# (verified with torch 2.11+cu128, ctranslate2 4.8, paddleocr 3.4: ~0.08 s/frame, ~1.2 GB VRAM).
#
# Usage: scripts/install_paddle_gpu.sh [path/to/python]   (default: .venv/bin/python)
# Re-run it after any `pip install -r requirements.txt`, which reinstalls the CPU wheel.
set -euo pipefail

PY="${1:-.venv/bin/python}"
VERSION="${PYSCRIBE_PADDLE_GPU_VERSION:-3.3.1}"
INDEX="${PYSCRIBE_PADDLE_GPU_INDEX:-https://www.paddlepaddle.org.cn/packages/stable/cu126/}"

if [[ ! -x "$PY" ]]; then
  echo "Python not found or not executable: $PY" >&2
  exit 1
fi

uv pip uninstall --python "$PY" paddlepaddle paddlepaddle-gpu || true
uv pip install --python "$PY" --no-deps --index-url "$INDEX" "paddlepaddle-gpu==${VERSION}"

"$PY" -c "import paddle; assert paddle.is_compiled_with_cuda(), 'CUDA build not active'; print('paddle', paddle.__version__, 'CUDA build OK')"
