#!/usr/bin/env bash
set -euo pipefail
export PIP_REQUIRE_VIRTUALENV=true
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_ROOT="${INFANTPOSE_ENV_ROOT:-$HOME/.local/share/infantpose/venvs}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
command -v "$PYTHON_BIN" >/dev/null || { echo "Need $PYTHON_BIN plus python3.12-venv (Ubuntu 24.04 package)." >&2; exit 2; }
"$PYTHON_BIN" -c 'import sys; assert sys.version_info >= (3,12), sys.version'
mkdir -p "$ENV_ROOT" "$ROOT/requirements/locks" "$ROOT/06_logs"
if ! command -v nvidia-smi >/dev/null; then echo 'WSL cannot see the Windows NVIDIA driver. Do not install a Linux display driver inside WSL.' >&2; exit 3; fi
nvidia-smi -L
install_env() {
  local name="$1" requirements="$2" venv="$ENV_ROOT/$1"
  "$PYTHON_BIN" -m venv "$venv"
  "$venv/bin/python" -m pip install --upgrade pip
  if [[ "$name" == rtmpose || "$name" == yolo || "$name" == sapiens ]]; then
    "$venv/bin/python" -m pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu132
  fi
  "$venv/bin/python" -m pip install -r "$ROOT/requirements/$requirements"
  if [[ "$name" == movenet ]]; then
    tf_dir="$("$venv/bin/python" -c 'import os,tensorflow; print(os.path.dirname(tensorflow.__file__))')"
    (cd "$tf_dir" && ln -sfn ../nvidia/*/lib/*.so* .)
    ptxas="$(find "$(dirname "$tf_dir")/nvidia" -path '*/bin/ptxas' -print -quit)"
    if [[ -n "$ptxas" ]]; then ln -sfn "$ptxas" "$venv/bin/ptxas"; fi
  fi
  if [[ "$name" == rtmpose ]]; then
    # RTMLib's default dependency is CPU onnxruntime; the GPU wheel is already installed above.
    "$venv/bin/python" -m pip install --no-deps rtmlib
  fi
  "$venv/bin/python" -m pip install ipykernel
  "$venv/bin/python" -m ipykernel install --user --name "infantpose-$name" --display-name "InfantPose ($name)"
  "$venv/bin/python" -m pip freeze > "$ROOT/requirements/locks/$name-linux.txt"
}
install_env mediapipe mediapipe.in
install_env movenet movenet.in
install_env rtmpose rtmpose.in
install_env yolo yolo.in
install_env openpose openpose.in
echo "Created isolated model kernels under $ENV_ROOT. Run ../scripts/gpu_diagnostics.py in each kernel and keep the generated lock files with the environment report."
