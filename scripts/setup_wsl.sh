#!/usr/bin/env bash
set -euo pipefail
export PIP_REQUIRE_VIRTUALENV=true
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
command -v "$PYTHON_BIN" >/dev/null || { echo "Need $PYTHON_BIN plus python3.11-venv (Ubuntu packages)." >&2; exit 2; }
"$PYTHON_BIN" -c 'import sys; assert sys.version_info >= (3,10), sys.version'
mkdir -p "$ROOT/.venvs" "$ROOT/requirements/locks" "$ROOT/06_logs"
if ! command -v nvidia-smi >/dev/null; then echo 'WSL cannot see the Windows NVIDIA driver. Do not install a Linux display driver inside WSL.' >&2; exit 3; fi
nvidia-smi -L
install_env() {
  local name="$1" requirements="$2" venv="$ROOT/.venvs/$1"
  "$PYTHON_BIN" -m venv "$venv"
  "$venv/bin/python" -m pip install --upgrade pip
  if [[ "$name" == rtmpose || "$name" == yolo || "$name" == sapiens ]]; then
    "$venv/bin/python" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
  fi
  "$venv/bin/python" -m pip install -r "$ROOT/requirements/$requirements"
  "$venv/bin/python" -m pip install ipykernel
  "$venv/bin/python" -m ipykernel install --user --name "infantpose-$name" --display-name "InfantPose ($name)"
  "$venv/bin/python" -m pip freeze > "$ROOT/requirements/locks/$name-linux.txt"
}
install_env mediapipe mediapipe.in
install_env movenet movenet.in
install_env rtmpose rtmpose.in
install_env yolo yolo.in
echo 'Created isolated model kernels. Run ../scripts/gpu_diagnostics.py in each kernel and keep the generated lock files with the environment report.'
