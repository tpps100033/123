#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_ROOT="${INFANTPOSE_ENV_ROOT:-$HOME/.local/share/infantpose/venvs}"
THERMAL_LIMIT="${INFANTPOSE_MAX_GPU_TEMP_C:-85}"
mkdir -p "$ROOT/06_logs"
LOG="$ROOT/06_logs/research_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$LOG") 2>&1
printf 'Starting full local research run. GPU thermal cutoff: %s C. Log: %s\n' "$THERMAL_LIMIT" "$LOG"
run_model() {
  local model="$1" env_name="$2" dataset
  local python="$ENV_ROOT/$env_name/bin/python"
  [[ -x "$python" ]] || { printf 'Missing isolated environment: %s\n' "$python" >&2; return 2; }
  if [[ "$model" == openpose_coco18 ]]; then
    local cache_root="${INFANTPOSE_CACHE_ROOT:-$HOME/.local/share/infantpose}"
    export INFANTPOSE_OPENPOSE_PYTHON_DIR="$cache_root/build/openpose-cuda-final/python/openpose"
    export INFANTPOSE_OPENPOSE_MODEL_FOLDER="$cache_root/src/openpose/models"
    export LD_LIBRARY_PATH="$cache_root/build/openpose-cuda-final/src/openpose:$cache_root/build/openpose-cuda-final/caffe/lib:$cache_root/build/cuda-toolkit-13.2-shim/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
  for dataset in coco_singleperson_val2017 syrip_test100 coco_unihuman_child_val134; do
    INFANTPOSE_MAX_GPU_TEMP_C="$THERMAL_LIMIT" "$python" "$ROOT/scripts/run_research.py" --model "$model" --dataset "$dataset"
  done
}

run_model mediapipe_pose_heavy mediapipe
run_model movenet_thunder movenet
run_model rtmpose_m rtmpose
run_model yolo11x_pose yolo
run_model openpose_coco18 openpose
run_model sapiens_1b_coco sapiens
"$ENV_ROOT/rtmpose/bin/python" "$ROOT/scripts/merge_results.py"
"$ENV_ROOT/rtmpose/bin/python" "$ROOT/scripts/analyse_results.py"
printf 'All available model/dataset runs completed. Partial, failed, or thermally stopped runs are excluded from merge.\n'
