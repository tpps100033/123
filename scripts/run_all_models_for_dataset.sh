#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_ROOT="${INFANTPOSE_ENV_ROOT:-$HOME/.local/share/infantpose/venvs}"
THERMAL_LIMIT="${INFANTPOSE_MAX_GPU_TEMP_C:-85}"
DATASET="${1:-}"
case "$DATASET" in
  coco_singleperson_val2017|syrip_test100|coco_unihuman_child_val134|simuletic_cctv_fall_pilot111|simuletic_cctv_fall_singleperson_pilot111|nameframe_quarry_crowd_singleperson_pilot300) ;;
  *) echo "Usage: $0 {coco_singleperson_val2017|syrip_test100|coco_unihuman_child_val134|simuletic_cctv_fall_pilot111|simuletic_cctv_fall_singleperson_pilot111|nameframe_quarry_crowd_singleperson_pilot300}" >&2; exit 2 ;;
esac
mkdir -p "$ROOT/06_logs"
LOG="$ROOT/06_logs/research_${DATASET}_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$LOG") 2>&1
echo "Starting six-model run for $DATASET; GPU thermal cutoff ${THERMAL_LIMIT} C; log $LOG"

run_model() {
  local model="$1" env_name="$2" python="$ENV_ROOT/$2/bin/python"
  [[ -x "$python" ]] || { echo "Missing isolated environment: $python" >&2; return 2; }
  if [[ "$model" == openpose_coco18 ]]; then
    local cache_root="${INFANTPOSE_CACHE_ROOT:-$HOME/.local/share/infantpose}"
    export INFANTPOSE_OPENPOSE_PYTHON_DIR="$cache_root/build/openpose-cuda-final/python/openpose"
    export INFANTPOSE_OPENPOSE_MODEL_FOLDER="$cache_root/src/openpose/models"
    export LD_LIBRARY_PATH="$cache_root/build/openpose-cuda-final/src/openpose:$cache_root/build/openpose-cuda-final/caffe/lib:$cache_root/build/cuda-toolkit-13.2-shim/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
  echo "=== $model / $DATASET ==="
  INFANTPOSE_MAX_GPU_TEMP_C="$THERMAL_LIMIT" "$python" "$ROOT/scripts/run_research.py" --model "$model" --dataset "$DATASET"
}

failed=0
for entry in \
  'mediapipe_pose_heavy mediapipe' \
  'movenet_thunder movenet' \
  'rtmpose_m rtmpose' \
  'yolo11x_pose yolo' \
  'openpose_coco18 openpose' \
  'sapiens_1b_coco sapiens'; do
  read -r model env_name <<< "$entry"
  if ! run_model "$model" "$env_name"; then
    echo "FAILED: $model / $DATASET; continuing with the remaining independent models."
    failed=1
  fi
done
echo "Finished six-model dataset run. failed=$failed"
exit "$failed"
