#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."
root="$(pwd)"
env_root="${INFANTPOSE_ENV_ROOT:-$HOME/.local/share/infantpose/venvs}"
limit="${INFANTPOSE_MAX_GPU_TEMP_C:-85}"
dataset="sviro_child_i3_test53_coco12"
[[ -f "02_datasets/canonical/$dataset/dataset_manifest.json" ]] || {
  echo "Prepare the child dataset on Windows first: py -3.9 scripts/prepare_sviro_child.py" >&2
  exit 2
}
models=(mediapipe_pose_heavy movenet_thunder rtmpose_m yolo11x_pose sapiens_1b_coco openpose_coco18)
envs=(mediapipe movenet rtmpose yolo sapiens openpose)
for i in "${!models[@]}"; do
  model="${models[$i]}"
  python="$env_root/${envs[$i]}/bin/python"
  if [[ "$model" == openpose_coco18 ]]; then
    cache_root="${INFANTPOSE_CACHE_ROOT:-$HOME/.local/share/infantpose}"
    export INFANTPOSE_OPENPOSE_PYTHON_DIR="$cache_root/build/openpose-cuda-final/python/openpose"
    export INFANTPOSE_OPENPOSE_MODEL_FOLDER="$cache_root/src/openpose/models"
    export LD_LIBRARY_PATH="$cache_root/build/openpose-cuda-final/src/openpose:$cache_root/build/openpose-cuda-final/caffe/lib:$cache_root/build/cuda-toolkit-13.2-shim/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
  echo "[SVIRO child] validate $model"
  "$python" "$root/scripts/validate_model.py" --model "$model" --dataset "$dataset"
  echo "[SVIRO child] evaluate $model"
  INFANTPOSE_MAX_GPU_TEMP_C="$limit" "$python" "$root/scripts/run_research.py" --model "$model" --dataset "$dataset"
done
"$env_root/rtmpose/bin/python" "$root/scripts/compare_sviro_child.py"
