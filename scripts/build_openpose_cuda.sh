#!/usr/bin/env bash
set -euo pipefail
export PIP_REQUIRE_VIRTUALENV=true

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_ROOT="${INFANTPOSE_CACHE_ROOT:-$HOME/.local/share/infantpose}"
ENV_ROOT="$CACHE_ROOT/venvs"
SRC="$CACHE_ROOT/src/openpose"
BUILD="$CACHE_ROOT/build/openpose-cuda-final"
CUDA_SHIM="$CACHE_ROOT/build/cuda-toolkit-13.2-shim"
BUILD_ENV="$ENV_ROOT/openpose-build"
MODEL_CACHE="$ROOT/02_datasets/model_cache/openpose_coco18"

for package in g++ make git pkg-config protoc; do
  if ! command -v "$package" >/dev/null 2>&1; then
    echo "Missing WSL build dependency: $package. Install the documented OpenPose apt dependencies first." >&2
    exit 2
  fi
done
for header in /usr/include/google/protobuf/message.h /usr/include/gflags/gflags.h /usr/include/glog/logging.h /usr/include/opencv4/opencv2/core.hpp /usr/include/boost/version.hpp /usr/include/eigen3/Eigen/Core; do
  if [[ ! -f "$header" ]]; then echo "Missing WSL development header: $header" >&2; exit 2; fi
done
command -v nvidia-smi >/dev/null || { echo 'WSL cannot see the Windows NVIDIA driver.' >&2; exit 3; }
nvidia-smi -L

mkdir -p "$ENV_ROOT" "$CACHE_ROOT/src" "$CACHE_ROOT/build" "$ROOT/06_logs"
if [[ ! -x "$BUILD_ENV/bin/python" ]]; then python3.12 -m venv "$BUILD_ENV"; fi
"$BUILD_ENV/bin/python" -m pip install -r "$ROOT/requirements/openpose-build.in"
"$BUILD_ENV/bin/python" -m pip freeze > "$ROOT/requirements/locks/openpose-build-linux.txt"

if [[ ! -d "$SRC/.git" ]]; then
  git clone --recurse-submodules --branch v1.7.0 --depth 1 https://github.com/CMU-Perceptual-Computing-Lab/openpose.git "$SRC"
fi
git -C "$SRC" checkout --detach v1.7.0
git -C "$SRC/3rdparty/pybind11" fetch --depth 1 origin tag v2.13.6
git -C "$SRC/3rdparty/pybind11" checkout --detach v2.13.6

apply_patch_once() {
  local repo="$1" patch_file="$2"
  if git -C "$repo" apply --unidiff-zero --check "$patch_file"; then
    git -C "$repo" apply --unidiff-zero "$patch_file"
  elif git -C "$repo" apply --unidiff-zero --reverse --check "$patch_file"; then
    echo "Already applied: $patch_file"
  else
    echo "Patch does not match the cached source: $patch_file" >&2
    exit 4
  fi
}
apply_patch_once "$SRC" "$ROOT/patches/openpose-1.7.0-root-cxx17.patch"
apply_patch_once "$SRC/3rdparty/caffe" "$ROOT/patches/openpose-1.7.0-caffe-cxx17.patch"

CUDA_PIP_ROOT="$BUILD_ENV/lib/python3.12/site-packages/nvidia/cu13"
if [[ ! -x "$CUDA_PIP_ROOT/bin/nvcc" ]]; then echo "CUDA 13.2 nvcc missing from $BUILD_ENV" >&2; exit 5; fi
mkdir -p "$CUDA_SHIM"
ln -sfn "$CUDA_PIP_ROOT/bin" "$CUDA_SHIM/bin"
ln -sfn "$CUDA_PIP_ROOT/include" "$CUDA_SHIM/include"
ln -sfn "$CUDA_PIP_ROOT/lib" "$CUDA_SHIM/lib64"
ln -sfn "$CUDA_PIP_ROOT/nvvm" "$CUDA_SHIM/nvvm"
ln -sfn cccl/thrust "$CUDA_PIP_ROOT/include/thrust"
ln -sfn cccl/cub "$CUDA_PIP_ROOT/include/cub"
ln -sfn cccl/cuda "$CUDA_PIP_ROOT/include/cuda"
mkdir -p "$CUDA_PIP_ROOT/include/nv"
ln -sfn ../cccl/nv/target "$CUDA_PIP_ROOT/include/nv/target"
for entry in libcudart.so.13 libcublas.so.13 libcurand.so.10; do
  ln -sfn "$CUDA_PIP_ROOT/lib/$entry" "$CUDA_PIP_ROOT/lib/${entry%.*}"
done

WEIGHT="$MODEL_CACHE/pose_iter_440000.caffemodel"
[[ -s "$WEIGHT" ]] || { echo "Missing cached OpenPose weights: $WEIGHT" >&2; exit 6; }
echo "5156d31f670511fce9b4e28b403f2939  $WEIGHT" | md5sum --check --status || { echo 'OpenPose weight MD5 mismatch.' >&2; exit 6; }
mkdir -p "$SRC/models/pose/coco"
ln -sfn "$WEIGHT" "$SRC/models/pose/coco/pose_iter_440000.caffemodel"

export CUDA_BIN_PATH="$CUDA_SHIM"
"$BUILD_ENV/bin/cmake" -S "$SRC" -B "$BUILD" -G 'Unix Makefiles' \
  -DGPU_MODE=CUDA \
  -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_SHIM" \
  -DCUDA_NVCC_EXECUTABLE="$CUDA_SHIM/bin/nvcc" \
  -DCUDA_ARCH=Manual -DCUDA_ARCH_BIN=12.0 -DCUDA_ARCH_PTX=12.0 \
  -DUSE_CUDNN=OFF -DBUILD_EXAMPLES=ON -DBUILD_PYTHON=ON \
  -DDOWNLOAD_BODY_25_MODEL=OFF -DDOWNLOAD_BODY_COCO_MODEL=OFF \
  -DDOWNLOAD_BODY_MPI_MODEL=OFF -DDOWNLOAD_FACE_MODEL=OFF \
  -DDOWNLOAD_HAND_MODEL=OFF
"$BUILD_ENV/bin/cmake" --build "$BUILD" -- -j"${OPENPOSE_BUILD_JOBS:-2}" 2>&1 | tee "$ROOT/06_logs/openpose-cuda-build.log"

EXTENSION="$(find "$BUILD/python/openpose" -maxdepth 1 -name 'pyopenpose*.so' -print -quit)"
[[ -n "$EXTENSION" ]] || { echo 'CUDA build completed but pyopenpose extension is missing.' >&2; exit 7; }
echo "OpenPose CUDA build completed: $EXTENSION"
