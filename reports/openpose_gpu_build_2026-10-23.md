# OpenPose CUDA build and evaluation

Date: 2026-10-23 (Asia/Taipei)

## Environment and build

- WSL2 Ubuntu 24.04.5, Python 3.12.3, RTX 5060 Laptop GPU, 8,151 MiB VRAM, compute capability 12.0 (`sm_120`). WSL uses the existing Windows NVIDIA driver.
- OpenPose v1.7.0 source commit: `8ca5c1d95a42340b323e9273654d1db98bec779c`.
- CUDA build environment is isolated at `~/.local/share/infantpose/venvs/openpose-build/`; runtime is isolated in `~/.local/share/infantpose/venvs/openpose/`.
- CUDA Toolkit 13.2.1 components, CMake 3.31.10, and pybind11 v2.13.6. OpenPose was built with CUDA for `sm_120`, cuDNN disabled, Caffe enabled, and its official Python API enabled.
- Native build dependencies were installed only inside the WSL Ubuntu distro. No Linux display driver was installed; WSL uses the Windows driver.
- The COCO weights are cached in the project and their MD5 matches OpenPose's expected `5156d31f670511fce9b4e28b403f2939`.

`scripts/build_openpose_cuda.sh` recreates the build in the WSL cache. It uses the pinned build requirements, applies the two source patches, checks the local weight hash, and defaults to two build jobs. The recorded patches change the OpenPose/Caffe C++ dialect to C++17 for CUDA 13 CCCL, guard three removed CUDA 13 device-property diagnostics, and use Protobuf's current one-argument `SetTotalBytesLimit` API. The OpenPose submodule's pybind11 development snapshot did not compile with CPython 3.12, so the script pins the upstream v2.13.6 tag.

Version references: [NVIDIA CUDA Toolkit 13.2.1 component metadata](https://pypi.org/project/cuda-toolkit/13.2.1/), [NVIDIA CCCL source and compatibility information](https://github.com/NVIDIA/cccl), and [pybind11 CPython compatibility changelog](https://pybind11.readthedocs.io/en/stable/changelog.html).

## Runtime verification

The official `pyopenpose` wrapper completed single-image and two-image sequential inference. It returned OpenPose COCO-18 arrays shaped `(people, 18, 3)` and the adapter maps those coordinates/scores to the benchmark's COCO-17 order. GPU telemetry during smoke inference reached 77% utilization, 6,127 MiB VRAM, and 55°C. The adapter imports the compiled OpenPose module only after setting its isolated module path and reads images with Pillow; it does not import the venv's OpenCV wheel because loading that wheel alongside OpenPose's system OpenCV caused a native crash.

An initial dynamic-width, 368-pixel-high configuration climbed to about 7.9 GiB VRAM on the mixed image sizes and then hit Caffe's fatal CUDA out-of-memory path at about 66°C. That run is incomplete and was not scored. The adapter now uses a fixed 480×368 network input and batch size one. The smoke peak fell to 6,127 MiB, and the two full dataset runs completed without OOM or a thermal stop. COCO and SyRIP are scored with the unchanged official `cocodataset/cocoapi.COCOeval(iouType='keypoints')` protocol.

| Dataset | Rows | OK | No detection | Errors | AP | AP50 | AP75 | AR | Peak VRAM | Peak temperature |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| COCO single-person val2017 subset | 921 | 829 | 92 | 0 | 0.588541 | 0.788966 | 0.649972 | 0.640825 | 6,127 MiB | 76°C |
| SyRIP test100 | 100 | 99 | 1 | 0 | 0.476561 | 0.918898 | 0.328991 | 0.494000 | 6,127 MiB | 68°C |

COCO run: `openpose_coco18_coco_singleperson_val2017_20261023T041950Z_e2bd52`.

SyRIP run: `openpose_coco18_syrip_test100_20261023T042150Z_9f9858`.

Logs: `06_logs/model_validation_openpose_coco18.json`, `06_logs/openpose_coco_fixed_2026-10-23.log`, and `06_logs/openpose_syrip_2026-10-23.log`.

## Results comparison

No imported Colab run with the official COCO keypoint evaluator was found for OpenPose, so a direct Colab-to-local numerical delta cannot be calculated. Historical project-specific OKS/PCK scores use different definitions and are not compared as if they were COCO AP. The fixed 480×368 input is recorded in each run manifest; comparisons to any prior OpenPose run at a different network resolution should account for this inference setting.

The merged 10-run table is `04_merged_results/merged_20261023T042218Z/benchmark_table.csv`; the paired analysis is under `05_analysis_results/analysis_20261023T042224Z/`.
