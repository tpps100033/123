# Migration report

Date: 2026-10-22 (Asia/Taipei)

## Source inventory

Source: the user-provided Drive folder `105Q9cXxPajv--JWFCGYX8bUky-Q81zKJ`. It contains project configuration and source, datasets, model cache, raw/merged results, analysis outputs, paper/demo files, and nine formal/analysis/demo notebooks. The migration copied all nine original notebooks without editing them to `notebooks/original/`; they remain locally preserved and are not tracked by Git. The project-owned 63 KB source ZIP was retained as `source_archive.zip`; unsafe archive paths were checked before extraction. Compiled Python cache files were excluded from the working source tree. The five user-provided Drive export ZIP parts were checked for path safety and duplicate members before selectively importing the original COCO/SyRIP archives, annotation JSON, 1,021 exact selected images, frozen handoff manifests, seven cached model asset files, and historical 03/04/05 outputs. The ZIP parts were not modified.

The notebooks use Colab `drive.mount`, `/content/drive/MyDrive/姿態判定`, authenticated Google Drive API calls, direct package installation, and Ubuntu-only OpenPose shell/build commands. Local code now uses project-relative paths, a centralized model cache, isolated kernels, and local scripts. No model/dataset assets were committed.

## Research definition found

The source is internally inconsistent and must not be silently harmonized:

- The Drive root README describes six models (MediaPipe, MoveNet, RTMPose, YOLO, Sapiens, OpenPose) and requires 12 model × dataset rows in the merge, with legacy rows explicitly excluded from timing rankings.
- The archived v4 PRD/source README defines the formal benchmark as four models (MediaPipe, MoveNet, RTMPose, YOLO) and says Sapiens/OpenPose are outside v4 scope.
- The user task explicitly requests real validation of MediaPipe, MoveNet Thunder, RTMPose, and OpenPose.

This migration keeps the four v4 Python adapters runnable, preserves the OpenPose and Sapiens Colab originals, and marks OpenPose as blocked until a local WSL2 binary is built and executes inference. It does not add OpenPose or Sapiens to formal rankings. The local merge accepts only complete v4 `fair_reference_v1` runs (921 COCO or 100 SyRIP rows).

## Hardware and runtime compatibility

Verified after the user enabled WSL2 and restarted:

- Windows 11, RTX 5060 Laptop GPU, 8,151 MiB VRAM, NVIDIA driver 610.71 (CUDA UMD 13.3).
- Ubuntu 24.04.5 under WSL2, kernel `6.18.54.1-microsoft-standard-WSL2`, Python 3.12.3; GPU is exposed through the existing Windows driver.
- Isolated model venvs and Jupyter kernels are under `/home/tpps1/.local/share/infantpose/venvs/`; no project dependencies were installed into Windows/global Python.
- RTMPose environment: PyTorch 2.12.0+cu132 and ONNX Runtime 1.31.0. The two RTMLib ONNX sessions report actual `CUDAExecutionProvider` plus CPU fallback providers.
- Model-specific resolved dependencies are snapshotted in `requirements/locks/*-linux.txt`.

The RTX 5060 is Blackwell compute capability 12.0; the PyTorch CUDA 13.2 wheel in the isolated RTMPose/YOLO environments completed an actual CUDA matrix operation and model inference. WSL uses the Windows driver; no separate Linux NVIDIA display driver was installed.

Sources: [NVIDIA RTX prerequisites and compute capability](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/installing-tensorrt-rtx/prerequisites.html), [NVIDIA CUDA on WSL guide](https://docs.nvidia.com/cuda/archive/13.2.0/wsl-user-guide/index.html), [PyTorch 2.7 Blackwell release notes](https://pytorch.org/blog/pytorch-2-7/), [PyTorch local installation selector](https://pytorch.org/get-started/locally/).

## Changes made

- Preserved the source archive and unmodified Drive notebooks.
- Added WSL environment check, isolated per-model environment bootstrap, model asset cache/download script, dataset handoff preparation, GPU diagnostics, model inference smoke validation, formal runner entry point, merge and analysis scripts, and local notebook generator.
- Added local paths/configuration, dependency input files, ignored data/weights/outputs, and GPU memory/environment fields in run manifests/logs.
- The model environment bootstrap uses one WSL-home venv per framework/model and requires pip to be inside a venv. It does not install Python packages globally or alter the installed Windows Python.
- At the user's request, the local scoring path was changed from project-defined per-image mean OKS/PCK to official `cocodataset/cocoapi.COCOeval(iouType='keypoints')`; unchanged COCO defaults produce AP/AP50/AP75/AP-medium/AP-large and AR equivalents. COCO output is explicitly the 921-image single-person subset; SyRIP is custom-domain COCO-OKS AP. Pose coverage is diagnostic only. Each pose detection score is the mean of its 17 model keypoint confidence values.
- Added single/batch adapter entry points. Smoke validation explicitly checks an actual single image and a two-image sequential batch and writes `06_logs/model_validation_<model>.json`.
- Added OOM recovery for per-image inference so one failed sample is logged and later images continue. Error/no-detection rows remain zero-scored in the original denominator; an incomplete run cannot enter accuracy merge, and CPU/failure runs cannot enter GPU timing rankings.
- Added a sequential one-command research runner with an explicit temperature limit and a nonzero exit for partial/thermal-stop runs.

## Validation status

| Model | Single image + batch two | GPU inference | Full COCO subset | Full SyRIP | Result |
|---|---|---|---|---|---|
| MediaPipe Pose Heavy | Pass; 17 points and matching first batch output | No. MediaPipe GPU task rejects WSL `llvmpipe`; CPU fallback used | Complete, CPU | Complete, CPU | AP 0.41765 / 0.73032 |
| MoveNet Thunder | Pass; 17 points and matching first batch output | No. TensorFlow cannot load its GPU stack and TFLite GPU delegate library is absent; CPU/XNNPACK fallback used | Complete, CPU | Complete, CPU | AP 0.103 / 0.417 |
| RTMPose-M + YOLOX-M | Pass; 17 points and matching first batch output | Yes; both ONNX sessions selected CUDAExecutionProvider | Complete, 921/921 | Complete, 100/100 | AP 0.77481 / 0.83417 |
| YOLO11x-Pose | Pass; 17 points and matching first batch output | Yes; PyTorch CUDA on RTX 5060 | Latest retry with ASUS Full Speed fan mode stopped at 156/921 at 71°C; not scored. Earlier stock-mode attempt reached 670/921 at the same temperature. | Complete, 100/100 | SyRIP AP 0.75071; no COCO score |
| OpenPose COCO-18 | No local adapter/runtime to execute the model | Not tested | Not run | Not run | Blocked; original notebooks, prototxt, and weights are preserved |

All accuracy figures above use `cocodataset/cocoapi.COCOeval(iouType='keypoints')` defaults. COCO results are for the selected 921-image single-person subset, not the full COCO val2017 leaderboard. SyRIP uses the same COCO-OKS evaluator and is a custom-domain result. The merged complete-run table is `04_merged_results/merged_20261022T164221Z/benchmark_table.csv`; the paired cross-domain report and plots are under `05_analysis_results/analysis_20261022T164221Z/`. Partial YOLO COCO results are excluded from merge and scoring.

Smoke logs are in `06_logs/model_validation_*.json`; environment diagnostics are in `06_logs/gpu_diagnostics_*.json`. The user enabled ASUS MyASUS Full Speed fan mode, with AC power connected; the Windows active cooling policy was already enabled. WSL reports fan speed as `N/A`. The latest YOLO COCO retry used a 70°C cutoff with checks every two images, stopped at 71°C after 156 predictions, and was not scored. GPU temperature cooled to 61°C at idle after that run. The hard 75°C default was not reached. Future one-click runs default to a 65°C cutoff and five-image polling interval. No matching official-metric Colab score pair was found, so there is no numerical Colab-to-local delta. Historical project-specific OKS/PCK results remain preserved and are not mixed with these COCOeval results.

## Remaining host-level requirement

OpenPose remains unexecuted: the preserved source calls a Colab-cached OpenPose binary and has no portable adapter/runtime. A CUDA 13 / SM 12.0 WSL build plus COCO-18-to-COCO-17 mapping is still required before it can be validated. The configured Git repository has no remote URL, so no GitHub push was possible; local commit status is reported separately.
