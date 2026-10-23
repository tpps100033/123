# Migration report

Date: 2026-10-23 (Asia/Taipei)

## Source inventory

Source: the user-provided Drive folder `105Q9cXxPajv--JWFCGYX8bUky-Q81zKJ`. It contains project configuration and source, datasets, model cache, raw/merged results, analysis outputs, paper/demo files, and nine formal/analysis/demo notebooks. The migration copied all nine original notebooks without editing them to `notebooks/original/`; they remain locally preserved and are not tracked by Git. The project-owned 63 KB source ZIP was retained as `source_archive.zip`; unsafe archive paths were checked before extraction. Compiled Python cache files were excluded from the working source tree. The five user-provided Drive export ZIP parts were checked for path safety and duplicate members before selectively importing the original COCO/SyRIP archives, annotation JSON, 1,021 exact selected images, frozen handoff manifests, seven cached model asset files, and historical 03/04/05 outputs. The ZIP parts were not modified.

The notebooks use Colab `drive.mount`, `/content/drive/MyDrive/姿態判定`, authenticated Google Drive API calls, direct package installation, and Ubuntu-only OpenPose shell/build commands. Local code now uses project-relative paths, a centralized model cache, isolated kernels, and local scripts. No model/dataset assets were committed.

## Research definition found

The source is internally inconsistent and must not be silently harmonized:

- The Drive root README describes six models (MediaPipe, MoveNet, RTMPose, YOLO, Sapiens, OpenPose) and requires 12 model × dataset rows in the merge, with legacy rows explicitly excluded from timing rankings.
- The archived v4 PRD/source README defines the formal benchmark as four models (MediaPipe, MoveNet, RTMPose, YOLO) and says Sapiens/OpenPose are outside v4 scope.
- The user task explicitly requests real validation of MediaPipe, MoveNet Thunder, RTMPose, and OpenPose.

The user clarified that the intended comparison is all six models in the Drive root README. The local scope therefore extends the archived four-model v4 PRD: it retains the formal v4 models, adds the explicitly requested OpenPose validation, and now integrates Sapiens-1B. The merge requires all 12 complete `fair_reference_v1` model × dataset runs (921 COCO or 100 SyRIP rows); the prior five-model table remains historical until Sapiens has two local runs.

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
- Added a sequential one-command research runner with an 85°C default temperature limit, per-image checks, and nonzero exit for partial/thermal-stop runs. It writes per-image prediction checkpoints so native runtime aborts preserve completed work.
- Built official OpenPose 1.7.0 CUDA/Caffe and `pyopenpose` in the isolated WSL cache for CUDA 13.2 and RTX 5060 `sm_120`. Compatibility patches update C++ to C++17, guard removed CUDA 13 device-property log fields, and use the current Protobuf `SetTotalBytesLimit` API. The bundled pybind11 development snapshot failed against Python 3.12, so the build pins official pybind11 v2.13.6.
- Added an OpenPose CUDA adapter with OpenPose COCO-18 to COCO-17 mapping, PIL input, persistent model context, VRAM and temperature telemetry, and sequential one-image inference. A Python OpenCV wheel loaded beside the system OpenCV linked into OpenPose caused a native crash; the OpenPose venv therefore excludes that wheel.
- Fixed OpenPose network input to 480×368. The initial dynamic-width 368-high configuration grew toward the 8 GiB VRAM limit across image sizes and eventually hit Caffe's fatal CUDA out-of-memory path. At 480×368, the smoke run peaked at 6,127 MiB and the complete COCO/SyRIP runs stayed below the 85°C limit.

## Validation status

| Model | Single image + batch two | GPU inference | Full COCO subset | Full SyRIP | Result |
|---|---|---|---|---|---|
| MediaPipe Pose Heavy | Pass; 17 points and matching first batch output | No. MediaPipe GPU task rejects WSL `llvmpipe`; CPU fallback used | Complete, CPU | Complete, CPU | AP 0.41765 / 0.73032 |
| MoveNet Thunder | Pass; 17 points and matching first batch output | No. TensorFlow cannot load its GPU stack and TFLite GPU delegate library is absent; CPU/XNNPACK fallback used | Complete, CPU | Complete, CPU | AP 0.103 / 0.417 |
| RTMPose-M + YOLOX-M | Pass; 17 points and matching first batch output | Yes; both ONNX sessions selected CUDAExecutionProvider | Complete, 921/921 | Complete, 100/100 | AP 0.77481 / 0.83417 |
| YOLO11x-Pose | Pass; 17 points and matching first batch output | Yes; PyTorch CUDA on RTX 5060 | Complete, 921/921; official COCO AP 0.72192 | Complete, 100/100 | COCO AP 0.72192 / SyRIP AP 0.75071 |
| OpenPose COCO-18 | Pass; official CUDA wrapper, 17-point mapping, single image and sequential two-image batch | Yes; Caffe CUDA on RTX 5060, observed 77% utilization and 6,127 MiB | Complete, 921/921; official COCO AP 0.58854 | Complete, 100/100; official COCO-OKS AP 0.47656 | Passed at fixed 480×368, no thermal stop |
| Sapiens-1B COCO | Blocked before model load; checkpoint unavailable locally, so no model result is claimed | PyTorch CUDA 2.12/cu132 diagnostics pass on RTX 5060 `sm_120`; model GPU use remains unverified locally | Not run | Not run | Isolated venv/kernel, adapter, and official Lite source prepared; requires the original 4,676,154,239-byte Drive checkpoint in the WSL cache |

All accuracy figures above use `cocodataset/cocoapi.COCOeval(iouType='keypoints')` defaults. COCO results are for the selected 921-image single-person subset, not the full COCO val2017 leaderboard. SyRIP uses the same COCO-OKS evaluator and is a custom-domain result. The merged complete-run table is `04_merged_results/merged_20261022T164221Z/benchmark_table.csv`; the paired cross-domain report and plots are under `05_analysis_results/analysis_20261022T164221Z/`. Partial YOLO COCO results are excluded from merge and scoring.

The source Drive contains older Sapiens T4 runs, but those record the historical custom OKS/PCK metrics and cannot be compared numerically to this project's official COCOeval results. Sapiens was added to the local runner, validator, environment bootstrap, and notebook; its real model run is blocked at the missing checkpoint. The Drive connector rejects files over 256 MiB, and `gdown` cannot download this account-private file without public sharing. No Drive permissions were changed. The required local path is `/home/tpps1/.local/share/infantpose/model_cache/sapiens_1b/sapiens_1b_coco_best_coco_AP_821_torchscript.pt2`; its exact expected size is 4,676,154,239 bytes. After it is present, run `python scripts/validate_model.py --model sapiens_1b_coco --dataset coco_singleperson_val2017`, then `python scripts/run_research.py` for each dataset, and regenerate merge/analysis.

The Sapiens isolated environment uses PyTorch 2.12.0+cu132 on Python 3.12.3; CUDA diagnostics report RTX 5060, capability 12.0, and a successful CUDA matrix smoke. The official Lite demo source checkout is commit `2cb07227a740cf09896309ea3a3b8fa44429865c2`; its Python CLI/import check passes. `06_logs/model_validation_sapiens_1b_coco.json` records `blocked` before model load, with the exact cache path and no claimed inference result. The local in-app browser also redirects the private Drive file to Google sign-in; no credentials or sharing permissions were changed. GPU was idle at 50°C after these checks.

Smoke logs are in `06_logs/model_validation_*.json`; environment diagnostics are in `06_logs/gpu_diagnostics_*.json`. The user enabled ASUS MyASUS Full Speed fan mode with AC power connected; Windows active cooling was already enabled. WSL reports fan speed as `N/A`. YOLO11x-Pose completed COCO with 921/921 successful images and official COCOeval AP 0.721922, AP50 0.918698, AP75 0.794329, and AR 0.778827. OpenPose completed both datasets with a fixed 480×368 network input, 85°C stop limit, and per-image checks. Its COCO run ID is `openpose_coco18_coco_singleperson_val2017_20261023T041950Z_e2bd52` (AP 0.588541, AP50 0.788966, AP75 0.649972, AR 0.640825); SyRIP run ID is `openpose_coco18_syrip_test100_20261023T042150Z_9f9858` (AP 0.476561, AP50 0.918898, AP75 0.328991, AR 0.494000). OpenPose observed peak temperature was 76°C and peak VRAM 6,127 MiB; no thermal stops or model errors occurred. Both runs are in `04_merged_results/merged_20261023T042218Z/benchmark_table.csv` and `05_analysis_results/analysis_20261023T042224Z/`. An earlier dynamic-resolution trial hit a Caffe native OOM and was not scored; fixed resolution lowered the peak and completed both datasets. No matching official-metric Colab score pair is present in the imported results. Historical OKS/PCK metrics use different definitions, so their values are preserved but not compared numerically with COCOeval.

## Remaining host-level requirement

OpenPose, COCO, and SyRIP validation are complete. `scripts/build_openpose_cuda.sh` records the WSL native dependencies, isolated CUDA toolkit, upstream source/pybind versions, weight hash check, compatibility patches, and low-parallel build. The configured Git repository has no remote URL, so a GitHub push is unavailable; local commit status is reported after the final review.
