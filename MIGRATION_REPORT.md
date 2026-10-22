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

Observed on the host:

- Windows 11 host, NVIDIA GeForce RTX 5060, 8,151 MiB reported VRAM.
- NVIDIA driver 610.71; `nvidia-smi` reports CUDA UMD 13.3.
- WSL reports the installation instructions and has no Ubuntu distribution (`wsl --list --verbose` is empty/not configured).
- Python launcher has only Python 3.9.13. No Torch, TensorFlow, MediaPipe, OpenCV, Jupyter, or ONNX Runtime packages are installed.
- Git is available in the Visual Studio installation; the new standalone `InfantPose` repository is on branch `migration/windows-local-gpu`.

NVIDIA lists RTX 5000 series Blackwell at compute capability 12.0, first supported by CUDA 12.8. PyTorch's official 2.7 release introduced Blackwell support with CUDA 12.8 wheels. The bootstrap resolves current packages from the CUDA 12.8 wheel index and snapshots the resolved versions; it does not pin an old PyTorch version. The installed host driver reports CUDA 13.3, so it can run applications using the earlier 12.8 runtime. WSL2 GPU compute uses the Windows driver; do not install a second Linux GPU driver inside WSL.

Sources: [NVIDIA RTX prerequisites and compute capability](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/installing-tensorrt-rtx/prerequisites.html), [NVIDIA CUDA on WSL guide](https://docs.nvidia.com/cuda/archive/13.2.0/wsl-user-guide/index.html), [PyTorch 2.7 Blackwell release notes](https://pytorch.org/blog/pytorch-2-7/), [PyTorch local installation selector](https://pytorch.org/get-started/locally/).

## Changes made

- Preserved the source archive and unmodified Drive notebooks.
- Added WSL environment check, isolated per-model environment bootstrap, model asset cache/download script, dataset handoff preparation, GPU diagnostics, model inference smoke validation, formal runner entry point, merge and analysis scripts, and local notebook generator.
- Added local paths/configuration, dependency input files, ignored data/weights/outputs, and GPU memory/environment fields in run manifests/logs.
- The model environment bootstrap uses one project-local venv per framework/model and requires pip to be inside a venv. It does not install Python packages globally or alter the installed Windows Python.
- Added single/batch adapter entry points. Smoke validation explicitly checks an actual single image and a two-image sequential batch and writes `06_logs/model_validation_<model>.json`.
- Added OOM recovery for per-image inference so one failed sample is logged and later images continue. Error/no-detection rows remain zero-scored in the original denominator; an incomplete run cannot enter accuracy merge, and CPU/failure runs cannot enter GPU timing rankings.

## Validation status

| Model | Source | Local load | Real single image | Two-image batch | GPU observed | Status |
|---|---|---:|---:|---:|---:|---|
| MediaPipe Pose Heavy | Drive notebook + source adapter; 30,664,242-byte cached task file | Not run | Not run | Not run | Not run | Blocked: WSL/runtime unavailable |
| MoveNet Thunder | Drive notebook + source adapter; 12,584,128-byte cached TFLite file | Not run | Not run | Not run | Not run | Blocked: WSL/runtime unavailable; GPU delegate must be detected or CPU explicitly recorded |
| RTMPose-M | Drive notebook ONNX Runtime CUDA flow; two cached ONNX files | Not run | Not run | Not run | Not run | Blocked: WSL/runtime unavailable |
| OpenPose COCO-18 | Drive Colab notebook; COCO-18 prototxt/caffemodel cached | Not run | Not run | Not run | Not run | Blocked: original notebook relies on cached Colab Ubuntu binary; no local adapter/build yet |

No model is labeled passed. The local handoffs were imported and verified: COCO source index SHA-256 `ec067b4cda12ba7f57a038d1d27ee545b2c6954eb5740c997396777a4d295ccb` (921 rows); SyRIP source index SHA-256 `d5b807198d123a86b6fac15fb5f6913b8fd503dc537fb03c378b57246b2c6e64` (100 rows). Only image paths changed from Colab absolute paths to dataset-relative paths; original CSVs remain as `sample_index_colab.csv`, and portable manifests retain `source_index_sha256`. Historical outputs are in ignored `03_model_raw_outputs/legacy_colab_archive/`, `04_merged_results/legacy_colab_archive/`, and `05_analysis_results/legacy_colab_archive/`; `06_logs/historical_colab_index.json` maps archived output files to original Drive paths. `reports/result_comparison.md` records numerical local-vs-Colab deltas as pending real local inference.

## Remaining host-level requirement

Installing the WSL2 optional feature/Ubuntu may require administrator approval and a Windows reboot. This environment does not have an Ubuntu distribution to install packages into, and the available Python 3.9 is below the current PyTorch Windows minimum (3.10). The project includes WSL setup and validation scripts, but model inference must wait until WSL2 and Python 3.11 are available. The local Git commit is independent of WSL. No remote URL is configured, so no GitHub push was attempted.
