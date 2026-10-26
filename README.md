# InfantPose-Bench — Windows 11 + WSL2 migration

## Updated research direction

The next study compares the zero-shot performance of the six existing pose systems on synthetic human imagery from multiple synthesis eras and pipelines. The previous COCO, SyRIP, and real-child runs are retained as historical baselines; they are not being relabeled as synthetic results. One 300-image NameFrame pilot has been downloaded and scored; broader cross-source acquisition and comparison remain pending. Candidate datasets, annotation limitations, access requirements, and the sampling/evaluation contract are documented in [docs/SYNTHETIC_DATASET_SELECTION.md](docs/SYNTHETIC_DATASET_SELECTION.md). This pilot does not complete the multi-source study.

For an early pipeline check, the public Simuletic CCTV fall sample is configured as a 111-image pilot with exact COCO-17 labels. All six adapters completed the one-person-per-image protocol with official COCO keypoint evaluation; four used CUDA and two recorded CPU fallback reasons. Results and limitations are in [reports/simuletic_cctv_singleperson_pilot_2026-10-26.md](reports/simuletic_cctv_singleperson_pilot_2026-10-26.md). It is deliberately excluded from the primary general-human cohort because it is small and fall-dominated. CURDIE3 is a stronger full-study candidate but its HF repository is 919 GB; it was not downloaded under the current network budget.

The SURREAL archive endpoint returned HTTP 403, so that source was replaced for the active run. The public NameFrame Quarry Crowd sample (300 frames) is converted into COCO-17 slots and scored with the official COCO evaluator. It provides up to 12 body joints; nose, eyes, and ears remain unlabeled, and source visibility flags are preserved. All six models completed the 300-image run. The report records the unusually distant/crowded scene, near-duplicates, source audit discrepancy, model scores, and CPU/GPU execution. See [the NameFrame pilot report](reports/nameframe_quarry_crowd_pilot_2026-10-27.md) and [dataset selection notes](docs/SYNTHETIC_DATASET_SELECTION.md).

This project keeps the Colab research artifacts intact and adds local WSL2 entry points. The fixed benchmark contract remains batch 1, 10 warm-up images, FP32, with AMP, compilation, quantization, and TensorRT disabled. Timing includes image decode, preprocessing, inference, and postprocessing. Accuracy uses the official `cocodataset/cocoapi` `COCOeval(iouType='keypoints')` implementation with its default OKS thresholds, keypoint sigmas, area ranges, recall thresholds, and `maxDets=20`. COCO-subset AP is reported as subset AP, not as a full-val2017 leaderboard score. SyRIP uses the same evaluator and is labeled as a custom-domain COCO-OKS evaluation. Pose coverage is diagnostic only, not an accuracy metric.

## Current state

The RTX 5060 Laptop GPU is visible inside WSL2 Ubuntu 24.04.5 with 8,151 MiB VRAM. Python 3.12 and isolated model environments are installed under the WSL Linux home directory (not in Windows Python and not inside the project). Six models passed real single-image and two-image inference validation; four used the RTX 5060 through CUDA, while MediaPipe and MoveNet used CPU because their GPU runtimes are unavailable in this WSL setup. All six models completed COCO (921 images), SyRIP infant (100 images), and COCO-UniHuman apparent-age child (134 crops). Child images were deduplicated against COCO by source image ID and source-file SHA-256. SVIRO BMW i3 runs and outputs were removed from the active project at the user's request.

The Sapiens checkpoint was found in the local `姿態判定` Downloads folder and copied into the ignored WSL model cache. Both original 4,676,154,239-byte downloads remain untouched. Sapiens passed single-image and batch inference on CUDA and completed COCO 921/921 and SyRIP 100/100.

## Completed benchmark domains

The three evaluation domains are COCO single-person val2017 (921 images), SyRIP infant RGB (100 images), and COCO-UniHuman apparent-age children (134 crops). The child cohort is COCO-derived, with one selected child per unique source image; all 921 project COCO evaluation image IDs are excluded. The annotations already use the COCO-17 schema, so the conversion only translates coordinates into each person crop. Ages 3–17 are manually assessed apparent ages, not verified birth ages. BabyPose and SVIRO are excluded from active comparisons. Dataset provenance, selection, and limitations are recorded in [docs/REAL_CHILD_DATASET_AVAILABILITY.md](docs/REAL_CHILD_DATASET_AVAILABILITY.md).


## Setup

1. On Windows, run `powershell -ExecutionPolicy Bypass -File .\scripts\check_environment.ps1`.
2. Install WSL2 and Ubuntu if the check reports that no distribution is installed. This changes Windows optional features and may require administrator approval and a reboot. The existing Windows NVIDIA driver is used by WSL; do not install a Linux NVIDIA display driver inside Ubuntu.
3. In Ubuntu 24.04, install OpenPose's native build headers once: `sudo apt-get install libprotobuf-dev protobuf-compiler libgoogle-glog-dev libgflags-dev libopencv-dev libboost-all-dev libhdf5-dev libatlas-base-dev libopenblas-dev libeigen3-dev libgoogle-perftools-dev libtbb-dev`.
4. Run `bash scripts/setup_wsl.sh` from this project directory. It creates isolated MediaPipe, MoveNet, RTMPose, YOLO, Sapiens, and OpenPose runtime environments under `~/.local/share/infantpose/venvs/`, installs `pycocotools` in each for official evaluation, registers one kernel per environment, and records resolved package versions under `requirements/locks/`. The setup enforces `PIP_REQUIRE_VIRTUALENV` and never installs Python model packages into global or Windows Python.
5. Run `bash scripts/build_openpose_cuda.sh` to build official OpenPose v1.7.0 plus its Python CUDA extension in the isolated WSL cache. The script pins CUDA Toolkit 13.2.1 and pybind11 v2.13.6, targets `sm_120`, checks the cached weight hash, applies the recorded CUDA 13 compatibility patches, and limits compilation to two jobs by default.
6. Run `python scripts/gpu_diagnostics.py` inside each kernel. An import by itself is not a GPU validation: the diagnostic runs a CUDA matrix operation and records the actual framework device/provider.
7. The source COCO/SyRIP archives and annotations, plus the COCO-UniHuman v2 age/keypoint annotation file, are cached locally. `scripts/prepare_datasets.py` verifies existing COCO/SyRIP handoffs and creates the 134-image, deduplicated child cohort. Child image crops use Pillow; install it into a small isolated Windows preparation environment with `py -3.9 -m venv .venvs/dataset-prep` and `.venvs/dataset-prep/Scripts/python.exe -m pip install -r requirements/dataset-prep.txt`, then prepare with `.venvs/dataset-prep/Scripts/python.exe scripts/prepare_datasets.py --dataset coco_unihuman_child_val134`. Keep all data, weights, raw predictions, logs, and generated reports outside Git. The Sapiens checkpoint belongs in the ignored WSL cache path listed above; never commit a real token or checkpoint.
8. Run `python scripts/validate_model.py --model <model> --dataset coco_singleperson_val2017` for an actual model smoke test. Then run an individual dataset with `python scripts/run_research.py --model <model> --dataset <dataset>`. GPU runs stop at `INFANTPOSE_MAX_GPU_TEMP_C` (85°C by default) and check temperature after every image. A thermally stopped run is never scored; per-image prediction checkpoints remain available if a native runtime aborts.
9. Run local notebooks `90` and `91` or `scripts/merge_results.py` and `scripts/analyse_results.py` to merge and analyze complete COCO, SyRIP, and child runs. The merge expects six models × three datasets, with exactly 921, 100, and 134 images respectively.


See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for WSL, CUDA, OpenPose memory, and thermal guidance.

Launch JupyterLab from the project root with the environment activated, then open `notebooks/local/`. Original Colab notebooks are kept under `notebooks/original/` and excluded from Git; local entry points are generated by `python tools/build_local_notebooks.py`.

## Commands

```bash
bash scripts/setup_wsl.sh
bash scripts/build_openpose_cuda.sh
python scripts/gpu_diagnostics.py
python scripts/prepare_datasets.py
.venvs/dataset-prep/Scripts/python.exe scripts/prepare_datasets.py --dataset coco_unihuman_child_val134
python scripts/prepare_simuletic_pilot.py
INFANTPOSE_MAX_GPU_TEMP_C=85 bash scripts/run_all_models_for_dataset.sh simuletic_cctv_fall_singleperson_pilot111
python scripts/prepare_nameframe_quarry_crowd.py
INFANTPOSE_MAX_GPU_TEMP_C=85 bash scripts/run_all_models_for_dataset.sh nameframe_quarry_crowd_singleperson_pilot300
python scripts/validate_model.py --model yolo11x_pose --dataset simuletic_cctv_fall_pilot111
INFANTPOSE_MAX_GPU_TEMP_C=85 bash scripts/run_all_models_for_dataset.sh simuletic_cctv_fall_pilot111
python scripts/prepare_model_assets.py
python scripts/validate_model.py --model mediapipe_pose_heavy --dataset coco_singleperson_val2017
python scripts/validate_model.py --model sapiens_1b_coco --dataset coco_singleperson_val2017
python scripts/run_research.py --model mediapipe_pose_heavy --dataset coco_singleperson_val2017
INFANTPOSE_MAX_GPU_TEMP_C=85 bash scripts/run_all_models_for_dataset.sh coco_unihuman_child_val134
bash scripts/run_all_research.sh
python scripts/merge_results.py
python scripts/analyse_results.py
python tools/build_local_notebooks.py
```

Each model runtime has its own environment because TensorFlow, MediaPipe, ONNX Runtime, and PyTorch stacks can constrain conflicting dependencies. The PyTorch setup uses the official PyTorch 2.12.0 / torchvision 0.27.0 CUDA 13.2 wheels, matching the RTX 5060 Blackwell architecture. ONNX Runtime CUDA/cuDNN libraries are installed as RTMPose venv dependencies. Resolved package versions are recorded in the lock snapshots.

`run_all_research.sh` executes all six local model adapters against COCO, SyRIP, and the child cohort sequentially with an 85°C cutoff by default and checks temperature after every image. Sapiens polls temperature from its streamed official-demo progress output and terminates the child process at the cutoff. Override with `INFANTPOSE_MAX_GPU_TEMP_C` and `INFANTPOSE_TEMP_CHECK_INTERVAL`; a thermal stop exits the script. OpenPose runs in its own venv and keeps one image in flight to limit VRAM use. Its Python adapter uses PIL and deliberately does not import the OpenCV Python wheel, which conflicts with the system OpenCV linked into OpenPose.

## Outputs and cache

- `02_datasets/model_cache/`: reusable official model assets and SHA-256 inventory.
- `02_datasets/canonical/`: extracted immutable inputs and hashed handoffs.
- `03_model_raw_outputs/<model>/<dataset>/fair_reference/<run-id>/`: predictions, per-image metrics, and run manifest.
- `04_merged_results/` and `05_analysis_results/`: reproducible CSV/JSON summaries.
- `06_logs/`: GPU diagnostics, per-model validation, and append-only run summaries.

Research results are not regenerated from imported historical summaries. `90` only considers complete new fair-reference runs with successful COCOeval output. COCO and SyRIP handoffs retain their original sample selection and mapping. The child handoff excludes all 921 COCO comparison image IDs and checks source-image SHA-256 values against COCO and within the child cohort; if a source annotation or image is invalid, preparation stops instead of substituting a different sample.
