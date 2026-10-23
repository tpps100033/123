# InfantPose-Bench — Windows 11 + WSL2 migration

This project keeps the Colab research artifacts intact and adds local WSL2 entry points. The fixed benchmark contract remains batch 1, 10 warm-up images, FP32, with AMP, compilation, quantization, and TensorRT disabled. Timing includes image decode, preprocessing, inference, and postprocessing. Accuracy uses the official `cocodataset/cocoapi` `COCOeval(iouType='keypoints')` implementation with its default OKS thresholds, keypoint sigmas, area ranges, recall thresholds, and `maxDets=20`. COCO-subset AP is reported as subset AP, not as a full-val2017 leaderboard score. SyRIP uses the same evaluator and is labeled as a custom-domain COCO-OKS evaluation. Pose coverage is diagnostic only, not an accuracy metric.

## Current state

The RTX 5060 Laptop GPU is visible inside WSL2 Ubuntu 24.04.5 with 8,151 MiB VRAM. Python 3.12 and isolated model environments are installed under the WSL Linux home directory (not in Windows Python and not inside the project). Six models passed real single-image and two-image inference validation; four used the RTX 5060 through CUDA, while MediaPipe and MoveNet used CPU because their GPU runtimes are unavailable in this WSL setup. COCO (921 images) and SyRIP (100 infant images) are the primary completed image domains. SVIRO's 53-scene BMW i3 result is a supplementary synthetic stress test, not a real-child benchmark; 710 scenes are an annotation-only inventory. Research into a separately labeled real-child image benchmark and its COCO-17 mapping is documented in [docs/REAL_CHILD_DATASET_AVAILABILITY.md](docs/REAL_CHILD_DATASET_AVAILABILITY.md).

The Sapiens checkpoint was found in the local `姿態判定` Downloads folder and copied into the ignored WSL model cache. Both original 4,676,154,239-byte downloads remain untouched. Sapiens passed single-image and batch inference on CUDA and completed COCO 921/921 and SyRIP 100/100.

## Three benchmark domains

The completed image comparisons are COCO single-person val2017 (921 images), SyRIP infant RGB (100 images), and supplementary SVIRO child RGB (53 synthetic BMW i3 scenes). An annotation-only inventory identified 710 SVIRO child scenes across ten vehicle splits, but only the BMW i3 images were prepared and scored. BabyPose is excluded from execution and comparison because all six RGB pose models scored AP=0 on its depth frames. A real-child image benchmark with pose ground truth remains unavailable among the public candidates reviewed; see the candidate analysis and exact COCO-17 mapping contract in [docs/REAL_CHILD_DATASET_AVAILABILITY.md](docs/REAL_CHILD_DATASET_AVAILABILITY.md). Do not present the SVIRO results as evidence on real children outside a vehicle interior.

The supplementary synthetic child subset comes from the official [SVIRO dataset](https://sviro.kl.dfki.de/), licensed CC BY-NC-SA 4.0. The completed result uses 53 held-out BMW i3 scenes with exactly one class-2 child and no adult or infant. A separate scan found 710 candidate scenes across ten vehicle interiors; their additional RGB images and model runs are not completed. Twelve released limb joints map to COCO-17; face joints are unlabelled. These results are restricted to the synthetic vehicle-interior domain. See [SVIRO protocol](docs/SVIRO_CHILDREN.md), [SVIRO comparison report](reports/sviro_child_comparison.md), and [real-child dataset availability and mapping analysis](docs/REAL_CHILD_DATASET_AVAILABILITY.md).

## Setup

1. On Windows, run `powershell -ExecutionPolicy Bypass -File .\scripts\check_environment.ps1`.
2. Install WSL2 and Ubuntu if the check reports that no distribution is installed. This changes Windows optional features and may require administrator approval and a reboot. The existing Windows NVIDIA driver is used by WSL; do not install a Linux NVIDIA display driver inside Ubuntu.
3. In Ubuntu 24.04, install OpenPose's native build headers once: `sudo apt-get install libprotobuf-dev protobuf-compiler libgoogle-glog-dev libgflags-dev libopencv-dev libboost-all-dev libhdf5-dev libatlas-base-dev libopenblas-dev libeigen3-dev libgoogle-perftools-dev libtbb-dev`.
4. Run `bash scripts/setup_wsl.sh` from this project directory. It creates isolated MediaPipe, MoveNet, RTMPose, YOLO, Sapiens, and OpenPose runtime environments under `~/.local/share/infantpose/venvs/`, installs `pycocotools` in each for official evaluation, registers one kernel per environment, and records resolved package versions under `requirements/locks/`. The setup enforces `PIP_REQUIRE_VIRTUALENV` and never installs Python model packages into global or Windows Python.
5. Run `bash scripts/build_openpose_cuda.sh` to build official OpenPose v1.7.0 plus its Python CUDA extension in the isolated WSL cache. The script pins CUDA Toolkit 13.2.1 and pybind11 v2.13.6, targets `sm_120`, checks the cached weight hash, applies the recorded CUDA 13 compatibility patches, and limits compilation to two jobs by default.
6. Run `python scripts/gpu_diagnostics.py` inside each kernel. An import by itself is not a GPU validation: the diagnostic runs a CUDA matrix operation and records the actual framework device/provider.
7. The source COCO/SyRIP archives, annotations, selected images, handoff manifests, and available model weights have already been imported from the user-provided Drive project. `scripts/prepare_datasets.py` verifies and reuses the COCO, SyRIP, and SVIRO handoffs. Keep all data, weights, raw predictions, logs, and generated reports outside Git. The Sapiens checkpoint belongs in the ignored WSL cache path listed above; never commit a real token or checkpoint.
8. Run `python scripts/validate_model.py --model <model> --dataset coco_singleperson_val2017` for an actual model smoke test. Then run an individual dataset with `python scripts/run_research.py --model <model> --dataset <dataset>`. GPU runs stop at `INFANTPOSE_MAX_GPU_TEMP_C` (85°C by default) and check temperature after every image. A thermally stopped run is never scored; per-image prediction checkpoints remain available if a native runtime aborts.
9. Run local notebooks `90` and `91` or `scripts/merge_results.py` and `scripts/analyse_results.py` to merge and analyze the original COCO/SyRIP results. Those frozen runs require exactly 921 COCO rows or 100 SyRIP rows and the original verified COCO-17 mapping. The supplementary synthetic SVIRO comparison is documented separately because it uses a distinct child-only test split.

The SVIRO annotation-to-COCOeval path can be checked with `py -3.9 scripts/verify_sviro_coco_metric.py --dataset sviro_child_i3_test53_coco12`. It uses ground-truth coordinates as an oracle and must produce AP=1 and AR=1 across all 53 BMW i3 images; it is an evaluator integrity check, not a model score. The 710-scene all-vehicle inventory is exploratory only; its images and model results were not prepared.

See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for WSL, CUDA, OpenPose memory, and thermal guidance.

Launch JupyterLab from the project root with the environment activated, then open `notebooks/local/`. Original Colab notebooks are kept under `notebooks/original/` and excluded from Git; local entry points are generated by `python tools/build_local_notebooks.py`.

## Commands

```bash
bash scripts/setup_wsl.sh
bash scripts/build_openpose_cuda.sh
python scripts/gpu_diagnostics.py
python scripts/prepare_datasets.py
py -3.9 scripts/prepare_sviro_child.py
python scripts/prepare_model_assets.py
python scripts/validate_model.py --model mediapipe_pose_heavy --dataset coco_singleperson_val2017
python scripts/validate_model.py --model sapiens_1b_coco --dataset coco_singleperson_val2017
python scripts/run_research.py --model mediapipe_pose_heavy --dataset coco_singleperson_val2017
bash scripts/run_all_research.sh
python scripts/merge_results.py
python scripts/analyse_results.py
python scripts/compare_sviro_child.py
python tools/build_local_notebooks.py
```

Each model runtime has its own environment because TensorFlow, MediaPipe, ONNX Runtime, and PyTorch stacks can constrain conflicting dependencies. The PyTorch setup uses the official PyTorch 2.12.0 / torchvision 0.27.0 CUDA 13.2 wheels, matching the RTX 5060 Blackwell architecture. ONNX Runtime CUDA/cuDNN libraries are installed as RTMPose venv dependencies. Resolved package versions are recorded in the lock snapshots.

`run_all_research.sh` executes all six local model adapters against COCO, SyRIP, and the 53-image SVIRO BMW i3 supplementary set sequentially with an 85°C cutoff by default and checks temperature after every image. Sapiens polls temperature from its streamed official-demo progress output and terminates the child process at the cutoff. Override with `INFANTPOSE_MAX_GPU_TEMP_C` and `INFANTPOSE_TEMP_CHECK_INTERVAL`; a thermal stop exits the script. OpenPose runs in its own venv and keeps one image in flight to limit VRAM use. Its Python adapter uses PIL and deliberately does not import the OpenCV Python wheel, which conflicts with the system OpenCV linked into OpenPose.

## Outputs and cache

- `02_datasets/model_cache/`: reusable official model assets and SHA-256 inventory.
- `02_datasets/canonical/`: extracted immutable inputs and hashed handoffs.
- `03_model_raw_outputs/<model>/<dataset>/fair_reference/<run-id>/`: predictions, per-image metrics, and run manifest.
- `04_merged_results/` and `05_analysis_results/`: reproducible CSV/JSON summaries.
- `06_logs/`: GPU diagnostics, per-model validation, and append-only run summaries.

Research results are not regenerated from imported historical summaries. `90` only considers complete new fair-reference runs with successful COCOeval output. The COCO/SyRIP handoffs must retain their original sample selection and mapping; if a source annotation is missing, preparation stops instead of substituting a different dataset.
