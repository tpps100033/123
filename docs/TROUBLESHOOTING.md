# Troubleshooting

## WSL cannot see the GPU

Run `nvidia-smi` inside Ubuntu. WSL should use the NVIDIA Windows driver supplied by the host. Do not install a Linux display driver in the WSL distribution. If the command is missing or reports no device, update the Windows NVIDIA driver and check `wsl --status` / `wsl --list --verbose` from PowerShell.

## Sapiens checkpoint is missing

The Drive connector limits individual downloads to 256 MiB, while the source Sapiens TorchScript checkpoint is 4,676,154,239 bytes and is not publicly downloadable. Download it through an authenticated Drive client, then place it at `~/.local/share/infantpose/model_cache/sapiens_1b/sapiens_1b_coco_best_coco_AP_821_torchscript.pt2` in Ubuntu. Check the byte size before running `python scripts/validate_model.py --model sapiens_1b_coco --dataset coco_singleperson_val2017`. Sapiens has its own `infantpose-sapiens` kernel and venv; do not install its packages in the other model environments.

## OpenPose build fails before compilation

Run `bash scripts/build_openpose_cuda.sh` from Ubuntu after installing the native OpenPose development headers listed in [README.md](../README.md). The build cache, CUDA wheels, pybind11 checkout, and extension are under `~/.local/share/infantpose/`; model weights stay in the project cache. The build uses CUDA Toolkit 13.2.1 and must generate `sm_120` flags for the RTX 5060. If the local OpenPose source or submodules changed, the script stops when its compatibility patches no longer match.

## `pyopenpose` crashes or cannot import

Use the `infantpose-openpose` kernel or the `openpose` WSL venv. Do not import `cv2` in the same process: the OpenPose binary links Ubuntu's OpenCV, while `opencv-python` wheels bundle a second OpenCV build. The OpenPose venv intentionally omits that wheel and uses Pillow to read inputs. `run_all_research.sh` sets the compiled module, model, and shared-library paths for the OpenPose run.

## OpenPose runs out of VRAM

The verified default is fixed `480x368` with one image in flight. This uses more than 2 GiB less VRAM than the earlier dynamic-width configuration on this RTX 5060. If other applications are using the GPU, lower the input before running:

```bash
INFANTPOSE_OPENPOSE_NET_RESOLUTION=368x368 python scripts/validate_model.py --model openpose_coco18 --dataset coco_singleperson_val2017
INFANTPOSE_OPENPOSE_NET_RESOLUTION=368x368 python scripts/run_research.py --model openpose_coco18 --dataset coco_singleperson_val2017
```

The input size is part of model provenance and changes inference behavior, so record it when comparing scores. A native CUDA abort leaves `predictions.partial.jsonl` with completed images; partial runs are not scored or merged. A successful retry creates a separate run ID.

## A run stops at the thermal limit

The runner defaults to an 85°C GPU stop limit and checks temperature after each image. Lower the limit when you want more cooling margin:

```bash
INFANTPOSE_MAX_GPU_TEMP_C=75 python scripts/run_research.py --model openpose_coco18 --dataset coco_singleperson_val2017
```

Thermally stopped runs retain predictions and telemetry but receive no accuracy score. MyASUS performance and fan profiles are host settings; WSL cannot report fan speed.
