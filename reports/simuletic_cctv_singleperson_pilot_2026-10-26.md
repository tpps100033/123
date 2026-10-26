# Simuletic synthetic fall-pose pilot — 2026-10-26

## Scope and provenance

This is an execution and adapter-validation pilot, not a general synthetic-human benchmark. The source is the public [Simuletic CCTV Fall/Lying Down Detection HF sample](https://huggingface.co/datasets/Simuletic/CCTV_Incident_Dataset_Fall_Lying_Down_Detection), cloned at revision `143e1d1349d3467e64fc79373b274f3476fc382e`. The source card declares CC BY 4.0; retain attribution to Simuletic and check the source card before redistribution. The local clone was about 172 MB.

The free sample contains 111 PNG images and 121 labeled people. Its labels already contain COCO-order 17-point normalized `xyv` keypoints. The preparation script converts normalized coordinates to image pixels and COCO JSON without changing keypoint semantics. The benchmark's established single-person contract selects the largest labeled bounding box in each image. Consequently, this scored handoff has 111 images/people, all labeled as lying; ten additional standing labels in the source were not selected. The sample is therefore strongly pose/domain-skewed and cannot answer broad synthetic-human generalization questions or satisfy the planned 300+ image cohort size.

The frozen handoff and its hashes are under `02_datasets/canonical/simuletic_cctv_fall_singleperson_pilot111/`; source data remains in the ignored download cache. Results below use the same selected list for all models.

## Protocol and machine

- Evaluator: official `cocodataset/cocoapi` `COCOeval(iouType='keypoints')`, default keypoint parameters, OKS AP over IoU 0.50:0.95, plus AP50, AP75, and AR.
- Inference contract: one image at a time, 10 warm-up images excluded from timing, FP32; no AMP, compilation, quantization, or TensorRT. Latency includes decode, preprocessing, inference, and postprocessing.
- Machine: Windows 11 with Ubuntu 24.04.5 under WSL2, NVIDIA GeForce RTX 5060 Laptop GPU (8 GB class, compute capability 12.0).
- Thermal protection: 85°C stop threshold checked per image. No run was thermally stopped. Peak observed during the six-model sequence was 78°C (Sapiens manifest); independent live monitoring observed 77°C. The GPU returned to about 76°C at the end of Sapiens.
- Every model loaded, completed all 111 images, wrote a complete official evaluation, and recorded zero inference errors. “No detection” below is distinct from an execution error.

## Results

| Model | AP | AP50 | AP75 | AR | No detection | Runtime device | Median / p95 ms per image |
|---|---:|---:|---:|---:|---:|---|---:|
| MediaPipe Pose Heavy | 0.9088 | 0.9738 | 0.9595 | 0.9288 | 1 | CPU | 221.26 / 251.73 |
| MoveNet Thunder | 0.5866 | 0.7768 | 0.6384 | 0.6586 | 0 | CPU | 79.25 / 94.89 |
| RTMPose-M | 0.9353 | 0.9901 | 0.9792 | 0.9495 | 1 | RTX 5060 CUDA EP + CPU EP | 91.82 / 111.76 |
| YOLO11x-Pose | 0.8034 | 0.8973 | 0.8530 | 0.8369 | 0 | RTX 5060 CUDA | 92.75 / 107.00 |
| Sapiens 1B COCO | 0.8750 | 0.9766 | 0.9367 | 0.8919 | 0 | RTX 5060 CUDA | 67.29 / 83.18 |
| OpenPose COCO-18 | 0.7534 | 0.8638 | 0.8148 | 0.7928 | 0 | RTX 5060 CUDA | 159.67 / 193.48 |

The four CUDA runtimes were confirmed by runtime/provider provenance (OpenPose's own CUDA backend, RTMPose CUDAExecutionProvider, PyTorch CUDA for YOLO and Sapiens). MediaPipe fell back to CPU because its WSL graph did not expose a supported physical GPU service. MoveNet fell back to CPU because the TensorFlow Lite GPU delegate shared library is unavailable. Thus, GPU-capable paths ran on the RTX 5060; the two adapters with unavailable GPU backends were still exercised and their fallback reasons recorded. CPU and GPU latency numbers should not be treated as a hardware-normalized speed ranking.

All 111 people are large under the official COCO keypoint area bins, so the evaluator's medium-person AP/AR is undefined (`-1` in the console / null in manifests). This is an expected consequence of the sample, not a missing evaluation.

## Reproducibility and interpretation

Run from the repository root in WSL2:

```bash
python scripts/prepare_simuletic_pilot.py
INFANTPOSE_MAX_GPU_TEMP_C=85 bash scripts/run_all_models_for_dataset.sh simuletic_cctv_fall_singleperson_pilot111
```

Per-model `run_manifest.json`, predictions, latency, runtime provenance, and error counts are in `03_model_raw_outputs/<model>/simuletic_cctv_fall_singleperson_pilot111/fair_reference/<run-id>/`. Generated data, downloaded images, and predictions are intentionally ignored by Git.

An earlier exploratory run scored all 121 source people, while these adapters emit one pose per image. Its numbers are not protocol-aligned and are not included here. This report is not directly comparable to COCO val2017, SyRIP, real-child results, or another synthetic dataset: it describes only this 111-image fall-pose pilot. The planned study still needs at least three independently sourced synthetic datasets with 300+ audited images each. CURDIE3 was not downloaded because its reported 919 GB archive is impractical under the stated daily network budget. No claim of research completion or general synthetic-domain ranking is made from this pilot.
