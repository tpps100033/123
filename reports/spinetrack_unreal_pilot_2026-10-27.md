# SpineTrack-Unreal synthetic pose pilot — 2026-10-27

## Dataset choice and provenance

SpineTrack-Unreal is a stronger match for this study than the earlier NameFrame Quarry Crowd sample: it contains 25,000 single-person Unreal Engine synthetic frames and its published skeleton includes all 17 COCO body keypoints, plus 20 additional anatomical points. The authors describe a 25,000-image Unreal Engine synthetic subset with OpenSim-aligned labels in the [paper](https://arxiv.org/abs/2504.08110). The source archive also contains real subsets; this experiment uses only `train-unreal`.

- Source: [Hugging Face dataset card and files](https://huggingface.co/datasets/saifkhichi96/spinetrack)
- Keypoint order: [authors' official `metainfo.py`](https://github.com/dfki-av/spinepose/blob/main/src/spinepose/metainfo.py)
- License: CC BY-NC 4.0; attribution is required and commercial use is excluded. The source archive is not included in Git or the paper bundle.
- Source image archive: 19,407,030,602 bytes (about 19.4 GB decimal), below the user's stated 30 GB/day transfer allowance. Expected SHA-256: `4212a2b5c9925e5ff7cfe8f3e28f5af9b6319b748edb635d407be08a08fe35c7`.
- Source annotation archive: 24,789,197 bytes. Expected SHA-256: `3032ffad3984f91bb462bbf6cdeeed392a21f986cfd76d023aa6fb7ae17e5f6b`.
- Hugging Face revision: `70ec456`.

## Annotation and sample protocol

The authors' metainfo explicitly assigns source indices 0–16 to `nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle`. These 17 coordinates and visibility values are copied without interpolation. The other 20 points are kept in a separate local source-audit JSON and excluded from COCO scoring. All model results use the unmodified benchmark's official `pycocotools.COCOeval(iouType="keypoints")` defaults.

The sample target is 1,000 images, selected with seed `20261027` by round-robin across source motion-sequence folders, requiring a five-frame gap within each sequence. Before sampling, the 54 source records containing any labeled COCO-17 point outside the image are excluded as invalid under their `v=2` labels; coordinates and visibility are never repaired. Exact duplicate image bytes are skipped. One person is annotated per image, matching the existing single-person adapter contract. The sample index and each extracted image receive SHA-256 values. Images and model weights remain local and ignored by Git.

The full source annotation audit found 25,000 images and 25,000 linked person annotations across 159 source-sequence folders. Source `num_keypoints` agrees with the number of labeled points on every annotation, and there are no non-finite keypoint coordinates. Among the first 17 points, visibility values are 0 or 2. The source contains 218 labeled COCO point occurrences outside image bounds, affecting 54 annotations; 661 person boxes also extend beyond image bounds. Since COCO v=2 denotes a labeled point visible in the image, those 54 structurally invalid records are excluded before sampling, without editing any retained coordinates or visibility values. The resulting 1,000-image sample has no out-of-bounds labeled COCO points; 41 retained source person boxes still extend beyond the frame and are preserved. It covers 157 source sequences. COCO-17 point coverage is 730 people with 17 labeled points, 208 with 16, 34 with 15, 14 with 14, 6 with 13, and 8 with 12. Median source bbox area is 15.4% of the image and median bbox height is 73.2% of image height. A 20-image overlay contact sheet was visually reviewed: labels broadly align with visible body joints, though composite backgrounds and occlusions vary. The dataset maintainers warn that a small number of synthetic annotations may be corrupted, so subtle errors may remain. The immutable index SHA-256 is `a2f20ab44fa9e2289923079847f31e9788cf05c81c50bf4c34b24648f33dc01d`.

## Six-model run

Execution status: **complete**. All six models processed the 1,000-image handoff and produced complete official COCO keypoint evaluations. No model had inference errors or thermal stops. The full run took about 50 minutes; the GPU temperature observed in per-image logs peaked at 73°C, below the 85°C stop limit.

The run will use the established frozen six-model `fair_reference` profile (batch 1, FP32, 10 warm-up images), official COCO keypoint AP/AR, and an 85°C GPU stop limit. Models run sequentially to limit heat and VRAM pressure. MediaPipe and MoveNet may use CPU if their installed WSL runtimes lack GPU support; device evidence will be recorded per model.

| Model | COCO AP | AP50 | AP75 | AR | Successful predictions | Device | Peak GPU temperature |
|---|---:|---:|---:|---:|---:|---|---:|
| MediaPipe Pose Heavy | 0.231 | 0.684 | 0.154 | 0.404 | 924/1,000 | CPU (GPU unsupported in WSL graph) | 53°C |
| MoveNet Thunder | 0.005 | 0.037 | 0.000 | 0.022 | 1,000/1,000 | CPU (TensorFlow Lite GPU delegate unavailable) | 52°C |
| RTMPose-M | 0.282 | 0.848 | 0.171 | 0.473 | 999/1,000 | CUDA provider requested; see runtime caveat below | 64°C |
| YOLO11x-Pose | **0.292** | 0.827 | **0.186** | 0.466 | 1,000/1,000 | RTX 5060 CUDA | 66°C |
| Sapiens 1B COCO | 0.088 | 0.252 | 0.064 | 0.197 | 1,000/1,000 | RTX 5060 CUDA | 56°C |
| OpenPose COCO18 | 0.196 | 0.561 | 0.155 | 0.360 | 999/1,000 | RTX 5060 CUDA | **73°C** |

All six manifests report `metric_complete=true`, 1,000 selected images, zero inference errors, and no thermal stop. The one missing prediction for RTMPose and OpenPose is recorded as no detection, not a runtime error. MediaPipe also completed without errors but found no pose on 76 images. The open-source PyTorch CUDA runtime is confirmed for YOLO and Sapiens; OpenPose is built and configured for CUDA and completed the run. RTMPose's ONNX sessions list CUDA and CPU providers, while its logs also emit an ONNX Runtime warning about no registered CUDA provider device. Its manifest therefore records CUDA as requested/configured; treat actual GPU kernel execution as not fully independently confirmed until its provider is repaired or profiled.

Median end-to-end latency per image (decode + preprocessing + inference + postprocessing) was 96.7 ms for MediaPipe, 70.3 ms for MoveNet, 93.1 ms for RTMPose, 81.6 ms for YOLO, 53.5 ms for Sapiens, and 141.3 ms for OpenPose. These timing values are descriptive: MediaPipe and MoveNet ran on CPU, and RTMPose's CUDA execution is uncertain, so cross-device speed ordering is not a hardware-controlled comparison. Predictions, per-image metrics, environment, device, thermal, and official COCOeval metadata are in each local ignored run directory under `03_model_raw_outputs/<model>/spinetrack_unreal_singleperson_pilot1000/fair_reference/`.

## Interpretation

For context, the existing 921-image COCO val2017 single-person results under the same COCOeval setup were: MediaPipe 0.420 AP, MoveNet 0.103, RTMPose 0.779, YOLO 0.718, Sapiens 0.536, and OpenPose 0.589. The SpineTrack sample is lower for all six (AP differences, SpineTrack minus COCO: −0.189, −0.098, −0.497, −0.426, −0.448, and −0.393 respectively). This consistent decline is compatible with a synthetic-to-real/domain and annotation-convention gap, but does not isolate its cause: model predictions are matched to only one selected synthetic person, source point visibility is binary 0/2, and the cohorts differ in scene composition and selection. It is not a controlled causal comparison.

On this cohort, YOLO11x-Pose and RTMPose-M lead by COCO AP (0.292 and 0.282); MediaPipe follows at 0.231. OpenPose reaches 0.196, Sapiens 0.088, and MoveNet 0.005. The large AP50-to-AP75 drops for several models indicate that many detected poses are approximately aligned but not localized precisely enough at stricter OKS thresholds. MoveNet's near-zero result and MediaPipe's 76 missed images warrant per-joint and overlay error analysis before drawing model-quality conclusions; they are not evidence that the dataset labels should be changed.

This is a frozen-model, zero-shot result on one Unreal Engine / biomechanical synthesis pipeline and a sequence-stratified sample. It is not a general ranking over synthetic image techniques or eras, nor a direct substitute for COCO val2017. Comparison with NameFrame is exploratory because NameFrame lacks five face landmarks and depicts a distant crowd. The COCO evaluation definition was not changed. No original Colab baseline was available for this exact SpineTrack sample, so numerical Colab-to-local equivalence is not claimed. The small number of source annotation errors noted by the maintainers remains a limitation.
