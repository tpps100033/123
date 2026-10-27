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

| Model | COCO OKS-AP | AP50 | AP75 | Mean OKS¹ | PCK@0.10² | PCK@0.20² | Pose coverage | Device | Peak GPU temperature |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| MediaPipe Pose Heavy | 0.231 | 0.684 | 0.154 | 0.610 | 0.617 | 0.713 | 92.4% | CPU (GPU unsupported in WSL graph) | 53°C |
| MoveNet Thunder | 0.005 | 0.037 | 0.000 | 0.324 | 0.238 | 0.733 | 100.0% | CPU (TensorFlow Lite GPU delegate unavailable) | 52°C |
| RTMPose-M | 0.282 | 0.848 | 0.171 | **0.677** | **0.665** | **0.767** | 99.9% | CUDA provider requested; see runtime caveat below | 64°C |
| YOLO11x-Pose | **0.292** | 0.827 | **0.186** | 0.670 | 0.660 | 0.762 | 100.0% | RTX 5060 CUDA | 66°C |
| Sapiens 1B COCO | 0.088 | 0.252 | 0.064 | 0.463 | 0.522 | 0.640 | 100.0% | RTX 5060 CUDA | 56°C |
| OpenPose COCO18 | 0.196 | 0.561 | 0.155 | 0.599 | 0.587 | 0.681 | 99.9% | RTX 5060 CUDA | **73°C** |

¹ Mean single-instance OKS is the mean over all 1,000 images using the official COCO keypoint sigmas, each ground-truth instance's annotated area, and all labeled (`v>0`) points; a no-detection image contributes zero. It is a descriptive localization statistic, not the official COCO AP. ² PCK is computed per labeled joint from Euclidean pixel error divided by `max(ground-truth bbox width, ground-truth bbox height)`. PCK@t counts errors `<= t`; a missing/no-detection pose counts incorrect for every labeled point in that image. The same 16,618 labeled keypoints form every model's PCK denominator. This is explicitly named bbox-long-side PCK because PCK variants use different person-scale normalizers; it is not MPII PCKh. The COCO primary keypoint measure remains OKS-based AP averaged over OKS thresholds 0.50–0.95 by the official evaluator. The saved analyzer reproduces these supplemental values from the frozen prediction files.

All six manifests report `metric_complete=true`, 1,000 selected images, zero inference errors, and no thermal stop. The one missing prediction for RTMPose and OpenPose is recorded as no detection, not a runtime error. MediaPipe also completed without errors but found no pose on 76 images. The open-source PyTorch CUDA runtime is confirmed for YOLO and Sapiens; OpenPose is built and configured for CUDA and completed the run. RTMPose's ONNX sessions list CUDA and CPU providers, while its logs also emit an ONNX Runtime warning about no registered CUDA provider device. Its manifest therefore records CUDA as requested/configured; treat actual GPU kernel execution as not fully independently confirmed until its provider is repaired or profiled.

Median end-to-end latency per image (decode + preprocessing + inference + postprocessing) was 96.7 ms for MediaPipe, 70.3 ms for MoveNet, 93.1 ms for RTMPose, 81.6 ms for YOLO, 53.5 ms for Sapiens, and 141.3 ms for OpenPose. These timing values are descriptive: MediaPipe and MoveNet ran on CPU, and RTMPose's CUDA execution is uncertain, so cross-device speed ordering is not a hardware-controlled comparison. Predictions, per-image metrics, environment, device, thermal, and official COCOeval metadata are in each local ignored run directory under `03_model_raw_outputs/<model>/spinetrack_unreal_singleperson_pilot1000/fair_reference/`.

## Interpretation

### COCO val2017 comparison

The six existing COCO val2017 runs were reprocessed with the same supplemental OKS/PCK formulas used above. The COCO comparison handoff contains one selected person annotation in each of 921 images. The values below show COCO → SpineTrack; PCK deltas are percentage points.

| Model | COCO OKS-AP → SpineTrack | Δ AP | Mean OKS → SpineTrack | PCK@0.10 → SpineTrack | Δ | PCK@0.20 → SpineTrack | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| MediaPipe | 0.418 → 0.231 | −0.187 | 0.551 → 0.610 | 70.9% → 61.7% | −9.2 pp | 75.5% → 71.3% | −4.2 pp |
| MoveNet Thunder | 0.103 → 0.005 | −0.098 | 0.453 → 0.324 | 61.9% → 23.8% | −38.1 pp | 84.8% → 73.3% | −11.5 pp |
| RTMPose-M | 0.775 → 0.282 | −0.492 | 0.841 → 0.677 | 93.9% → 66.5% | −27.4 pp | 96.0% → 76.7% | −19.3 pp |
| YOLO11x-Pose | 0.722 → 0.292 | −0.430 | 0.846 → 0.670 | 93.6% → 66.0% | −27.6 pp | 97.6% → 76.2% | −21.4 pp |
| Sapiens 1B COCO | 0.535 → 0.088 | −0.447 | 0.714 → 0.463 | 80.3% → 52.2% | −28.1 pp | 87.4% → 64.0% | −23.3 pp |
| OpenPose COCO18 | 0.589 → 0.196 | −0.393 | 0.724 → 0.599 | 86.7% → 58.7% | −28.0 pp | 89.7% → 68.1% | −21.6 pp |

COCO OKS-AP and both PCK thresholds decline for all six models. The largest strict-PCK decline is MoveNet (−38.1 pp); the other five lose about 9–28 pp. MediaPipe's mean OKS rises from 0.551 to 0.610 even though its AP and strict PCK decline. Mean OKS is an unranked per-image localization average, while AP includes confidence-ranked detections across OKS thresholds, so they need not change together.

The comparison is descriptive, not a paired or statistically controlled test. COCO has 10,567 labeled keypoints (11.5 per image on average); SpineTrack has 16,618 (16.6 per image). Both use the same COCO-17 slots and formulas, but scene content, sample sizes, and labeled-joint coverage differ. No confidence interval or causal effect is claimed. The observed decreases are consistent with a synthetic-image domain gap and/or source annotation-convention differences, but this experiment does not separate those causes.

On this cohort, YOLO11x-Pose and RTMPose-M lead by COCO AP (0.292 and 0.282); RTMPose has the highest mean OKS (0.677) and PCK@0.10 (0.665), narrowly ahead of YOLO (0.670 and 0.660). These measures do not have to produce identical rankings: mean OKS averages localization per image without confidence ranking, while COCO AP also evaluates ranked detections over ten OKS thresholds. MoveNet's PCK rises from 0.238 at 0.10 to 0.733 at 0.20; its low strict-threshold accuracy alongside much higher lenient-threshold PCK indicates substantial localization error even though a majority of labeled joints fall within 20% of the person's long-side box scale. The large AP50-to-AP75 drops for several models likewise show difficulty at stricter OKS thresholds. MediaPipe's 76 missed poses warrant per-joint and overlay error analysis; none of these results is a reason to alter ground-truth labels.

Across the six models, mean per-joint PCK@0.10 is lowest at the ankles (left 0.310, right 0.320), knees and left elbow (about 0.33), and wrists (0.36–0.37). Shoulders, hips, and face points range roughly 0.67–0.73. This points to a distal-limb localization gap in this sample; the aggregate alone cannot tell whether it comes from model behavior, body articulation/occlusion, or source annotation noise.

This is a frozen-model, zero-shot result on one Unreal Engine / biomechanical synthesis pipeline and a sequence-stratified sample. It is not a general ranking over synthetic image techniques or eras, nor a direct substitute for COCO val2017. Comparison with NameFrame is exploratory because NameFrame lacks five face landmarks and depicts a distant crowd. The COCO evaluation definition was not changed. No original Colab baseline was available for this exact SpineTrack sample, so numerical Colab-to-local equivalence is not claimed. The small number of source annotation errors noted by the maintainers remains a limitation.

The official COCO metric definition and implementation are documented by the [COCO keypoint evaluation page](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/keypoints-eval.htm) and [cocodataset/cocoapi `COCOeval`](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py). PCK normalization conventions differ; the MPII benchmark popularized head-normalized PCKh, which is not interchangeable with this explicitly bbox-normalized PCK ([MPII benchmark paper](https://openaccess.thecvf.com/content_cvpr_2014/html/Andriluka_2D_Human_Pose_2014_CVPR_paper.html)).

Supplemental aggregate metrics, per-joint breakdown, and the COCO-to-SpineTrack comparison are saved in `coco_vs_spinetrack_localization_metrics_2026-10-27.csv`, `coco_vs_spinetrack_per_joint_pck_2026-10-27.csv`, and `coco_vs_spinetrack_model_comparison_2026-10-27.csv`. Recompute them with `python3 scripts/analyse_spinetrack_pose_metrics.py` after the complete COCO and SpineTrack prediction runs are available. SpineTrack-only exports are retained as `spinetrack_unreal_pilot_2026-10-27_localization_metrics.csv` and `spinetrack_unreal_pilot_2026-10-27_per_joint_pck.csv`.
