# Child-only pose model evaluation — 2026-11-01

## Scope and data

This analysis evaluates six existing frozen pose models on the COCO-UniHuman v2 real-image child cohort only. The selected evaluation set contains 134 unique COCO photographs, one eligible child person crop per image, with 1,827 labeled COCO-17 keypoints. The source's ages 3–17 are manually estimated apparent ages, not verified chronological ages. The original adult-dominant COCO comparison is retained as a historical exploratory analysis and is not part of the active child-only results. Dataset selection, image access, filtering, crop construction, and provenance are documented in [REAL_CHILD_DATASET_AVAILABILITY.md](../docs/REAL_CHILD_DATASET_AVAILABILITY.md).

All six models have complete prior inference runs on this same fixed cohort; this report recomputes the metrics from those predictions and does not claim a new model-inference run. No model was fine-tuned on the child images.

## Metrics

- **COCO AP / AP50 / AP75:** Official `pycocotools.COCOeval(iouType="keypoints")` keypoint evaluation, with COCO's OKS sigmas and thresholds. This is AP on the selected 134-image one-person subset, not full COCO val2017 AP.
- **Mean OKS:** Per-image OKS using the official sigmas, the ground-truth annotation area, and all labeled joints; a missing pose scores zero. This descriptive statistic is not AP.
- **PCK@t:** A visible ground-truth joint is correct if pixel error is at most `t × max(ground-truth bbox width, height)`. Only joints with `v>0` enter the denominator; missing poses count incorrect.
- **Pose coverage:** Images with a valid predicted pose divided by 134; diagnostic only, not an accuracy score.

## Results

| Model | COCO AP | AP50 | AP75 | Mean OKS | PCK@0.10 | PCK@0.20 | Pose coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| RTMPose-M | **0.869** | **0.960** | **0.901** | **0.899** | **94.7%** | 96.5% | 99.3% |
| YOLO11x-Pose | 0.676 | 0.801 | 0.740 | 0.759 | 81.7% | 85.4% | 100.0% |
| Sapiens 1B COCO | 0.659 | 0.788 | 0.701 | 0.769 | 81.5% | 85.6% | 100.0% |
| OpenPose COCO18 | 0.627 | 0.801 | 0.710 | 0.779 | 83.4% | 86.3% | 100.0% |
| MediaPipe Pose Heavy | 0.578 | 0.830 | 0.625 | 0.743 | 84.0% | 89.7% | 94.0% |
| MoveNet Thunder | 0.407 | 0.897 | 0.253 | 0.703 | 90.1% | **97.2%** | 100.0% |

RTMPose-M has the highest COCO AP, mean OKS, and PCK@0.10 on this cohort; MoveNet has the highest PCK@0.20. MoveNet's much lower AP75 shows that this threshold-based PCK result does not translate to COCO's stricter OKS ranking. The metrics answer different questions and should be reported together. MediaPipe has the lowest pose coverage (126/134); RTMPose misses one crop (133/134), while the other four return a pose on all 134.

## Interpretation limits

These are descriptive results on 134 crops from one source dataset, with no confidence intervals or significance tests. The apparent ages are not verified, and results can depend on crop construction, source image content, annotation visibility, and the selected-person protocol. Do not generalize these numbers to every child age, infant pose, or unconstrained full-scene child detection. The COCO-UniHuman cohort is derived from COCO photographs, not an independent image source.

## Reproducible outputs

Run `python scripts/analyse_pose_localization_comparison.py` to regenerate the child-only JSON, aggregate metrics CSV, and per-joint PCK CSV from the frozen predictions. The script also accepts `--datasets` for optional cross-dataset analyses.

- `child_coco_unihuman_val134_model_metrics_2026-11-01.json`
- `child_coco_unihuman_val134_model_metrics_2026-11-01_metrics.csv`
- `child_coco_unihuman_val134_model_metrics_2026-11-01_per_joint_pck.csv`
