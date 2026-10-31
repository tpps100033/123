# Adult-dominant COCO vs COCO-UniHuman child pose comparison — 2026-11-01

## Scope

The active comparison is the existing adult-dominant COCO val2017 single-person reference (921 images) versus the real-image COCO-UniHuman child cohort (134 person crops). The child set uses apparent-age labels 3–17; these are manual apparent-age estimates, not verified ages. Standard COCO val2017 has no age labels, so it is an adult-dominant general-human reference rather than a verified adult-only set. See [child-data provenance and selection](../docs/REAL_CHILD_DATASET_AVAILABILITY.md).

All six frozen models already have complete runs on both datasets. The comparison reuses those predictions and evaluates them with the same official `pycocotools.COCOeval(iouType="keypoints")` AP/AR and supplemental metrics:

- **Mean OKS:** official COCO sigmas and each ground-truth person's annotation area; average over all selected images; missing detections contribute zero. This is a descriptive mean localization score, not COCO AP.
- **PCK@t:** a labeled joint is correct when its Euclidean pixel error is at most `t × max(ground-truth bbox width, height)`. Thresholds are 0.10 and 0.20 here. Missing detections count as incorrect, and only ground-truth points with `v>0` are in the denominator.
- **Pose coverage:** images with a valid predicted 17-point pose divided by the selected image count.

The COCO AP remains the official OKS-based AP; PCK is a separately defined bbox-long-side variant. The metric calculation is reproducible with `python scripts/analyse_pose_localization_comparison.py` (defaults to adult COCO and the child cohort).

## Results

Values are shown as adult COCO → child cohort. AP and mean OKS are 0–1 scores; PCK is a percentage.

| Model | COCO OKS-AP | Mean OKS | PCK@0.10 | PCK@0.20 | Adult pose coverage → child |
|---|---:|---:|---:|---:|---:|
| MediaPipe Pose Heavy | 0.418 → 0.578 | 0.551 → 0.743 | 70.9% → 84.0% | 75.5% → 89.7% | 71.4% → 94.0% |
| MoveNet Thunder | 0.103 → 0.407 | 0.453 → 0.703 | 61.9% → 90.1% | 84.8% → 97.2% | 99.9% → 100.0% |
| RTMPose-M | 0.775 → 0.869 | 0.841 → 0.899 | 93.9% → 94.7% | 96.0% → 96.5% | 95.1% → 99.3% |
| YOLO11x-Pose | 0.722 → 0.676 | 0.846 → 0.759 | 93.6% → 81.7% | 97.6% → 85.4% | 100.0% → 100.0% |
| Sapiens 1B COCO | 0.535 → 0.659 | 0.714 → 0.769 | 80.3% → 81.5% | 87.4% → 85.6% | 100.0% → 100.0% |
| OpenPose COCO18 | 0.589 → 0.627 | 0.724 → 0.779 | 86.7% → 83.4% | 89.7% → 86.3% | 90.0% → 100.0% |

The child-minus-COCO AP differences are +0.161 for MediaPipe, +0.304 for MoveNet, +0.094 for RTMPose, −0.046 for YOLO, +0.124 for Sapiens, and +0.039 for OpenPose. PCK@0.10 changes are +13.1, +28.3, +0.8, −12.0, +1.2, and −3.3 percentage points respectively. This is a mixed model response, not a uniform child advantage: YOLO is lower on all three localization scores, while RTMPose is nearly unchanged on PCK.

## Interpretation and limitations

Five models have higher COCO AP on the child cohort; YOLO is lower. Several gains coincide with higher pose-detection coverage, especially MediaPipe and OpenPose. The child sample is small (134 images) compared with COCO (921); there are 1,827 labeled child keypoints (13.6 per image) versus 10,567 adult-reference labeled keypoints (11.5 per image). No confidence intervals or significance tests were computed, so these values are descriptive.

The comparison does **not isolate age**. The child images are person crops with 12% context, while the COCO adult-reference images are uncropped scenes. Crop scale, background, visible-person selection, and dataset size may all affect pose detection and localization. The child labels are apparent ages 3–17, and COCO val2017 has no verified adult-age labels. Therefore do not interpret this result as proof that the models are inherently better or worse on children. A controlled age comparison would need comparable person crops and reliable age-group labels for both cohorts.

## Reproducible output files

- `adult_coco_val2017_vs_child_coco_unihuman_val134_pose_comparison_2026-11-01_metrics.csv`: per-dataset/model COCO AP, mean OKS, PCK, coverage, and error summaries.
- `adult_coco_val2017_vs_child_coco_unihuman_val134_pose_comparison_2026-11-01_model_comparison.csv`: child-minus-COCO deltas by model.
- `adult_coco_val2017_vs_child_coco_unihuman_val134_pose_comparison_2026-11-01_per_joint_pck.csv`: per-joint PCK breakdown.
- Matching JSON file contains metric definitions, run IDs, aggregate values, and deltas.
