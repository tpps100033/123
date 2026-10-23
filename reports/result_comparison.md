# Three-domain COCO keypoint evaluation

Input table: `04_merged_results/merged_20261023T140227Z/benchmark_table.csv`

Complete three-domain model pairs: 6

All AP/AR fields use official cocodataset/cocoapi COCOeval keypoint defaults. COCO values use the selected 921-image single-person subset; SyRIP uses 100 infant images; the child domain uses 134 COCO-UniHuman apparent-age crops. Child source image IDs do not overlap the 921-image COCO subset, but both cohorts use COCO val2017 source imagery. SyRIP and child AP are custom-domain COCO-OKS evaluations, not leaderboard results.

Pose coverage is diagnostic only. Latency is reported only for complete GPU runs.

| Model | COCO AP | SyRIP infant AP | Child AP | Child − COCO AP |
|---|---:|---:|---:|---:|
| mediapipe_pose_heavy | 0.418 | 0.730 | 0.578 | 0.161 |
| movenet_thunder | 0.103 | 0.417 | 0.407 | 0.304 |
| openpose_coco18 | 0.589 | 0.477 | 0.627 | 0.039 |
| rtmpose_m | 0.775 | 0.834 | 0.869 | 0.094 |
| sapiens_1b_coco | 0.535 | 0.755 | 0.659 | 0.124 |
| yolo11x_pose | 0.722 | 0.751 | 0.676 | -0.046 |

Domain AP differences are descriptive because the datasets have different populations and source protocols. A paired original Colab run with official COCOeval metrics was not available, so a numerical Colab-to-local delta cannot be calculated. Historical project-defined OKS/PCK outputs remain separate and are not treated as COCOeval scores.
