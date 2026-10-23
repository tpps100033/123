# InfantPose local result comparison

Final merged results: `04_merged_results/merged_20261023T054218Z/benchmark_table.csv`

Machine-generated paired analysis: `05_analysis_results/analysis_20261023T054220Z/`

All six models completed 921-image COCO single-person-subset and 100-image SyRIP runs. All accuracy scores use the official `cocodataset/cocoapi.COCOeval(iouType='keypoints')` evaluator with its default keypoint OKS sigmas and thresholds. COCO values describe the selected 921-image subset, not the full COCO val2017 leaderboard. SyRIP is a custom-domain dataset evaluated with the same COCO OKS implementation; its score is not a COCO leaderboard result.

| Model | COCO subset AP | SyRIP COCO-OKS AP | Δ AP (COCO − SyRIP) | COCO AP50 | SyRIP AP50 | COCO AP75 | SyRIP AP75 |
|---|---:|---:|---:|---:|---:|---:|---:|
| MediaPipe Pose Heavy | 0.418 | 0.730 | -0.313 | 0.606 | 0.924 | 0.468 | 0.900 |
| MoveNet Thunder | 0.103 | 0.417 | -0.314 | 0.326 | 0.898 | 0.048 | 0.349 |
| OpenPose COCO18 | 0.589 | 0.477 | 0.112 | 0.789 | 0.919 | 0.650 | 0.329 |
| RTMPose-M | 0.775 | 0.834 | -0.059 | 0.896 | 0.970 | 0.840 | 0.937 |
| Sapiens 1B COCO | 0.535 | 0.755 | -0.221 | 0.742 | 0.931 | 0.584 | 0.830 |
| YOLO11x Pose | 0.722 | 0.751 | -0.029 | 0.919 | 0.987 | 0.794 | 0.863 |

## Runtime validation

Each model passed a real single-image and batch-two inference check with 17 mapped output keypoints per image. RTMPose, YOLO11x Pose, OpenPose, and Sapiens used the RTX 5060. MediaPipe and MoveNet completed on CPU because their GPU execution providers are unavailable in this WSL setup. Both datasets completed for every model, giving 12 of 12 complete runs. The Sapiens run recorded 5,727 MiB peak device use and 80°C maximum GPU temperature; the configured thermal stop was 85°C and did not trigger.

The Δ column is descriptive because the COCO and SyRIP datasets have different distributions. No paired original Colab result using official COCOeval metrics was available, so a numerical Colab-to-local difference cannot be computed. Historical project-defined OKS/PCK results remain preserved as historical artifacts and are not compared to COCOeval values.
