# Result comparison — official COCO keypoint metrics

The former project-defined per-image mean OKS and PCK@0.10 values are retained only as historical artifacts. New accuracy results use the official `cocodataset/cocoapi.COCOeval(iouType='keypoints')` implementation with default OKS thresholds, COCO sigmas, recall thresholds, area ranges, and `maxDets=20`.

COCO scores below cover the exact selected 921-image single-person subset; they are not full val2017 leaderboard scores. SyRIP is evaluated with the same COCO OKS implementation and is a custom-domain result. Pose coverage is diagnostic only. Each prediction's instance score is the mean of its 17 keypoint confidences.

| Model | Runtime | Dataset | AP | AP50 | AP75 | AR | Completeness |
|---|---|---|---:|---:|---:|---:|---|
| MediaPipe Pose Heavy | CPU | COCO 921 | 0.41765 | 0.60562 | 0.46800 | 0.46645 | Complete |
| MediaPipe Pose Heavy | CPU | SyRIP 100 | 0.73032 | 0.92443 | 0.90044 | 0.76400 | Complete |
| MoveNet Thunder | CPU/XNNPACK | COCO 921 | 0.103 | 0.326 | 0.048 | 0.198 | Complete |
| MoveNet Thunder | CPU/XNNPACK | SyRIP 100 | 0.417 | 0.900 | 0.349 | 0.494 | Complete |
| RTMPose-M + YOLOX-M | CUDA | COCO 921 | 0.77481 | 0.89555 | 0.83961 | 0.80510 | Complete |
| RTMPose-M + YOLOX-M | CUDA | SyRIP 100 | 0.83417 | 0.96999 | 0.93737 | 0.84800 | Complete |
| YOLO11x-Pose | CUDA | SyRIP 100 | 0.75071 | 0.98653 | 0.86314 | 0.78300 | Complete |
| YOLO11x-Pose | CUDA + ASUS Full Speed fans | COCO 921 | 0.72192 | 0.91870 | 0.79433 | 0.77883 | Complete, 921/921 |

MediaPipe and MoveNet GPU inference could not be enabled in this WSL setup. MoveNet's TensorFlow GPU libraries and TFLite GPU delegate are unavailable; MediaPipe's GPU task rejects WSL's software renderer. Their CPU predictions remain useful for metric comparison but are not GPU timing baselines. RTMPose and YOLO passed actual CUDA inference smoke checks. With ASUS MyASUS Full Speed selected on AC power, YOLO completed all 921 COCO images using an 85°C thermal stop limit and a temperature check on every image. The run manifest recorded a 75°C GPU temperature; independent monitoring sampled at most 72°C. No thermal stop occurred. The complete result was merged into `04_merged_results/merged_20261022T170631Z/benchmark_table.csv`; paired analysis is in `05_analysis_results/analysis_20261022T170631Z/`.

No paired official COCOeval score from the original Colab run was found, so a numerical Colab-to-local delta cannot be calculated. Historical custom OKS/PCK outputs remain preserved and are not reinterpreted as COCOeval results. The machine-generated paired comparison is in `05_analysis_results/analysis_20261022T170631Z/comparison_report.md`.
