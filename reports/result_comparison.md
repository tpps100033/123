# Result comparison — official metric migration

The former per-image mean OKS and PCK@0.10 were project-defined metrics, not COCO keypoint AP. They are no longer used for benchmark accuracy or ranking.

New accuracy values come from the official `cocodataset/cocoapi` `COCOeval(iouType='keypoints')` implementation. Defaults are retained for OKS thresholds, COCO keypoint sigmas, recall thresholds, area ranges, and `maxDets=20`. Results require a dataset-level set of predictions, so `metrics.jsonl` now records per-image inference status only; `run_manifest.json` stores AP/AR metrics.

The COCO run evaluates the exact selected 921-image single-person subset with its supplied COCO-format annotations. Its AP is not the full COCO val2017 leaderboard score. SyRIP annotations use the same COCO-17 skeleton and are passed through the same evaluator, but those values are labeled as a custom-domain COCO-OKS evaluation, not official COCO dataset results. Pose coverage is reported only as a diagnostic. Each pose detection's COCO result `score` is defined as the mean of its 17 model keypoint confidences.

No local model predictions were produced before the evaluator change, so there are no legacy numerical results to compare or reinterpret. MediaPipe Heavy was then run on both handoffs with its Windows CPU fallback. There is no matching Colab run manifest/score pair in the imported artifacts for a numerical Colab-versus-local comparison.

| Model | Dataset | COCOeval AP | AP50 | AP75 | AR | Local inference | Comparison |
|---|---|---:|---:|---:|---:|---|---|
| MediaPipe Pose Heavy (CPU) | COCO 921-image single-person subset | 0.41765 | 0.60562 | 0.46800 | 0.466 | Complete; GPU unavailable in Windows MediaPipe build | No paired Colab metric run available |
| MediaPipe Pose Heavy (CPU) | SyRIP Test100 (custom domain) | 0.73032 | 0.92443 | 0.90044 | 0.764 | Complete; GPU unavailable in Windows MediaPipe build | No paired Colab metric run available |

The historical custom OKS/PCK outputs remain preserved as legacy artifacts, but they are not mixed into the COCOeval result table or accuracy rankings.

RTMPose-M passed single-image and two-image CPU smoke inference with 17 points each. A full dataset run was not started after the overheating caution; therefore it has no AP/AR score row.
