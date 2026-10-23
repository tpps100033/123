# BabyPose supplementary comparison — 2026-10-23

The project now includes an independently sourced labelled infant dataset: 160 8-bit depth frames, selected as 10 frames from the last 250 annotated frames of each of 16 patients. The [BabyPose v2 annotation CSV](https://doi.org/10.5281/zenodo.5336836) provides the labels and the [v1 archive](https://doi.org/10.5281/zenodo.3891404) provides the matching PNGs. Both records identify CC BY 4.0. The [dataset paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC7551984/) describes the original infant cohort and split. See [docs/BABYPOSE.md](../docs/BABYPOSE.md) for the exact selection, mapping, and source discrepancy.

| Model | 160-image run | COCOeval AP | AP50 | Detections | Device |
| --- | --- | ---: | ---: | ---: | --- |
| MediaPipe Heavy | complete | 0.0000 | 0.0000 | 50 / 160 | CPU |
| MoveNet Thunder | complete | 0.0000 | 0.0000 | 160 / 160 | CPU |
| RTMPose-M | complete | 0.0000 | 0.0000 | 16 / 160 | CUDA |
| YOLO11x-Pose | complete | 0.0000 | 0.0000 | 160 / 160 | CUDA |
| Sapiens 1B | complete | 0.0000 | 0.0000 | 160 / 160 | CUDA |
| OpenPose COCO-18 | complete | 0.0000 | 0.0000 | 25 / 160 | CUDA |

All six runs processed 160 inputs, recorded zero model errors, completed official `pycocotools.COCOeval(iouType='keypoints')`, and had no thermal stop. Sapiens recorded the highest GPU temperature, 80°C, under the 85°C limit. The machine-readable table and each model's prediction, per-image status, environment, weight provenance, and temperature evidence are in the ignored local `04_merged_results/babypose_supplementary/` and `03_model_raw_outputs/<model>/babypose_test160_limb12/` directories.

These AP values evaluate **only the 12 annotated limb joints** mapped into COCO's 17 slots; the five unlabelled face positions have `v=0`. The COCO OKS algorithm and its default thresholds are unchanged, but the area comes from a tight rectangle around visible limb points because BabyPose supplies no person bounding boxes. Depth images differ strongly from the RGB COCO and SyRIP domains. Consequently, BabyPose AP values are a separate stress test and cannot be compared numerically with those full 17-point AP results. The zero AP values do not rank these six models by localization accuracy on this dataset; their detection coverage differs, as the table shows.

The zero scores were investigated. The v2 CSV stores coordinates as `y,x`; 249 overlapping frames in the v1 workbook verified the reversal and the hand→wrist and foot→ankle mapping. A YOLO prediction on the first image placed the left shoulder at approximately (245, 369) versus ground truth (290, 273), with similarly large hip error. In a 16-image, one-per-patient sensitivity check, rotating the images 180° and mapping predictions back left YOLO AP50 at 0. Feeding the 160 ground-truth poses to the evaluator as predictions gave AP=1.0 and AR=1.0. This rules out a broken evaluation connection; the poor outcomes are consistent with the large input-domain gap and do not justify replacing the original AP definition.

To reproduce in WSL2, run `bash scripts/run_babypose_wsl.sh`. It prepares the dataset by HTTP ranges (about 36 MiB transferred in this run), then validates and evaluates each model in its existing isolated environment. Run `python3 scripts/compare_babypose.py` to regenerate the supplementary table without changing the frozen COCO/SyRIP merged analysis.
