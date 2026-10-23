# BabyPose supplementary infant benchmark

> Archived exploratory dataset. BabyPose is excluded from the supported three-domain benchmark and standard run commands. All six generic RGB pose models scored AP=0 on its depth-image subset, so those results are not used for model comparison. The material below documents the historical investigation only.

## Source and rights

BabyPose is a real preterm-infant depth-image dataset by Migliorelli and colleagues. We use the 16,000-row annotation CSV from [Zenodo v2](https://doi.org/10.5281/zenodo.5336836) and the corresponding 8-bit depth PNGs from the [original Zenodo v1 ZIP](https://doi.org/10.5281/zenodo.3891404). Both records identify CC BY 4.0. The dataset paper is [The babyPose dataset](https://pmc.ncbi.nlm.nih.gov/articles/PMC7551984/). Cite the paper and both records in work using these data. The project's images and source annotations stay under ignored `02_datasets/` paths and are not pushed to Git.

Both published ZIP files are approximately 3 GB. The v2 ZIP contains a compressed nested image ZIP, so `scripts/prepare_babypose.py` fetches just its annotation CSV and fetches matching images by HTTP byte ranges from v1. It verifies the ZIP CRC for every selected image and records the source annotation CSV SHA-256, old workbook SHA-256, and resulting handoff SHA-256. The published v1 archive MD5 is recorded as source metadata; the full ZIP is **not** downloaded or independently MD5-checked.

## Selection and point mapping

The v2 CSV contains 1,000 manually annotated images from each of 16 patients. The paper specifies the last 250 annotated frames per patient as the test partition. The supplementary protocol sorts each patient's frames numerically, then selects indices 762, 787, ..., 987 (zero-based within each 1,000-frame group): 10 per patient, 160 total. This reconstructs the paper's split from v2 metadata; the downloadable v1 `babyPose-test.xlsx` unexpectedly lists all 4,000 rows under `pz3`, so it is used only to verify the coordinate mapping on overlapping frames. The exact selected paths and all 16 patient counts are stored in the manifest.

Only the 8-bit depth PNGs are used. The v2 CSV coordinate strings are stored as `y,x`; the script reverses them to image-plane `x,y`. It checks all overlapping v1 spreadsheet entries to verify this order and that v2 `hand`/`foot` points correspond to v1 `wrist`/`ankle` points. Coordinates are otherwise left unchanged. The depth PNG is decoded as a grayscale image by each model's existing image loader. Negative coordinates and coordinates outside the 640×480 image are marked unlabelled (`v=0`) rather than clipped.

| COCO index (0-based) | BabyPose source |
| --- | --- |
| 0–4 (nose, eyes, ears) | unlabelled (`v=0`) |
| 5–6 (shoulders) | left_shoulder, right_shoulder |
| 7–8 (elbows) | left_elbow, right_elbow |
| 9–10 (wrists) | left_hand, right_hand (verified against v1 LW/RW) |
| 11–12 (hips) | left_hip, right_hip |
| 13–14 (knees) | left_knee, right_knee |
| 15–16 (ankles) | left_foot, right_foot (verified against v1 LA/RA) |

The COCO-format person bounding box and area are the tight rectangle enclosing the visible labelled limb joints. This is a declared derived scale because BabyPose has no person bounding boxes. The 17 output slots allow the same `pycocotools.COCOeval(iouType='keypoints')` implementation to run, and the absent face points do not enter OKS for each ground-truth pose. These are **12-limb-keypoint custom-subset AP/AR** scores, not full COCO 17-keypoint AP and not directly comparable in magnitude with the COCO or SyRIP columns. Depth images also create a different input domain from the RGB datasets. Compare model order or changes within this BabyPose subset, with those limits stated.

## Archived experiment

BabyPose is no longer part of the supported preparation, model validation, or research commands. The historical source notebook, annotations, model predictions, and exploratory comparison remain archived for provenance. `scripts/run_babypose_wsl.sh` is disabled. Completed image work covers COCO and SyRIP, with SVIRO as a supplementary synthetic vehicle-interior stress test. A real-child image benchmark is still outstanding; see [the dataset availability analysis](REAL_CHILD_DATASET_AVAILABILITY.md).
