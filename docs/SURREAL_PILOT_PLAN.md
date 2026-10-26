# SURREAL pilot acquisition and sampling record

## Request

Prepare a local pilot of **500 images** (within the requested 300–1,000 range), sampled across SURREAL test clips. Keep source files and generated images local in ignored dataset directories; do not commit or redistribute SURREAL data.

## Current acquisition status — 2026-10-27

**Superseded:** the SURREAL archive endpoint returned HTTP 403. No SURREAL samples were downloaded or scored. The active replacement is the 300-image NameFrame Quarry Crowd pilot in [the run report](../reports/nameframe_quarry_crowd_pilot_2026-10-27.md); this document remains as the original access and mapping plan only.

- Searched the active project source archive/canonical cache, the Downloads research folder, migration input, and the previously shared Google Drive project folder. No SURREAL source images, videos, or annotations were found.
- The user confirmed personal acceptance of the official data license and that download credentials are available. No SURREAL data have yet been downloaded or evaluated; the secure local terminal is waiting for credential entry.
- The license limits use to non-commercial scientific research and prohibits making the dataset available to third parties without prior written permission. Downloading or using the data itself signifies agreement. The user accepted personally; the agent will not do so on the user's behalf.
- Do not save credentials in Git, project configuration, run logs, or chat. Do not upload source data, extracted samples, or model outputs containing source images to GitHub/Drive.

## Sampling and annotation protocol to use after access

1. Use the official CMU **test** split and one overlap setting only (`run1`, documented as 30% clip overlap) to avoid mixing overlapping render variants from `run0`/`run2`.
2. A deterministic 500-frame plan is now generated: seed `20261027`, 100 distinct sequence folders, one clip per folder, five frames per clip at zero-based indices 10, 30, 50, 70, and 90. The official CMU file indexes report 12,528 test clips overall, including 5,700 in run1 across 703 sequence folders. Record source sequence, clip, frame index, SHA-256, and selection reason in a CSV/JSON manifest. If selected source clips prove invalid, replace them deterministically from other folders and document every replacement; never duplicate frames to fill the target.
3. Acquire only RGB video, `_info.mat` keypoint annotations, and the corresponding `_segm.mat` files needed for image extraction, person bounds, and visibility audit. Exclude depth and optical-flow files. Enforce the user's 30 GB/day transfer ceiling and stop before exceeding it.
4. SURREAL provides 24 projected SMPL joints (`joints2D`, pixel coordinates) rather than COCO-17. The mapping follows the SMPL-24 ordering documented in the official SURREAL metadata and the OpenMMLab MMHuman3D SMPL-24 convention. Map only the 12 semantically shared body joints: shoulders, elbows, wrists, hips, knees, and ankles. Do **not** map the SMPL head joint to COCO nose or invent eyes/ears. Keep the remaining five COCO face slots unlabeled. The result must be described as a 12-of-17 shared-joint subset, not full COCO-17 ground truth; report the exact mapping and coverage.
5. Inspect the official 24-joint ordering and coordinates, segmentation-based visibility policy, RGB/annotation frame alignment, and bbox construction on a small visual audit before freezing the 500-frame list. Preserve the original annotations and generated mapping metadata.
6. After the data audit, convert to a local immutable COCO-shaped handoff and run all six frozen model adapters with the existing 85°C thermal cutoff. The source has no explicit COCO visibility labels; the current preparation script derives visibility from alignment with the source person-part segmentation, and this policy must pass visual audit before it is frozen. Evaluate with the official COCO keypoint evaluator, retaining the default COCO sigmas at the mapped slots and explicitly identifying this as a partial-joint COCO-OKS evaluation. Its AP is not directly comparable with a full 17-joint COCO score.

## Official source references

- [SURREAL data page](https://www.di.ens.fr/willow/research/surreal/data/)
- [SURREAL download instructions and annotation schema](https://github.com/gulvarol/surreal#1-download-surreal-dataset)
- [SURREAL non-commercial research license](https://github.com/gulvarol/surreal/blob/master/LICENSE.md)
- [OpenMMLab MMHuman3D SMPL-24 joint ordering](https://github.com/open-mmlab/mmhuman3d/blob/main/mmhuman3d/core/conventions/keypoints_mapping/smpl.py)
- Varol et al., *Learning from Synthetic Humans*, CVPR 2017, [paper](https://arxiv.org/abs/1701.01370).
