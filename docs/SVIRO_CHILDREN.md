# SVIRO supplementary synthetic-child comparison

This is a synthetic vehicle-interior stress test, not the project's real-child image benchmark. Only the 53-image BMW i3 subset has a complete image handoff and six-model score report. The 710-scene all-vehicle figure is an annotation-only candidate inventory; its additional RGB handoff and model runs are not complete. Use `docs/REAL_CHILD_DATASET_AVAILABILITY.md` for the search and verified COCO-17 mapping analysis. Do not generalize this result to real children or ordinary environments.

## Source and selection

This annotation-only inventory scans the official SVIRO held-out test split for all ten vehicles: BMW i3, Hyundai Tucson, BMW X5, Renault Zoe, Lexus GS F, Toyota Hilux, Tesla Model 3, VW Tiguan, Mercedes A-Class, and Ford Escape. The official test split has 500 scenes per vehicle. The deterministic filter selects scenes with exactly one class-2 child and no class-1 infant or class-3 adult. The current official-label inventory contains 710 qualifying scenes. Only the 53-image BMW i3 subset has been prepared and evaluated.

| Vehicle | Child-only scenes |
| --- | ---: |
| BMW i3 | 53 |
| Hyundai Tucson | 83 |
| BMW X5 | 57 |
| Renault Zoe | 69 |
| Lexus GS F | 82 |
| Toyota Hilux | 77 |
| Tesla Model 3 | 65 |
| VW Tiguan | 74 |
| Mercedes A-Class | 74 |
| Ford Escape | 76 |
| **Total** | **710** |

Counts and pose archive SHA-256 values are recorded in `reports/sviro_children_inventory.json`. The importer checks the official `pose_wholeImage` and `boundingBoxes_wholeImage` ZIP contents, derives each image selection from the official class labels, and verifies every RGB image against the source ZIP member size and CRC. It reads only selected entries from the official RGB whole-image ZIPs using HTTP byte ranges; it does not download the full RGB archives.

The official SVIRO source is [sviro.kl.dfki.de](https://sviro.kl.dfki.de/), and the dataset is licensed CC BY-NC-SA 4.0. Cite M. Da Cruz et al., “SVIRO: Synthetic Vehicle Interior Rear Seat Occupancy Dataset and Benchmark,” WACV 2020, https://arxiv.org/abs/2001.03483.

## COCO-17 mapping and scoring

| COCO joint | SVIRO joint |
| --- | --- |
| left/right shoulder | `upperarm_l` / `upperarm_r` |
| left/right elbow | `lowerarm_l` / `lowerarm_r` |
| left/right wrist | `hand_l` / `hand_r` |
| left/right hip | `thigh_l` / `thigh_r` |
| left/right knee | `calf_l` / `calf_r` |
| left/right ankle | `foot_l` / `foot_r` |

The five COCO face joints are unlabelled because SVIRO does not annotate those landmarks. Original SVIRO visibility values for mapped joints are retained. The official class-3 person box supplies the COCO area; coordinates are not clipped or inferred from model predictions. Each image contains one evaluated child. AP, AP50, AP75, and AR use the unmodified `pycocotools.COCOeval(iouType='keypoints')` defaults.

The 710 candidates are not treated as 710 independent children. SVIRO reuses synthetic human models across vehicle domains. No pooled or per-vehicle model scores are available for these candidates because their RGB images and model runs were not completed.

## Prepare and run

From Windows PowerShell, run:

```powershell
py -3.9 scripts/scan_sviro_child_candidates.py
py -3.9 scripts/prepare_sviro_child.py
py -3.9 scripts/verify_sviro_coco_metric.py --dataset sviro_child_i3_test53_coco12
```

The preparation script caches the official annotation archives under the ignored `02_datasets/source_archives/` folder and creates `02_datasets/canonical/sviro_child_i3_test53_coco12/`. The separate all-vehicle inventory/preparer remains optional and does not feed the completed six-model package.

From WSL2, run the six isolated model kernels for this child dataset with:

```bash
bash scripts/run_sviro_child_wsl.sh
```

The 53-image BMW i3 handoff and its scores are preserved under `sviro_child_i3_test53_coco12` as a supplementary synthetic stress test. A separate real-child benchmark is still required for child-domain generalization claims.
