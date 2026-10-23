# SVIRO child comparison protocol

## Dataset

This comparison uses the official SVIRO BMW i3 held-out test split. SVIRO is a synthetic vehicle-interior dataset with labeled people, children, and infants. The source license is CC BY-NC-SA 4.0; cite the SVIRO dataset paper when publishing results.

This 53-scene BMW i3 handoff is the only completed SVIRO image evaluation. A broader annotation-only inventory identified 710 candidate scenes, but their RGB images and model results are not complete; see [SVIRO_CHILDREN.md](SVIRO_CHILDREN.md). SVIRO is synthetic vehicle-interior imagery and does not serve as a real-child benchmark.

The deterministic subset includes every test scene whose filename labels exactly one class-2 child and no class-1 infant or class-3 adult. The current release contains 53 qualifying scenes. Each selected scene has one pose annotation, one RGB frame, and one official person box. The child remains seated and partially occluded by the car seat, which is part of this domain's test conditions.

## COCO-17 mapping

| COCO joint | SVIRO joint |
| --- | --- |
| left/right shoulder | `upperarm_l` / `upperarm_r` |
| left/right elbow | `lowerarm_l` / `lowerarm_r` |
| left/right wrist | `hand_l` / `hand_r` |
| left/right hip | `thigh_l` / `thigh_r` |
| left/right knee | `calf_l` / `calf_r` |
| left/right ankle | `foot_l` / `foot_r` |

The five COCO face joints have visibility zero because SVIRO does not provide those landmarks. Original visibility values are kept for mapped joints. The official class-3 person bounding box defines annotation area. No coordinates are clipped or inferred from model predictions.

## Prepare and run

On Windows, run `py -3.9 scripts/prepare_sviro_child.py`. The importer validates the SHA-256 hashes of the small pose and box archives, reads RGB ZIP central-directory metadata by HTTPS ranges, and transfers only the selected images. It writes a hashed, relative-path COCO handoff under `02_datasets/canonical/sviro_child_i3_test53_coco12/`.

From WSL2, run an individual model with `python scripts/validate_model.py --model <model> --dataset sviro_child_i3_test53_coco12`, then `python scripts/run_research.py --model <model> --dataset sviro_child_i3_test53_coco12`. The full sequence runs with `bash scripts/run_all_research.sh` after preparation. GPU thermal controls and isolated model environments are the same as for COCO and SyRIP.

COCO AP, AP50, AP75, and AR use the unmodified official `pycocotools.COCOeval(iouType='keypoints')` defaults. Child-domain scores are not COCO leaderboard values and should be interpreted separately from COCO and SyRIP because the capture domain, visibility, and mapped joint coverage differ.
