# SVIRO child pose comparison

Generated: 2026-10-23T09:23:49.077883+00:00

This independent child set contains 53 scenes from the official BMW i3 test split. Each selected scene has exactly one child and no adult or infant. RGB frames, keypoints, visibility, and person boxes come from the official SVIRO release. Twelve limb joints map to COCO-17; the five unannotated face points have visibility zero. The released class-3 person box supplies area. Scores use official COCOeval defaults and are child-domain results, not COCO leaderboard scores.

| Model | Status | AP | AP50 | AP75 | AR | Images / errors | Device | Validation | GPU temp at end | Device memory at end |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |
| mediapipe_pose_heavy | complete | 0.3857 | 0.5436 | 0.3937 | 0.4434 | 53 / 0 | cpu | inference_pass_cpu_only | 53°C | 252 MiB |
| movenet_thunder | complete | 0.2687 | 0.8409 | 0.0172 | 0.3302 | 53 / 0 | cpu | inference_pass_cpu_only | 53°C | 252 MiB |
| rtmpose_m | complete | 0.8393 | 0.8614 | 0.8614 | 0.8491 | 53 / 0 | cuda | inference_pass | 57°C | 1005 MiB |
| yolo11x_pose | complete | 0.9406 | 1.0000 | 0.9783 | 0.9547 | 53 / 0 | cuda | inference_pass | 59°C | 869 MiB |
| sapiens_1b_coco | complete | 0.7872 | 0.9196 | 0.8595 | 0.8057 | 53 / 0 | cuda:0 | inference_pass | 75°C | 252 MiB |
| openpose_coco18 | complete | 0.8824 | 1.0000 | 0.9791 | 0.9038 | 53 / 0 | cuda:0 | inference_pass | 66°C | 355 MiB |

The GPU temperature and device-memory columns show the last sample stored in each run manifest, not a maximum. Across the complete run, the highest manually observed GPU temperature was 79°C; the 85°C stop was not reached. MediaPipe and MoveNet have no usable GPU provider in this WSL setup and ran on CPU. Validation requires real single-image and two-image inference; a no-detection result is accepted only when the adapter returns the expected empty keypoint list.

Full per-image predictions and run evidence remain in ignored `03_model_raw_outputs/`; the run IDs are in `comparison.csv`.
