# NameFrame Quarry Crowd synthetic pose pilot — 2026-10-27

## Outcome

SURREAL was not acquired: its available download endpoint returned HTTP 403. It was replaced with the public NameFrame Quarry Crowd sample, a 300-frame Unreal Engine 5.8 capture (1280×960). The 597,040,148-byte ZIP passed the publisher's SHA-256 check:

`802f5389b8dc2420ed1ed30bca9e831eae14d96d3f06f4795b22c0469983124c`

The source bundle, extracted images and labels, and generated annotations are in ignored local `02_datasets/source_archives/nameframe_quarry_crowd/` and `02_datasets/canonical/nameframe_quarry_crowd_singleperson_pilot300/` folders. They are not committed or included in paper bundles. The publisher permits local benchmarking and publication of results, but prohibits repackaging the pack as a dataset; redistribution of underlying Unreal assets is separately restricted. Review the included `TERMS.txt` before sharing images or derivatives.

## Annotation and sample construction

The pack contains 3,692 YOLO-pose person records over 300 frames. The 17 tuple positions follow COCO order. The first five face slots (nose, eyes, ears) are zero/unlabeled in every record; the 12 body slots are exact source coordinates, converted from normalized XYV to pixel XYV. Original visibility states are retained: 0 unlabeled, 1 labeled but occluded, 2 visible. No landmarks were inferred or filled. SHA-256 comparison found no exact duplicate RGB files; the publisher's perceptual audit identified 74 near-duplicate frames.

Twenty source people have zero visible/labeled keypoints. All 3,692 records are retained in the local `source_annotations_all.json` audit file. To remain comparable to the project's single-pose adapters, the scored COCO annotation selects the largest person with at least one labeled keypoint per image. This creates 300 deterministic targets. This is not a multi-person recall benchmark.

The conversion script verifies archive size/hash, exactly 300 PNG/label pairs, 1280×960 image dimensions, finite normalized boxes and keypoints, the 17-slot schema, absent face slots, visibility values, and nonempty scored targets. The frozen handoff index hash is `0ea05ed815157b77f0805ea8fed7070fe43021eb60e5a088b11400e6134fb0f0`.

## Six-model run

All six models processed 300/300 images with zero model errors, zero thermal stops, and successful official `cocodataset/cocoapi.COCOeval(iouType='keypoints')` evaluation. This uses the project's frozen fair-reference settings (batch 1, FP32, no AMP/compile/quantization/TensorRT). AP values are subset results for this capture; they are not full COCO scores.

| Model | OK / no detection | AP | AP50 | AP75 | AR | Runtime | Median latency (ms/image) |
|---|---:|---:|---:|---:|---|---|---:|
| MediaPipe Pose Heavy | 0 / 300 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | CPU | 118.94 |
| MoveNet Thunder | 297 / 3 | 0.0003 | 0.0011 | 0.0002 | 0.0043 | CPU | 127.52 |
| RTMPose-M | 207 / 93 | 0.1192 | 0.1519 | 0.1293 | 0.2050 | RTX 5060 CUDA | 138.03 |
| YOLO11x-Pose | 296 / 4 | 0.0887 | 0.1167 | 0.0942 | 0.1270 | RTX 5060 CUDA | 159.96 |
| OpenPose COCO18 | 91 / 209 | 0.0353 | 0.0499 | 0.0417 | 0.0387 | RTX 5060 CUDA | 227.43 |
| Sapiens 1B COCO | 300 / 0 | 0.0050 | 0.0050 | 0.0050 | 0.0033 | RTX 5060 CUDA | 111.19 |

The full AP values, model provenance, GPU snapshots, per-image predictions, and run IDs are preserved in each model's ignored `03_model_raw_outputs/<model>/nameframe_quarry_crowd_singleperson_pilot300/fair_reference/<run-id>/` folder. Model run IDs end in `a73a1d` (MediaPipe), `c246ad` (MoveNet), `b1b303` (RTMPose), `fccf21` (YOLO), `651aed` (OpenPose), and `67c458` (Sapiens).

MediaPipe and MoveNet fell back to CPU: this WSL runtime does not expose MediaPipe's physical-GPU service, and its TensorFlow Lite GPU delegate library is absent. RTMPose used ONNX Runtime CUDA, YOLO used CUDA, OpenPose used CUDA, and Sapiens used `cuda:0`. The Sapiens model briefly sustained full GPU utilization; observed peak temperature was about 75°C, below the 85°C stop limit. VRAM peaked near 5.9 GiB during OpenPose and 5.5 GiB during Sapiens.

## Interpretation and source-quality caveats

The scores are low and should be read as a domain-stress pilot, not a model ranking. This capture places many people at distance in a narrow overhead quarry camera zone; its packaged validator notes that 98/300 frames (32.7%) have a box under 12 px, and a single-person adapter cannot return all people in each crowded frame. The evaluator uses one largest eligible person as ground truth, while any extra model detections can count as false positives. Five face joints are absent. These conditions differ from the prior COCO/SyRIP/child tasks and make direct cross-dataset score comparisons inappropriate.

The publisher web page advertises a grade of A/95, while the bundled `reports/report.json` records B/89. We preserve that mismatch rather than choosing the more favorable figure. The bundle reports 74 near-duplicate frames, 2 of 3,492 engine-visible people unlabeled (0.06%), and a narrow single-map/camera-zone capture. Its own report also marks build provenance unverified. The source report's checksum and observations are preserved locally with the extracted bundle.

## Reproduction

From WSL Ubuntu at the project root:

```bash
python scripts/prepare_nameframe_quarry_crowd.py
INFANTPOSE_MAX_GPU_TEMP_C=85 bash scripts/run_all_models_for_dataset.sh nameframe_quarry_crowd_singleperson_pilot300
```

The source ZIP is already cached locally. The converter intentionally refuses to overwrite an existing canonical handoff; preserve it and inspect the manifest before making any new variant. The GPU task log is under `06_logs/research_nameframe_quarry_crowd_singleperson_pilot300_*.log`.

## Source

- [NameFrame Quarry Crowd source page and download details](https://getnameframe.com/datasets/quarry-crowd)
- Exact release: version 1.0, seed `20260930`, map `Quarry_Slate`, Unreal Engine `5.8`.
- Citation supplied by the publisher: NameFrame, “Quarry crowd: synthetic computer-vision dataset,” version 1.0, 2026.
