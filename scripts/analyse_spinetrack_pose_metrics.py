#!/usr/bin/env python3
"""Add bbox-normalized PCK and COCO-OKS summaries to the SpineTrack pilot.

PCK uses visible ground-truth joints and normalizes Euclidean error by the
longer side of the ground-truth person box. Missing/no-detection predictions
remain in the denominator and count as incorrect. The OKS column applies the
official COCO keypoint sigmas and annotation area to each image's one-person
prediction; it is a descriptive mean OKS, not a replacement for COCOeval AP.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "spinetrack_unreal_singleperson_pilot1000"
MODEL_IDS = (
    "mediapipe_pose_heavy",
    "movenet_thunder",
    "rtmpose_m",
    "yolo11x_pose",
    "sapiens_1b_coco",
    "openpose_coco18",
)
JOINT_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle",
)
COCO_SIGMAS = (
    .026, .025, .025, .035, .035, .079, .079, .072, .072,
    .062, .062, .107, .107, .087, .087, .089, .089,
)
PCK_THRESHOLDS = (.05, .10, .20, .50)


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def pick_complete_run(model_id: str, dataset_id: str) -> tuple[Path, dict]:
    parent = ROOT / "03_model_raw_outputs" / model_id / dataset_id / "fair_reference"
    candidates = []
    if parent.exists():
        for manifest_path in parent.glob("*/run_manifest.json"):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("metric_complete") and manifest.get("processed_images") == 1000:
                candidates.append((manifest_path.parent, manifest))
    if not candidates:
        raise RuntimeError(f"No complete 1000-image run found for {model_id}")
    return max(candidates, key=lambda item: item[1]["run_id"])


def oks_for_instance(pred_points, gt_points, area: float) -> float:
    visible = [(i, gt_points[i]) for i in range(17) if gt_points[i][2] > 0]
    if not visible or area <= 0:
        return 0.0
    total = 0.0
    for index, (gx, gy, _) in visible:
        px, py = pred_points[index]
        dx, dy = px - gx, py - gy
        variance = (2.0 * COCO_SIGMAS[index]) ** 2
        exponent = (dx * dx + dy * dy) / variance / area / 2.0
        total += math.exp(-exponent)
    return total / len(visible)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DATASET_ID)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()

    handoff = ROOT / "02_datasets" / "canonical" / args.dataset
    annotations = json.loads((handoff / "annotations.json").read_text(encoding="utf-8"))
    image_by_id = {int(image["id"]): image for image in annotations["images"]}
    ann_by_image = {int(ann["image_id"]): ann for ann in annotations["annotations"]}
    if len(image_by_id) != 1000 or len(ann_by_image) != 1000:
        raise RuntimeError("Expected exactly 1000 unique images and one annotation per image")

    aggregate = []
    per_joint = []
    for model_id in MODEL_IDS:
        run_dir, manifest = pick_complete_run(model_id, args.dataset)
        predictions = {}
        for prediction in read_jsonl(run_dir / "predictions.jsonl"):
            try:
                image_id = int(prediction["sample_id"].rsplit(":", 1)[1])
            except (KeyError, ValueError, IndexError) as exc:
                raise RuntimeError(f"Invalid sample_id in {run_dir}: {prediction.get('sample_id')}") from exc
            if image_id in predictions:
                raise RuntimeError(f"Duplicate prediction for {model_id} image {image_id}")
            predictions[image_id] = prediction
        if set(predictions) != set(image_by_id):
            raise RuntimeError(f"Prediction image IDs do not match the 1000-image handoff for {model_id}")

        correct = {threshold: 0 for threshold in PCK_THRESHOLDS}
        joint_correct = {threshold: [0] * 17 for threshold in PCK_THRESHOLDS}
        joint_total = [0] * 17
        distances_px = []
        normalized_distances = []
        total_labeled = 0
        oks_values = []
        detections = 0

        for image_id, gt_ann in ann_by_image.items():
            prediction = predictions[image_id]
            pred_points = prediction.get("keypoints_xy") if prediction.get("status") == "ok" else None
            if pred_points is not None and len(pred_points) == 17:
                detections += 1
            else:
                pred_points = None

            flat = gt_ann["keypoints"]
            gt_points = [tuple(map(float, flat[i:i + 3])) for i in range(0, 51, 3)]
            x, y, width, height = map(float, gt_ann["bbox"])
            del x, y
            scale = max(width, height)
            if scale <= 0:
                raise RuntimeError(f"Invalid ground-truth bbox scale for image {image_id}")
            area = float(gt_ann.get("area", width * height))
            if pred_points is not None:
                oks_values.append(oks_for_instance(pred_points, gt_points, area))
            else:
                oks_values.append(0.0)

            for joint_index, (gx, gy, visibility) in enumerate(gt_points):
                if visibility <= 0:
                    continue
                total_labeled += 1
                joint_total[joint_index] += 1
                if pred_points is None:
                    continue
                px, py = map(float, pred_points[joint_index])
                distance = math.hypot(px - gx, py - gy)
                normalized = distance / scale
                distances_px.append(distance)
                normalized_distances.append(normalized)
                for threshold in PCK_THRESHOLDS:
                    if normalized <= threshold:
                        correct[threshold] += 1
                        joint_correct[threshold][joint_index] += 1

        row = {
            "model_id": model_id,
            "run_id": manifest["run_id"],
            "images": len(image_by_id),
            "detections": detections,
            "pose_coverage": detections / len(image_by_id),
            "labeled_keypoints": total_labeled,
            "mean_OKS_single_prediction": sum(oks_values) / len(oks_values),
            "median_pixel_error_detected_labeled_joints": statistics.median(distances_px) if distances_px else None,
            "mean_bbox_normalized_error_detected_labeled_joints": sum(normalized_distances) / len(normalized_distances) if normalized_distances else None,
            "COCO_AP": manifest["timing"]["AP"],
            "COCO_AP50": manifest["timing"]["AP50"],
            "COCO_AP75": manifest["timing"]["AP75"],
            "normalizer": "max(ground-truth bbox width, height)",
            "missing_detections_count_as_PCK_incorrect": True,
        }
        for threshold in PCK_THRESHOLDS:
            row[f"PCK@{threshold:.2f}"] = correct[threshold] / total_labeled if total_labeled else None
        aggregate.append(row)

        for joint_index, name in enumerate(JOINT_NAMES):
            joint_row = {"model_id": model_id, "joint": name, "labeled_instances": joint_total[joint_index]}
            for threshold in PCK_THRESHOLDS:
                joint_row[f"PCK@{threshold:.2f}"] = (
                    joint_correct[threshold][joint_index] / joint_total[joint_index]
                    if joint_total[joint_index] else None
                )
            per_joint.append(joint_row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "spinetrack_unreal_pilot_2026-10-27_localization_metrics.json"
    csv_path = args.output_dir / "spinetrack_unreal_pilot_2026-10-27_localization_metrics.csv"
    joints_path = args.output_dir / "spinetrack_unreal_pilot_2026-10-27_per_joint_pck.csv"
    payload = {
        "dataset_id": args.dataset,
        "metric_definition": {
            "COCO": "official pycocotools COCOeval keypoint AP/AR using OKS thresholds 0.50:0.05:0.95 and default COCO keypoint sigmas",
            "mean_OKS_single_prediction": "per-image OKS using official COCO keypoint sigmas, ground-truth area, and all v>0 joints; no-detection scores 0; averaged over all 1000 images; descriptive companion statistic, not official AP",
            "PCK": "percentage of v>0 ground-truth joints whose Euclidean pixel error is <= threshold * max(ground-truth bbox width, height); missing/no-detection predictions count incorrect",
            "normalizer_source": "ground-truth annotation bbox, not model-predicted bbox",
        },
        "aggregate": aggregate,
        "per_joint": per_joint,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for path, records in ((csv_path, aggregate), (joints_path, per_joint)):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    print(json.dumps({"aggregate_csv": str(csv_path), "per_joint_csv": str(joints_path), "aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    main()
