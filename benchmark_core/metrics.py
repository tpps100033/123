"""Official COCO keypoint evaluation through the cocodataset/cocoapi implementation."""
from __future__ import annotations

import json
from pathlib import Path


COCO_KEYPOINT_STATS = (
    "AP", "AP50", "AP75", "AP_medium", "AP_large",
    "AR", "AR50", "AR75", "AR_medium", "AR_large",
)


def evaluate_coco_keypoints(annotation_file, rows, predictions):
    """Run COCOeval(keypoints) on the exact handoff image IDs.

    The benchmark produces one person pose per image. The per-instance result
    confidence is the arithmetic mean of the model's 17 keypoint confidences;
    this supplies COCOeval's required detection score without changing OKS.
    """
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise RuntimeError("Official metrics require pycocotools; install requirements/evaluation.in in this model environment.") from exc

    gt = COCO(str(annotation_file))
    row_by_sample = {row["sample_id"]: row for row in rows}
    image_ids = [int(row["image_id"]) for row in rows]
    categories = [int(category["id"]) for category in gt.dataset.get("categories", [])]
    if not image_ids or not categories:
        raise ValueError("COCO evaluation requires selected image IDs and person categories")

    results = []
    valid_by_sample = {}
    for prediction in predictions:
        valid = prediction.get("status") == "ok" and len(prediction.get("keypoints_xy") or []) == 17
        valid_by_sample[prediction["sample_id"]] = bool(valid)
        if not valid:
            continue
        points = prediction["keypoints_xy"]
        scores = prediction["keypoint_scores"]
        flat = [value for point, confidence in zip(points, scores) for value in (float(point[0]), float(point[1]), float(confidence))]
        results.append({
            "image_id": int(row_by_sample[prediction["sample_id"]]["image_id"]),
            "category_id": categories[0],
            "keypoints": flat,
            "score": sum(float(value) for value in scores) / len(scores),
        })

    if results:
        dt = gt.loadRes(results)
    else:
        # COCO.loadRes([]) is not supported by all pycocotools releases.
        dt = COCO()
        dt.dataset = {"images": gt.loadImgs(image_ids), "categories": gt.dataset["categories"], "annotations": []}
        dt.createIndex()

    evaluator = COCOeval(gt, dt, "keypoints")
    evaluator.params.imgIds = image_ids
    evaluator.params.catIds = categories
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    values = [float(value) if value >= 0 else None for value in evaluator.stats]
    if len(values) != len(COCO_KEYPOINT_STATS):
        raise RuntimeError(f"Unexpected COCO keypoint summary size: {len(values)}")
    ok = sum(valid_by_sample.values())
    return {
        "metric_protocol": "cocodataset/cocoapi.COCOeval(iouType='keypoints')",
        "metric_protocol_version": "pycocotools COCOeval default keypoint parameters",
        "dataset_annotation": str(Path(annotation_file)),
        "selected_images": len(rows),
        "images_with_pose": ok,
        "pose_coverage_diagnostic": ok / len(rows) if rows else 0.0,
        "instance_score": "mean of 17 model keypoint confidence values",
        **dict(zip(COCO_KEYPOINT_STATS, values)),
    }, valid_by_sample


def per_image_status(rows, predictions, valid_by_sample):
    """Per-image provenance only; official AP/AR are computed at dataset level."""
    pred_by_sample = {prediction["sample_id"]: prediction for prediction in predictions}
    records = []
    for row in rows:
        prediction = pred_by_sample[row["sample_id"]]
        records.append({
            "sample_id": row["sample_id"],
            "image_id": row["image_id"],
            "prediction_status": prediction["status"],
            "included_as_detection": valid_by_sample.get(row["sample_id"], False),
            "error_code": prediction.get("error_code"),
        })
    return records
