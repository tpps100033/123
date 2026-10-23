"""Check that the SVIRO COCO-17 annotation handoff scores as a perfect oracle."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark_core.data import load_handoff
from benchmark_core.metrics import evaluate_coco_keypoints


def main() -> None:
    dataset = "sviro_child_i3_test53_coco12"
    handoff_root = ROOT / "02_datasets" / "canonical" / dataset
    manifest, rows = load_handoff(handoff_root)
    predictions = []
    for row in rows:
        points = json.loads(row["gt_xy"])
        predictions.append({
            "sample_id": row["sample_id"],
            "status": "ok",
            "keypoints_xy": points,
            "keypoint_scores": [1.0] * 17,
        })
    annotation_file = ROOT / "02_datasets" / "person_keypoints_sviro_child_i3_test53_coco12.json"
    result, _ = evaluate_coco_keypoints(annotation_file, rows, predictions)
    if result["selected_images"] != 53 or result["images_with_pose"] != 53:
        raise RuntimeError("Oracle did not evaluate all 53 handoff images")
    if result["AP"] != 1.0 or result["AR"] != 1.0:
        raise RuntimeError(f"Perfect-coordinate COCOeval oracle failed: AP={result['AP']} AR={result['AR']}")
    print(json.dumps({"dataset_id": dataset, "index_sha256": manifest["index_sha256"],
                      "protocol": result["metric_protocol"], "selected_images": result["selected_images"],
                      "AP": result["AP"], "AR": result["AR"], "status": "pass"}, indent=2))


if __name__ == "__main__":
    main()
