"""Summarize completed BabyPose supplementary runs without changing main results."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = "babypose_test160_limb12"
MODELS = ("mediapipe_pose_heavy", "movenet_thunder", "rtmpose_m",
          "yolo11x_pose", "sapiens_1b_coco", "openpose_coco18")


def main():
    output = ROOT / "04_merged_results" / "babypose_supplementary"
    output.mkdir(parents=True, exist_ok=True)
    records = []
    handoff_hashes = set()
    for model in MODELS:
        candidates = sorted((ROOT / "03_model_raw_outputs" / model / DATASET / "fair_reference").glob("*/run_manifest.json"))
        complete = []
        for path in candidates:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if manifest.get("metric_complete") and manifest.get("counts", {}).get("input") == 160:
                complete.append((path, manifest))
        if not complete:
            records.append({"model": model, "status": "not_completed", "AP": "", "AP50": "", "AR": "",
                            "coverage": "", "device": "", "run_id": "", "manifest": ""})
            continue
        path, manifest = complete[-1]
        handoff_hashes.add(manifest.get("data_handoff_hash"))
        summary = manifest.get("timing", {})
        if summary.get("metric_protocol") != "cocodataset/cocoapi.COCOeval(iouType='keypoints')":
            raise RuntimeError(f"Unexpected metric protocol for {model}")
        provenance = manifest.get("model_provenance", manifest.get("provenance", {}))
        records.append({"model": model, "status": "complete", "AP": summary.get("AP"),
                        "AP50": summary.get("AP50"), "AR": summary.get("AR"),
                        "coverage": summary.get("pose_coverage_diagnostic"),
                        "device": provenance.get("runtime_device", ""),
                        "run_id": manifest.get("run_id", ""), "manifest": str(path)})
    if len(handoff_hashes) > 1:
        raise RuntimeError("BabyPose runs used different handoff hashes; rerun models on one frozen subset")
    target = output / "comparison.csv"
    with target.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=records[0]); writer.writeheader(); writer.writerows(records)
    lines = ["# BabyPose supplementary comparison", "", f"Generated: {datetime.now(timezone.utc).isoformat()}", "",
             "Official BabyPose test spreadsheet, deterministic 160-frame subset. Values are COCOeval OKS on the 12 labelled limb joints with a derived tight-limb bounding-box area. Depth-image domain and the missing five face labels make these scores numerically incomparable with full 17-point COCO or SyRIP AP.", "",
             "| Model | Status | AP | AP50 | AR | Pose coverage | Device |", "| --- | --- | ---: | ---: | ---: | ---: | --- |"]
    for row in records:
        def number(value):
            return f"{value:.4f}" if isinstance(value, (float, int)) else "—"
        lines.append(f"| {row['model']} | {row['status']} | {number(row['AP'])} | {number(row['AP50'])} | {number(row['AR'])} | {number(row['coverage'])} | {row['device'] or '—'} |")
    lines.extend(["", "Run provenance is in `comparison.csv` and each referenced run manifest. Scores are absent until a complete 160-image run finishes.", ""])
    (output / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"csv": str(target), "complete_models": sum(row["status"] == "complete" for row in records)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
