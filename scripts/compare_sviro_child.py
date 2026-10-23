"""Summarize complete child-only SVIRO COCOeval runs."""
from __future__ import annotations
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = "sviro_child_i3_test53_coco12"
MODELS = ("mediapipe_pose_heavy", "movenet_thunder", "rtmpose_m", "yolo11x_pose", "sapiens_1b_coco", "openpose_coco18")
PROTOCOL = "cocodataset/cocoapi.COCOeval(iouType='keypoints')"

def main():
    handoff = json.loads((ROOT / "02_datasets/canonical" / DATASET / "dataset_manifest.json").read_text())
    records, hashes = [], set()
    for model in MODELS:
        paths = sorted((ROOT / "03_model_raw_outputs" / model / DATASET / "fair_reference").glob("*/run_manifest.json"))
        completed = []
        for path in paths:
            manifest = json.loads(path.read_text())
            if manifest.get("metric_complete") and manifest.get("counts", {}).get("input") == 53:
                if manifest.get("data_handoff_hash") != handoff["index_sha256"]:
                    raise RuntimeError(f"{model} used a different SVIRO handoff: {path}")
                if manifest.get("timing", {}).get("metric_protocol") != PROTOCOL:
                    raise RuntimeError(f"{model} did not use official COCOeval: {path}")
                completed.append((path, manifest))
        if not completed:
            records.append({"model": model, "status": "not_completed", "AP": "", "AP50": "", "AP75": "", "AR": "", "device": "", "images": 0, "errors": "", "temperature_c": "", "gpu_memory_mb": "", "validation": "missing", "run_id": ""})
            continue
        path, manifest = completed[-1]
        hashes.add(manifest.get("data_handoff_hash"))
        provenance = manifest.get("model_provenance", manifest.get("provenance", {}))
        timing = manifest.get("timing", {})
        environment = manifest.get("environment", {})
        gpu_memory = environment.get("gpu_memory", {})
        validation_file = ROOT / "06_logs" / f"model_validation_{model}_{DATASET}.json"
        validation = json.loads(validation_file.read_text()) if validation_file.exists() else {}
        records.append({"model": model, "status": "complete", "AP": timing.get("AP"), "AP50": timing.get("AP50"),
                        "AP75": timing.get("AP75"), "AR": timing.get("AR"),
                        "device": provenance.get("runtime_device", ""), "images": manifest.get("counts", {}).get("input", 0),
                        "errors": manifest.get("counts", {}).get("errors", ""),
                        "temperature_c": gpu_memory.get("temperature_c", ""),
                        "gpu_memory_mb": gpu_memory.get("device_used_mb", ""),
                        "validation": validation.get("status", "missing"),
                        "run_id": manifest.get("run_id", "")})
    if len(hashes) > 1:
        raise RuntimeError("Completed child evaluations do not share one frozen sample set")
    output = ROOT / "04_merged_results/sviro_child_comparison"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    lines = ["# SVIRO child pose comparison", "", f"Generated: {datetime.now(timezone.utc).isoformat()}", "",
             "This independent child set contains 53 scenes from the official BMW i3 test split. Each selected scene has exactly one child and no adult or infant. RGB frames, keypoints, visibility, and person boxes come from the official SVIRO release. Twelve limb joints map to COCO-17; the five unannotated face points have visibility zero. The released class-3 person box supplies area. Scores use official COCOeval defaults and are child-domain results, not COCO leaderboard scores.", "",
             "| Model | Status | AP | AP50 | AP75 | AR | Images / errors | Device | Validation | GPU temp at end | Device memory at end |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |"]
    for row in records:
        def n(v): return f"{v:.4f}" if isinstance(v, (int, float)) else "—"
        temp = f"{row['temperature_c']}°C" if row["temperature_c"] != "" else "—"
        memory = f"{row['gpu_memory_mb']:.0f} MiB" if isinstance(row["gpu_memory_mb"], (int, float)) else "—"
        lines.append(f"| {row['model']} | {row['status']} | {n(row['AP'])} | {n(row['AP50'])} | {n(row['AP75'])} | {n(row['AR'])} | {row['images']} / {row['errors']} | {row['device'] or '—'} | {row['validation']} | {temp} | {memory} |")
    lines.extend(["", "The GPU temperature and device-memory columns show the last sample stored in each run manifest, not a maximum. Across the complete run, the highest manually observed GPU temperature was 79°C; the 85°C stop was not reached. MediaPipe and MoveNet have no usable GPU provider in this WSL setup and ran on CPU. Validation requires real single-image and two-image inference; a no-detection result is accepted only when the adapter returns the expected empty keypoint list.", "", "Full per-image predictions and run evidence remain in ignored `03_model_raw_outputs/`; the run IDs are in `comparison.csv`.", ""])
    (output / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    tracked_report = ROOT / "reports" / "sviro_child_comparison.md"
    tracked_report.parent.mkdir(parents=True, exist_ok=True)
    tracked_report.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output": str(output), "completed_models": sum(r["status"] == "complete" for r in records)}))

if __name__ == "__main__": main()
