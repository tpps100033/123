"""Report pooled and per-vehicle COCOeval scores for the 710-image SVIRO test set."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = "sviro_children_allcars_test710_coco12"
EXPECTED = 710
MODELS = ("mediapipe_pose_heavy", "movenet_thunder", "rtmpose_m",
          "yolo11x_pose", "sapiens_1b_coco", "openpose_coco18")
VEHICLES = ("i3", "tucson", "x5", "zoe", "gsf", "hilux", "model3", "tiguan", "aclass", "escape")
PROTOCOL = "cocodataset/cocoapi.COCOeval(iouType='keypoints')"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    from benchmark_core.data import load_handoff
    from benchmark_core.metrics import evaluate_coco_keypoints

    handoff_root = ROOT / "02_datasets/canonical" / DATASET
    handoff, rows = load_handoff(handoff_root)
    if len(rows) != EXPECTED:
        raise RuntimeError(f"Expected {EXPECTED} SVIRO child rows, found {len(rows)}")
    sources = handoff.get("source_selected_files", [])
    if len(sources) != EXPECTED:
        raise RuntimeError("Handoff manifest does not identify all selected vehicle source files")
    vehicle_by_image = {str(index + 1): source["vehicle"] for index, source in enumerate(sources)}
    if set(vehicle_by_image.values()) != set(VEHICLES):
        raise RuntimeError("Vehicle list differs from the frozen handoff")

    annotation_file = ROOT / "02_datasets" / f"person_keypoints_{DATASET}.json"
    pooled_rows, vehicle_rows = [], []
    handoff_hashes = set()
    for model in MODELS:
        candidates = []
        for path in (ROOT / "03_model_raw_outputs" / model / DATASET / "fair_reference").glob("*/run_manifest.json"):
            run = json.loads(path.read_text(encoding="utf-8"))
            if (run.get("metric_complete") and run.get("counts", {}).get("input") == EXPECTED
                    and run.get("counts", {}).get("errors") == 0
                    and run.get("data_handoff_hash") == handoff["index_sha256"]
                    and run.get("metric_protocol") == PROTOCOL):
                candidates.append((run.get("run_id", ""), path, run))
        if not candidates:
            pooled_rows.append({"model_id": model, "status": "not_completed", "images": 0})
            continue
        _, manifest_path, run = sorted(candidates, key=lambda item: item[0])[-1]
        handoff_hashes.add(run["data_handoff_hash"])
        predictions = read_jsonl(manifest_path.parent / "predictions.jsonl")
        pooled_rows.append({
            "model_id": model, "status": "complete", "images": EXPECTED,
            "errors": run["counts"]["errors"], "AP": run["timing"].get("AP"),
            "AP50": run["timing"].get("AP50"), "AP75": run["timing"].get("AP75"),
            "AR": run["timing"].get("AR"),
            "mean_latency_ms": run["timing"].get("latency_ms", {}).get("mean"),
            "device": run.get("model_provenance", {}).get("runtime_device", ""),
            "gpu": run.get("environment", {}).get("gpu"),
            "temperature_c_at_manifest": run.get("environment", {}).get("gpu_memory", {}).get("temperature_c"),
            "device_memory_mb": run.get("environment", {}).get("gpu_memory", {}).get("device_used_mb"),
            "thermal_stop": run.get("thermal_stop"), "run_id": run["run_id"],
        })
        prediction_by_sample = {prediction["sample_id"]: prediction for prediction in predictions}
        if len(predictions) != EXPECTED:
            raise RuntimeError(f"{model}: prediction file has {len(predictions)} rows, expected {EXPECTED}")
        for vehicle in VEHICLES:
            subset_rows = [row for row in rows if vehicle_by_image[str(row["image_id"])] == vehicle]
            subset_predictions = [prediction_by_sample[row["sample_id"]] for row in subset_rows]
            scores, _ = evaluate_coco_keypoints(annotation_file, subset_rows, subset_predictions)
            vehicle_rows.append({"model_id": model, "vehicle": vehicle, "images": len(subset_rows),
                                 "AP": scores["AP"], "AP50": scores["AP50"], "AP75": scores["AP75"],
                                 "AR": scores["AR"], "metric_protocol": scores["metric_protocol"]})
    if len(handoff_hashes) > 1:
        raise RuntimeError("Completed all-car evaluations do not share the same frozen handoff")

    overlap_rows = []
    expanded_by_model = {row["model_id"]: row for row in pooled_rows if row.get("status") == "complete"}
    for model in MODELS:
        i3 = next((row for row in vehicle_rows if row["model_id"] == model and row["vehicle"] == "i3"), None)
        old_paths = sorted((ROOT / "03_model_raw_outputs" / model /
                            "sviro_child_i3_test53_coco12" / "fair_reference").glob("*/run_manifest.json"))
        old_runs = []
        for path in old_paths:
            old = json.loads(path.read_text(encoding="utf-8"))
            if old.get("metric_complete") and old.get("counts", {}).get("input") == 53:
                old_runs.append((old.get("run_id", ""), old))
        current = expanded_by_model.get(model)
        if i3 and current and old_runs:
            _, old = sorted(old_runs, key=lambda item: item[0])[-1]
            deltas = {metric: i3.get(metric) - old.get("timing", {}).get(metric)
                      for metric in ("AP", "AP50", "AP75", "AR")}
            overlap_rows.append({"model_id": model, "old_run_id": old["run_id"],
                                 "new_run_id": current["run_id"], "old_images": 53,
                                 "new_i3_images": i3["images"], **{f"delta_{key}": value for key, value in deltas.items()},
                                 "within_0_02_all_metrics": all(value is not None and abs(value) <= 0.02
                                                                 for value in deltas.values())})
        else:
            overlap_rows.append({"model_id": model, "within_0_02_all_metrics": "not_comparable"})

    output = ROOT / "04_merged_results/sviro_children_allcars_comparison"
    output.mkdir(parents=True, exist_ok=True)
    for name, records in (("comparison.csv", pooled_rows), ("per_vehicle.csv", vehicle_rows),
                          ("i3_overlap_consistency.csv", overlap_rows)):
        if records:
            with (output / name).open("w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for record in records for k in record)))
                writer.writeheader()
                writer.writerows(records)

    lines = ["# SVIRO child pose comparison", "",
             f"Generated: {datetime.now(timezone.utc).isoformat()}", "",
             "The held-out child-only set combines all 10 official SVIRO vehicle test splits. Each split contributes its qualifying scenes from the 500-image official test set; a scene qualifies only when exactly one class-2 child and no class-1 infant or class-3 adult is present. The resulting set has 710 RGB scenes. Official pose labels, class-3 person boxes, and RGB whole-image archives are used. Twelve released limb joints map to COCO-17; the five unsupported face joints are marked unlabelled. All AP/AR values use the official `pycocotools.COCOeval(iouType='keypoints')` defaults. These are synthetic child-domain scores, not COCO leaderboard scores.", "",
             "## Pooled results", "",
             "| Model | Status | AP | AP50 | AP75 | AR | Images / errors | Device | Run |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |"]
    for row in pooled_rows:
        def number(value): return f"{value:.4f}" if isinstance(value, (int, float)) else "—"
        lines.append(f"| {row['model_id']} | {row['status']} | {number(row.get('AP'))} | {number(row.get('AP50'))} | {number(row.get('AP75'))} | {number(row.get('AR'))} | {row.get('images', 0)} / {row.get('errors', '—')} | {row.get('device', '—')} | {row.get('run_id', '—')} |")
    lines.extend(["", "## Per-vehicle COCOeval", "",
                  "`per_vehicle.csv` reports separate official COCOeval summaries for each vehicle. The pooled result weights each selected image through the standard COCOeval aggregation; the vehicle rows expose the domain spread. The 710 count refers to labeled scenes, not 710 independent children: SVIRO reuses synthetic human models across vehicle domains, so no person-level independence claim is made.", "",
                  "`i3_overlap_consistency.csv` compares each expanded run's BMW i3 subset to the preserved 53-image run under the same official evaluator. AP, AP50, AP75, and AR should remain within 0.02; inspect any row outside that tolerance before treating the expansion as methodologically consistent.", "",
                  "GPU temperature and VRAM are recorded in each run manifest. A run stopped at the configured 85°C limit is incomplete and is excluded from the complete-score table. MediaPipe and MoveNet may use CPU where no GPU provider is available. Full predictions and run evidence remain under ignored `03_model_raw_outputs/`.", ""])
    report = "\n".join(lines)
    (output / "comparison.md").write_text(report, encoding="utf-8")
    (ROOT / "reports/sviro_children_comparison.md").write_text(report, encoding="utf-8")
    print(json.dumps({"output": str(output), "completed_models": sum(row.get("status") == "complete" for row in pooled_rows),
                      "per_vehicle_rows": len(vehicle_rows)}, indent=2))


if __name__ == "__main__":
    main()
