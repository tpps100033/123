"""Build a thesis-writing bundle from the frozen three-domain experiment."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
MODELS = (
    "mediapipe_pose_heavy", "movenet_thunder", "rtmpose_m",
    "yolo11x_pose", "sapiens_1b_coco", "openpose_coco18",
)
DATASETS = {
    "coco_singleperson_val2017": (921, "COCO single-person val2017 subset"),
    "syrip_test100": (100, "SyRIP infant test100"),
    "sviro_child_i3_test53_coco12": (53, "SVIRO BMW i3 child-only test subset"),
}
DISPLAY = {
    "mediapipe_pose_heavy": "MediaPipe Pose Heavy", "movenet_thunder": "MoveNet Thunder",
    "rtmpose_m": "RTMPose-M", "yolo11x_pose": "YOLO11x Pose",
    "sapiens_1b_coco": "Sapiens 1B COCO", "openpose_coco18": "OpenPose COCO18",
}
OUTPUT = ROOT / "04_merged_results" / "InfantPose_paper_artifacts_20261023.zip"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(src: Path, stage: Path, rel: str) -> None:
    dst = stage / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def selected_runs(stage: Path) -> list[dict]:
    rows = []
    for dataset, (expected, dataset_name) in DATASETS.items():
        handoff = ROOT / "02_datasets" / "canonical" / dataset
        handoff_manifest = json.loads((handoff / "dataset_manifest.json").read_text(encoding="utf-8"))
        copy_file(handoff / "dataset_manifest.json", stage, f"data_provenance/{dataset}/dataset_manifest.json")
        copy_file(handoff / "sample_index.csv", stage, f"data_provenance/{dataset}/sample_index.csv")
        for model in MODELS:
            candidates = []
            for path in (ROOT / "03_model_raw_outputs" / model / dataset / "fair_reference").glob("*/run_manifest.json"):
                manifest = json.loads(path.read_text(encoding="utf-8"))
                if (manifest.get("metric_complete") and manifest.get("thermal_stop") is None
                        and manifest.get("counts", {}).get("input") == expected
                        and manifest.get("counts", {}).get("errors") == 0
                        and manifest.get("data_handoff_hash") == handoff_manifest["index_sha256"]):
                    candidates.append((manifest.get("started_utc", ""), path, manifest))
            if not candidates:
                raise RuntimeError(f"No complete error-free frozen run: {model} / {dataset}")
            _, manifest_path, manifest = sorted(candidates, key=lambda item: item[0])[-1]
            run_dir = manifest_path.parent
            # Keep paths short for Windows extraction while preserving run_id in the manifest/CSV.
            evidence_rel = f"run_evidence/{dataset}/{model}"
            for filename in ("run_manifest.json", "predictions.jsonl", "metrics.jsonl"):
                source = run_dir / filename
                if not source.is_file():
                    raise FileNotFoundError(source)
                copy_file(source, stage, f"{evidence_rel}/{filename}")
            timing = manifest["timing"]
            memory = manifest.get("environment", {}).get("gpu_memory", {})
            provenance = manifest.get("model_provenance", {})
            rows.append({
                "model_id": model, "model": DISPLAY[model], "dataset_id": dataset,
                "dataset": dataset_name, "images": expected, "errors": manifest["counts"]["errors"],
                "AP": timing.get("AP"), "AP50": timing.get("AP50"), "AP75": timing.get("AP75"),
                "AR": timing.get("AR"), "mean_latency_ms": timing.get("latency_ms", {}).get("mean"),
                "median_latency_ms": timing.get("latency_ms", {}).get("median"),
                "p95_latency_ms": timing.get("latency_ms", {}).get("p95"),
                "runtime_device": provenance.get("runtime_device"),
                "gpu_name": manifest.get("environment", {}).get("gpu"),
                "device_memory_at_manifest_sample_mb": memory.get("device_used_mb"),
                "temperature_at_manifest_sample_c": memory.get("temperature_c"),
                "thermal_limit_c": manifest.get("thermal_limit_c"),
                "data_handoff_sha256": manifest.get("data_handoff_hash"),
                "metric_protocol": timing.get("metric_protocol"),
                "run_id": manifest["run_id"],
            })

    csv_path = stage / "results/final_results_all_domains.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def copy_source_snapshot(stage: Path) -> None:
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode("utf-8").split("\0")
    excluded = {
        "docs/BABYPOSE.md", "scripts/prepare_babypose.py", "scripts/compare_babypose.py",
        "scripts/run_babypose_wsl.sh", "reports/babypose_comparison.md",
    }
    for rel in filter(None, tracked):
        path = PurePosixPath(rel)
        if rel in excluded or path.parts[0] in {"02_datasets", "03_model_raw_outputs", "04_merged_results", "05_analysis_results", "06_logs", ".git"}:
            continue
        src = ROOT.joinpath(*path.parts)
        if src.is_file():
            copy_file(src, stage, f"source/{rel}")
    # Original Drive-export notebooks are deliberately local-only in the project.
    for src in (ROOT / "notebooks" / "original").glob("*.ipynb"):
        copy_file(src, stage, f"source/notebooks/original/{src.name}")


def add_supporting_material(stage: Path) -> None:
    files = {
        "README.md": "README.md",
        "MIGRATION_REPORT.md": "MIGRATION_REPORT.md",
        "PRD_v4.md": "PRD_v4.md",
        "reports/result_comparison.md": "results/result_comparison.md",
        "reports/sviro_child_comparison.md": "results/sviro_child_comparison.md",
        "04_merged_results/merged_20261023T054218Z/benchmark_table.csv": "results/coco_syrip_official_cocoeval_table.csv",
        "04_merged_results/merged_20261023T054218Z/merge_manifest.json": "results/coco_syrip_merge_manifest.json",
        "05_analysis_results/analysis_20261023T054220Z/analysis_manifest.json": "results/analysis_manifest.json",
        "05_analysis_results/analysis_20261023T054220Z/comparison_report.md": "results/coco_syrip_analysis.md",
        "05_analysis_results/analysis_20261023T054220Z/cross_domain_summary.csv": "results/coco_syrip_cross_domain_summary.csv",
        "05_analysis_results/analysis_20261023T054220Z/latency_by_model_domain.csv": "results/coco_syrip_latency.csv",
        "05_analysis_results/analysis_20261023T054220Z/cross_domain_keypoint_ap.png": "figures/coco_syrip_cross_domain_ap.png",
        "05_analysis_results/analysis_20261023T054220Z/latency_median_two_domains.png": "figures/coco_syrip_median_latency.png",
        "02_datasets/person_keypoints_val2017_singleperson.json": "data_provenance/coco/person_keypoints_val2017_singleperson.json",
        "02_datasets/person_keypoints_validate_infant_test100.json": "data_provenance/syrip/person_keypoints_validate_infant_test100.json",
        "02_datasets/person_keypoints_sviro_child_i3_test53_coco12.json": "data_provenance/sviro/person_keypoints_sviro_child_i3_test53_coco12.json",
    }
    for src_rel, dst_rel in files.items():
        src = ROOT / src_rel
        if not src.is_file():
            raise FileNotFoundError(src)
        copy_file(src, stage, dst_rel)
    for src in sorted((ROOT / "docs").glob("*.md")):
        if src.name != "BABYPOSE.md":
            copy_file(src, stage, f"methods/docs/{src.name}")
    for src in sorted((ROOT / "06_logs").glob("model_validation_*.json")):
        if "babypose" not in src.name.lower():
            copy_file(src, stage, f"validation/{src.name}")

    citations = """# Citation and rights notes

Use the exact dataset/model version and locally recorded source metadata in each run manifest. Check the current dataset terms before sharing any source data. The package contains selected annotations and manifests for analysis; it does not contain dataset images or model weights.

## Dataset references

- COCO: T.-Y. Lin et al., “Microsoft COCO: Common Objects in Context,” ECCV 2014. https://arxiv.org/abs/1405.0312. Dataset home and terms: https://cocodataset.org/
- SyRIP: X. Huang, N. Fu, S. Liu, and S. Ostadabbas, “Invariant Representation Learning for Infant Pose Estimation with Small Data,” arXiv:2010.06100 (2020). https://arxiv.org/abs/2010.06100. See the original project for data access and terms: https://github.com/ostadabbas/Infant-Pose-Estimation
- SVIRO: M. Da Cruz et al., “SVIRO: Synthetic Vehicle Interior Rear Seat Occupancy Dataset and Benchmark,” WACV 2020. https://arxiv.org/abs/2001.03483. Dataset site: https://sviro.kl.dfki.de/. The selected source is documented as CC BY-NC-SA 4.0; preserve attribution and share-alike terms when redistributing adapted material.

## Model and metric references

- MediaPipe/BlazePose: V. Bazarevsky et al., “BlazePose: On-device Real-time Body Pose tracking,” arXiv:2006.10204. https://arxiv.org/abs/2006.10204
- MoveNet: Google TensorFlow Hub implementation and model documentation. https://www.tensorflow.org/hub/tutorials/movenet
- OpenPose: Z. Cao et al., “OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields,” arXiv:1812.08008. https://arxiv.org/abs/1812.08008
- RTMPose: T. Jiang et al., “RTMPose: Real-Time Multi-Person Pose Estimation based on MMPose,” arXiv:2303.07399. https://arxiv.org/abs/2303.07399
- Sapiens: R. Khirodkar et al., “Sapiens: Foundation for Human Vision Models,” arXiv:2408.12569. https://arxiv.org/abs/2408.12569
- YOLO11: cite the exact Ultralytics software release/checkpoint recorded in the run manifest; the project states that it has no formal YOLO11 research paper. https://docs.ultralytics.com/models/yolo11/
- Metrics: official COCO keypoint evaluation API. https://github.com/cocodataset/cocoapi and https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py

The benchmark scores are research results, not clinical validation. COCO is a selected 921-image single-person subset, SyRIP is a 100-image infant subset, and SVIRO is a 53-image synthetic child subset. The SVIRO mapping has 12 labeled limb joints; five face points are unannotated. Keep its scores separate from full 17-point COCO/SyRIP comparisons. No numerical comparison to the original Colab metric is supported because the original project used nonstandard metrics.
"""
    (stage / "CITATIONS_AND_RIGHTS.md").write_text(citations, encoding="utf-8")

    readme = """# InfantPose paper-writing package

Generated from the completed local research project. The bundle contains the frozen COCO, SyRIP, and SVIRO child results; 18 complete run packages with per-image predictions and metrics; official COCOeval summaries; selected dataset annotations and hashed sample indexes; validation records; reports, figures, source, notebooks, and citation/rights notes.

## Start here

1. Read `results/result_comparison.md` and `results/sviro_child_comparison.md`.
2. Use `results/final_results_all_domains.csv` for the combined AP/AP50/AP75/AR and latency table. Each row points to the exact `run_id` and handoff hash.
3. Inspect `run_evidence/<dataset>/<model>/<run_id>/` for predictions, per-image metrics, device information, and run manifests.
4. Read `CITATIONS_AND_RIGHTS.md` and `methods/docs/SVIRO_CHILD.md` before writing the methods and dataset sections.
5. Use the code/notebooks in `source/` to trace the unchanged preprocessing, keypoint mapping, and official metric implementation.

## Scope and interpretation

The experiments compare six pretrained pose systems on COCO single-person val2017 (921 images), SyRIP infant Test100, and a 53-image SVIRO BMW i3 child-only test subset. The reports explicitly separate the three image domains. SVIRO has 12 mapped limb joints and five unannotated face points. All score summaries use official `pycocotools.COCOeval(iouType='keypoints')` defaults. The perfect-coordinate evaluator check is documented in the source as an integrity check and is not a model result.

BabyPose is intentionally excluded. Historical Colab project-defined scores are not combined with COCOeval. The COCO table is a 921-image subset and must not be described as the full COCO val2017 leaderboard.

## Data and weights

Source images and model weights are not included. The package contains handoff indexes and annotation JSONs needed to understand and reproduce evaluation, but image access remains governed by the source dataset terms. Dataset manifests record hashes, source URLs, counts, and mapping versions. Model checkpoints are identified in the corresponding run manifests and must be acquired under their own licenses.

## Exact package provenance

Built from Git commit `""" + subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() + """`. The score table is generated by `scripts/build_paper_package.py`. Compare every selected `run_manifest.json` to `results/final_results_all_domains.csv` if reusing the table.
"""
    (stage / "README_PAPER_PACKAGE.md").write_text(readme, encoding="utf-8")


def write_checksums(stage: Path) -> None:
    files = sorted(p for p in stage.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt")
    lines = [f"{sha256(path)}  {path.relative_to(stage).as_posix()}" for path in files]
    (stage / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    # Keep the staging root short so nested run IDs fit Windows path limits.
    with tempfile.TemporaryDirectory(prefix="ipkg_") as temp:
        stage = Path(temp) / "InfantPose_Paper_Artifacts"
        stage.mkdir()
        rows = selected_runs(stage)
        copy_source_snapshot(stage)
        add_supporting_material(stage)
        write_checksums(stage)
        if len(rows) != len(MODELS) * len(DATASETS):
            raise RuntimeError(f"Expected 18 model/domain rows; found {len(rows)}")
        if OUTPUT.exists():
            OUTPUT.unlink()
        with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(stage.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(stage.parent).as_posix())
    with zipfile.ZipFile(OUTPUT) as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f"ZIP CRC check failed: {bad}")
        names = archive.namelist()
        if len([name for name in names if name.endswith("/run_manifest.json")]) != 18:
            raise RuntimeError("ZIP does not contain the 18 selected run manifests")
        if any("/raw/" in name or "model_cache" in name or "babypose" in name.lower() for name in names):
            raise RuntimeError("Source images or model cache leaked into package")
        if any(name.lower().endswith((".jpg", ".jpeg", ".bmp")) for name in names):
            raise RuntimeError("Dataset source images leaked into package")
        checksum_entry = next(name for name in names if name.endswith("/SHA256SUMS.txt"))
        for line in archive.read(checksum_entry).decode("utf-8").splitlines():
            expected, member = line.split("  ", 1)
            member_path = str(PurePosixPath(checksum_entry).parent / member)
            actual = hashlib.sha256(archive.read(member_path)).hexdigest()
            if actual != expected:
                raise RuntimeError(f"SHA-256 mismatch inside archive: {member}")
        table_entry = next(name for name in names if name.endswith("/results/final_results_all_domains.csv"))
        with archive.open(table_entry) as stream:
            result_rows = list(csv.DictReader((line.decode("utf-8-sig") for line in stream)))
        if len(result_rows) != 18:
            raise RuntimeError(f"Combined result table has {len(result_rows)} rows, expected 18")
    print(json.dumps({"status": "complete", "archive": str(OUTPUT), "files": len(names),
                      "size_mib": round(OUTPUT.stat().st_size / 1024**2, 2), "runs": len(rows),
                      "sha256": sha256(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
