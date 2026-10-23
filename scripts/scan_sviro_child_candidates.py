"""Count child-only scenes in cached official SVIRO pose annotation archives."""
from __future__ import annotations

import collections
import hashlib
import json
import re
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVES = ROOT / "02_datasets" / "source_archives"
VEHICLES = (
    "i3", "tucson", "x5", "zoe", "gsf", "hilux", "model3", "tiguan", "aclass", "escape",
)
POSE_SOURCES = {
    "i3": ("bmw-i3", 6, 468),
    "tucson": ("hyundai-tucson", 7, 472),
    "x5": ("bmw-x5", 7, 469),
    "zoe": ("renault-zoe", 7, 475),
    "gsf": ("lexus-gs-f", 6, 473),
    "hilux": ("toyota-hilux", 6, 477),
    "model3": ("tesla-model-3", 7, 476),
    "tiguan": ("vw-tiguan", 7, 478),
    "aclass": ("mercedes-class-a", 6, 474),
    "escape": ("ford-escape", 6, 471),
}
POSE_NAME = re.compile(r"_GT_([0-6])_([0-6])_([0-6])_pose\.json$")


def scan(path: Path, vehicle: str) -> dict:
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist()
                 if f"{vehicle}/test_with_labels/pose_wholeImage/" in n]
    labels_seen: collections.Counter[int] = collections.Counter()
    selected = []
    for name in names:
        match = POSE_NAME.search(Path(name).name)
        if not match:
            continue
        labels = [int(value) for value in match.groups()]
        labels_seen.update(labels)
        if labels.count(2) == 1 and not any(value in (1, 3) for value in labels):
            selected.append(name)
    result = {
        "vehicle": vehicle,
        "archive": path.name,
        "test_pose_files": len(names),
        "child_only_scenes": len(selected),
        "label_counts_by_seat": dict(sorted(labels_seen.items())),
        "selected_source_image_ids": sorted(
            int(match.group(1)) for name in selected
            if (match := re.search(r"_imageID_(\d+)_", Path(name).name))
        ),
        "archive_sha256": sha256(path),
    }
    if result["test_pose_files"] != 500:
        raise RuntimeError(f"Expected 500 official held-out test poses for {vehicle}, found {len(names)}")
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_pose_archive(vehicle: str) -> Path:
    path = ARCHIVES / f"{vehicle}_pose_wholeImage.zip"
    if path.is_file() and zipfile.is_zipfile(path):
        return path
    if path.exists():
        raise RuntimeError(f"Cached annotation archive is incomplete; inspect before replacing: {path}")
    slug, page_suffix, download_id = POSE_SOURCES[vehicle]
    url = f"https://sviro.kl.dfki.de/download/{slug}-{page_suffix}/?wpdmdl={download_id}"
    ARCHIVES.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    # SVIRO's WPDM endpoint can keep ordinary downloads open indefinitely;
    # a byte range returns the complete small annotation ZIP promptly.
    subprocess.run(["curl.exe", "--fail", "--silent", "--show-error", "--location",
                    "--range", "0-9999999", "--max-time", "180", url, "--output", str(temporary)],
                   check=True, timeout=190)
    if not zipfile.is_zipfile(temporary):
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"SVIRO pose endpoint did not return a complete ZIP: {url}")
    with zipfile.ZipFile(temporary) as archive:
        bad_entry = archive.testzip()
        if bad_entry:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"Corrupt entry {bad_entry} in downloaded annotation ZIP")
    temporary.replace(path)
    return path


def main() -> None:
    results = []
    for vehicle in VEHICLES:
        path = fetch_pose_archive(vehicle)
        results.append(scan(path, vehicle))
    report = {
        "status": "verified_inventory",
        "source": "https://sviro.kl.dfki.de/data/",
        "selection": "official test split; exactly one class-2 child and no class-1 infant or class-3 adult",
        "official_test_scenes_per_vehicle": 500,
        "total_child_only_scenes": sum(row["child_only_scenes"] for row in results),
        "vehicles": results,
    }
    output = ROOT / "reports" / "sviro_children_inventory.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "total_child_only_scenes": report["total_child_only_scenes"],
                      "vehicles": {row["vehicle"]: row["child_only_scenes"] for row in results},
                      "report": str(output)}, indent=2))


if __name__ == "__main__":
    main()
