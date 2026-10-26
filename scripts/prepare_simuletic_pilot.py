"""Prepare the small public Simuletic HF sample as an audited COCO-17 pilot.

The source is synthetic and only 111 labeled images, so its scores are a
workflow pilot—not a representative general-human benchmark or a 300+ cohort.
Raw source images stay in the ignored source_archives cache.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import struct
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "simuletic_cctv_fall_singleperson_pilot111"
SOURCE_URL = "https://huggingface.co/datasets/Simuletic/CCTV_Incident_Dataset_Fall_Lying_Down_Detection"
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
    [6, 12], [7, 13], [6, 7], [6, 8], [8, 10],
    [7, 9], [9, 11], [2, 3], [1, 2], [1, 3],
    [2, 4], [3, 5], [4, 6], [5, 7],
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_revision(source_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        header = stream.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"Invalid PNG header: {path}")
    return struct.unpack(">II", header[16:24])


def parse_label(label_path: Path, width: int, height: int, annotation_id: int, image_id: int):
    annotations = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 56:
            raise ValueError(f"{label_path}:{line_number}: expected 56 YOLO-pose values, got {len(fields)}")
        values = [float(value) for value in fields]
        class_id = int(values[0])
        if values[0] != class_id or class_id not in (0, 1):
            raise ValueError(f"{label_path}:{line_number}: unexpected class id {values[0]}")
        cx, cy, bw, bh = values[1:5]
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
            raise ValueError(f"{label_path}:{line_number}: invalid normalized bounding box")
        bbox = [(cx - bw / 2) * width, (cy - bh / 2) * height, bw * width, bh * height]
        keypoints, num_keypoints = [], 0
        for index in range(17):
            x, y, visibility = values[5 + index * 3:8 + index * 3]
            if visibility not in (0, 1, 2):
                raise ValueError(f"{label_path}:{line_number}: invalid COCO visibility {visibility}")
            if visibility == 0:
                keypoints.extend((0.0, 0.0, 0))
            else:
                if not (0 <= x <= 1 and 0 <= y <= 1):
                    raise ValueError(f"{label_path}:{line_number}: labeled keypoint outside normalized image")
                keypoints.extend((x * width, y * height, int(visibility)))
                num_keypoints += 1
        if num_keypoints == 0:
            raise ValueError(f"{label_path}:{line_number}: annotation has no labeled keypoints")
        annotations.append({
            "id": annotation_id + len(annotations), "image_id": image_id, "category_id": 1,
            "keypoints": keypoints, "num_keypoints": num_keypoints, "bbox": bbox,
            "area": bbox[2] * bbox[3], "iscrowd": 0, "source_class_id": class_id,
        })
    if not annotations:
        raise ValueError(f"No pose annotations in {label_path}")
    return annotations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root", type=Path,
        default=ROOT / "02_datasets/source_archives/simuletic_cctv_fall_pilot/laying_dataset",
        help="Downloaded HF repository's laying_dataset directory.",
    )
    args = parser.parse_args()
    source_root = args.source_root.expanduser().resolve()
    images_root, labels_root = source_root / "images", source_root / "labels"
    if not images_root.is_dir() or not labels_root.is_dir():
        raise FileNotFoundError(f"Expected images/ and labels/ under {source_root}")
    pairs = []
    for image_path in sorted(images_root.glob("*.png")):
        label_path = labels_root / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing pose label for image: {image_path.name}")
        pairs.append((image_path.resolve(), label_path.resolve()))
    if len(pairs) != 111:
        raise RuntimeError(f"Expected the source card's 111 labeled PNG files, found {len(pairs)}")

    output_root = ROOT / "02_datasets/canonical" / DATASET_ID
    annotation_path, index_path = output_root / "annotations.json", output_root / "sample_index.csv"
    manifest_path = output_root / "dataset_manifest.json"
    reuse_annotations = False
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("source_annotation_sha256") == sha256(annotation_path):
            reuse_annotations = True
        else:
            raise FileExistsError(f"Existing pilot handoff differs; preserve and inspect it: {output_root}")
    if not reuse_annotations and any(path.exists() for path in (annotation_path, index_path)):
        raise FileExistsError(f"Refusing to overwrite partial pilot outputs in {output_root}")

    images, annotations, rows = [], [], []
    annotation_id = 1
    source_annotations = 0
    source_class_counts = {"laying": 0, "standing": 0}
    scored_class_counts = {"laying": 0, "standing": 0}
    for image_id, (image_path, label_path) in enumerate(pairs, 1):
        width, height = png_size(image_path)
        image_annotations = parse_label(label_path, width, height, annotation_id, image_id)
        annotation_id += len(image_annotations)
        source_annotations += len(image_annotations)
        for item in image_annotations:
            source_class_counts["laying" if item["source_class_id"] == 0 else "standing"] += 1
        images.append({"id": image_id, "file_name": image_path.name, "width": width,
                       "height": height, "license": 1})
        primary = max(image_annotations, key=lambda item: item["area"])
        annotations.append(primary)
        scored_class_counts["laying" if primary["source_class_id"] == 0 else "standing"] += 1
        flat = primary["keypoints"]
        rows.append({
            "sample_id": f"{DATASET_ID}:{image_id}", "dataset_id": DATASET_ID,
            "image_id": image_id, "image_path": Path(os.path.relpath(image_path, output_root)).as_posix(), "width": width, "height": height,
            "bbox_xywh": json.dumps(primary["bbox"]),
            "gt_xy": json.dumps([[flat[i], flat[i + 1]] for i in range(0, 51, 3)]),
            "visibility": json.dumps([flat[i] for i in range(2, 51, 3)]),
        })

    coco = {
        "info": {"description": "Simuletic synthetic CCTV fall-pose free sample", "year": 2025,
                 "source": SOURCE_URL},
        "licenses": [{"id": 1, "name": "CC BY 4.0",
                      "url": "https://creativecommons.org/licenses/by/4.0/"}],
        "images": images, "annotations": annotations,
        "categories": [{"id": 1, "name": "person", "supercategory": "person",
                        "keypoints": KEYPOINT_NAMES, "skeleton": SKELETON}],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    if not reuse_annotations:
        annotation_path.write_text(json.dumps(coco, ensure_ascii=False), encoding="utf-8")
    with index_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    labels_digest = hashlib.sha256()
    for _, label_path in pairs:
        labels_digest.update(label_path.name.encode("utf-8"))
        labels_digest.update(bytes.fromhex(sha256(label_path)))
    manifest = {
        "schema_version": "v4.dataset-handoff", "dataset_id": DATASET_ID,
        "sample_count": len(rows), "annotation_count": len(annotations),
        "source_annotation_count": source_annotations,
        "index": index_path.name, "index_sha256": sha256(index_path),
        "source_annotation_sha256": sha256(annotation_path),
        "source_labels_sha256": labels_digest.hexdigest(),
        "annotation_contract": "COCO-17 exact source schema; YOLO normalized xyv converted to pixel xyv",
        "mapping_version": "simuletic_yolo_coco17_xyv_to_cocoapi_v1",
        "source_url": SOURCE_URL, "source_revision": source_revision(source_root.parent),
        "license": "CC BY 4.0", "benchmark_role": "single-person COCOeval pipeline pilot; not general-human benchmark",
        "person_selection": "largest labeled bounding-box area per image, matching the existing single-person handoff rule",
        "source_class_counts": source_class_counts,
        "scored_class_counts": scored_class_counts,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "prepared", "dataset_id": DATASET_ID, "sample_count": len(rows),
                      "annotation_count": len(annotations), "source_annotation_count": source_annotations,
                      "source_class_counts": source_class_counts, "scored_class_counts": scored_class_counts,
                      "source_revision": manifest["source_revision"], "manifest": str(manifest_path)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
