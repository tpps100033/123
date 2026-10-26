"""Create a reproducible COCO-17 single-person handoff from Quarry Crowd.

The source pack contains 300 Unreal Engine frames and 3,692 people. It labels
the 12 body joints that exist in COCO-17 but has no face landmarks. This
converter preserves every source annotation in an audit JSON, then selects
the largest labeled person in each image to match the project's existing
single-person inference contract. Missing face points stay unlabeled (v=0).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "nameframe_quarry_crowd_singleperson_pilot300"
SOURCE_URL = "https://getnameframe.com/datasets/quarry-crowd"
ARCHIVE_SHA256 = "802f5389b8dc2420ed1ed30bca9e831eae14d96d3f06f4795b22c0469983124c"
ARCHIVE_SIZE = 597040148
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


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        header = stream.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"Invalid PNG header: {path}")
    return struct.unpack(">II", header[16:24])


def parse_label(path: Path, width: int, height: int, annotation_id: int, image_id: int) -> list[dict]:
    result = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 56:
            raise ValueError(f"{path}:{line_no}: expected class + bbox + 17 xyv triplets; got {len(fields)} values")
        values = [float(value) for value in fields]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{path}:{line_no}: non-finite value")
        if values[0] != 0:
            raise ValueError(f"{path}:{line_no}: expected the single person class id 0")
        cx, cy, bw, bh = values[1:5]
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
            raise ValueError(f"{path}:{line_no}: invalid normalized bbox")
        bbox = [(cx - bw / 2) * width, (cy - bh / 2) * height, bw * width, bh * height]
        keypoints = []
        num_keypoints = 0
        for index in range(17):
            x, y, visibility = values[5 + index * 3:8 + index * 3]
            if visibility not in (0, 1, 2):
                raise ValueError(f"{path}:{line_no}: invalid visibility value {visibility}")
            if visibility == 0:
                keypoints.extend((0.0, 0.0, 0))
            else:
                if not (0 <= x <= 1 and 0 <= y <= 1):
                    raise ValueError(f"{path}:{line_no}: labeled keypoint outside normalized image")
                keypoints.extend((x * width, y * height, int(visibility)))
                num_keypoints += 1
        if keypoints[:15] != [0.0, 0.0, 0] * 5:
            raise ValueError(f"{path}:{line_no}: source no longer has the documented five unlabeled face points")
        result.append({
            "id": annotation_id + len(result), "image_id": image_id, "category_id": 1,
            "keypoints": keypoints, "num_keypoints": num_keypoints, "bbox": bbox,
            "area": bbox[2] * bbox[3], "iscrowd": 0,
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root", type=Path,
        default=ROOT / "02_datasets/source_archives/nameframe_quarry_crowd/extracted",
    )
    args = parser.parse_args()
    source = args.source_root.expanduser().resolve()
    archive = source.parent / "nameframe-quarry-crowd-v1.zip"
    if archive.stat().st_size != ARCHIVE_SIZE or sha256(archive) != ARCHIVE_SHA256:
        raise RuntimeError("Source archive size or SHA-256 does not match the official data page")
    if "images" not in {p.name for p in source.iterdir()} or "labels" not in {p.name for p in source.iterdir()}:
        raise FileNotFoundError(f"Expected images/ and labels/ in {source}")
    image_root, label_root = source / "images", source / "labels"
    pairs = []
    for image in sorted(image_root.glob("*.png")):
        label = label_root / f"{image.stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(f"Missing label for {image.name}")
        pairs.append((image.resolve(), label.resolve()))
    if len(pairs) != 300:
        raise RuntimeError(f"Expected exactly 300 images from the published sample, found {len(pairs)}")

    output = ROOT / "02_datasets/canonical" / DATASET_ID
    files = [output / name for name in ("annotations.json", "source_annotations_all.json", "sample_index.csv", "dataset_manifest.json")]
    if any(path.exists() for path in files):
        raise FileExistsError(f"Refusing to overwrite an existing handoff; inspect it first: {output}")

    coco_images, scored_annotations, all_annotations, rows = [], [], [], []
    labels_hash = hashlib.sha256()
    next_annotation_id = 1
    source_counts: list[int] = []
    valid_source_annotations = 0
    zero_keypoint_annotations = 0
    for image_id, (image, label) in enumerate(pairs, 1):
        width, height = png_size(image)
        if (width, height) != (1280, 960):
            raise ValueError(f"Unexpected image dimensions for {image.name}: {width}x{height}")
        current = parse_label(label, width, height, next_annotation_id, image_id)
        next_annotation_id += len(current)
        source_counts.append(len(current))
        all_annotations.extend(current)
        valid_source_annotations += sum(item["num_keypoints"] > 0 for item in current)
        zero_keypoint_annotations += sum(item["num_keypoints"] == 0 for item in current)
        labels_hash.update(label.name.encode("utf-8"))
        labels_hash.update(bytes.fromhex(sha256(label)))
        coco_images.append({"id": image_id, "file_name": image.name, "width": width, "height": height, "license": 1})
        eligible = [item for item in current if item["num_keypoints"] > 0]
        if not eligible:
            raise ValueError(f"No person with labeled keypoints in {image.name}; cannot score this image under COCOeval")
        primary = max(eligible, key=lambda item: item["area"])
        scored_annotations.append(primary)
        flat = primary["keypoints"]
        rows.append({
            "sample_id": f"{DATASET_ID}:{image_id}", "dataset_id": DATASET_ID, "image_id": image_id,
            "image_path": Path(os.path.relpath(image, output)).as_posix(), "width": width, "height": height,
            "bbox_xywh": json.dumps(primary["bbox"]),
            "gt_xy": json.dumps([[flat[i], flat[i + 1]] for i in range(0, 51, 3)]),
            "visibility": json.dumps([flat[i] for i in range(2, 51, 3)]),
        })

    categories = [{"id": 1, "name": "person", "supercategory": "person",
                   "keypoints": KEYPOINT_NAMES, "skeleton": SKELETON}]
    licenses = [{"id": 1, "name": "NameFrame sample pack terms", "url": SOURCE_URL}]
    common_info = {"year": 2026, "source": SOURCE_URL, "seed": 20260930,
                   "map": "Quarry_Slate", "engine": "Unreal Engine 5.8"}
    scored_coco = {"info": {**common_info, "description": "NameFrame Quarry Crowd 300-image largest-person COCO-17 handoff"},
                   "licenses": licenses, "images": coco_images, "annotations": scored_annotations,
                   "categories": categories}
    full_coco = {"info": {**common_info, "description": "NameFrame Quarry Crowd source annotations; all people retained"},
                 "licenses": licenses, "images": coco_images, "annotations": all_annotations,
                 "categories": categories}
    output.mkdir(parents=True, exist_ok=True)
    files[0].write_text(json.dumps(scored_coco, ensure_ascii=False), encoding="utf-8")
    files[1].write_text(json.dumps(full_coco, ensure_ascii=False), encoding="utf-8")
    with files[2].open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "schema_version": "v4.dataset-handoff", "dataset_id": DATASET_ID, "sample_count": len(rows),
        "annotation_count": len(scored_annotations), "source_annotation_count": len(all_annotations),
        "source_annotations_with_keypoints": valid_source_annotations,
        "source_annotations_without_keypoints": zero_keypoint_annotations,
        "source_annotation_sha256": sha256(files[0]), "all_source_annotations_sha256": sha256(files[1]),
        "index": files[2].name, "index_sha256": sha256(files[2]),
        "source_labels_sha256": labels_hash.hexdigest(), "source_archive_sha256": ARCHIVE_SHA256,
        "source_archive_bytes": ARCHIVE_SIZE, "annotation_contract": "COCO-17 slots; exact source xyv mapped for 12 body joints; face points explicitly unlabeled; source visibility preserved",
        "mapping_version": "nameframe_yolo_pose_12_body_coco17_xyv_to_pixel_v1",
        "source_url": SOURCE_URL, "source_revision": "sample_version_1.0_seed_20260930",
        "license": "NameFrame free benchmark terms; no redistribution as a dataset; third-party UE assets restrict redistribution",
        "benchmark_role": "single-person-per-image synthetic domain pilot; official COCOeval keypoint OKS",
        "person_selection": "largest labeled bbox per image; all 3,692 source person annotations preserved in source_annotations_all.json",
        "unlabeled_keypoints": KEYPOINT_NAMES[:5], "keypoint_counts_per_person_distribution": {
            str(k): sum(item["num_keypoints"] == k for item in all_annotations) for k in range(13)
        }, "keypoint_counts_per_image": {
            "min": min(source_counts), "median": sorted(source_counts)[len(source_counts)//2], "max": max(source_counts),
        },
        "known_source_limitations": [
            "Five COCO face points (nose, eyes, ears) are v=0 for every person; only 12 body joints are evaluated.",
            "Source validator reports 74 near-duplicate frames within the 300-image pack.",
            "Source reports 2/3492 engine-visible people without a label (0.06%).",
            "One quarry map and a narrow camera box; not a general-scene or full-person-scale benchmark.",
            "Single largest person per image is scored to preserve the project's single-pose adapter protocol; the full source annotations remain archived separately.",
        ],
    }
    files[3].write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "prepared", "dataset_id": DATASET_ID, "images": len(rows),
                      "scored_annotations": len(scored_annotations), "all_source_annotations": len(all_annotations),
                      "keypoints_per_person_distribution": manifest["keypoint_counts_per_person_distribution"],
                      "face_points_unlabeled": 5,
                      "index_sha256": manifest["index_sha256"], "manifest": str(files[3])},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
