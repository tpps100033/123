"""Create a deterministic 1,000-image COCO-17 handoff from SpineTrack-Unreal."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "spinetrack_unreal_singleperson_pilot1000"
SOURCE_URL = "https://huggingface.co/datasets/saifkhichi96/spinetrack"
SOURCE_REVISION = "70ec456"
IMAGE_ARCHIVE_SHA256 = "4212a2b5c9925e5ff7cfe8f3e28f5af9b6319b748edb635d407be08a08fe35c7"
ANNOTATION_ARCHIVE_SHA256 = "3032ffad3984f91bb462bbf6cdeeed392a21f986cfd76d023aa6fb7ae17e5f6b"
SEED = 20261027
TARGET = 1000
COCO_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
SOURCE_KEYPOINT_NAMES_37 = COCO_NAMES + [
    "head", "neck", "hip", "left_big_toe", "right_big_toe",
    "left_small_toe", "right_small_toe", "left_heel", "right_heel",
    "spine_01", "spine_02", "spine_03", "spine_04", "spine_05",
    "left_latissimus", "right_latissimus", "left_clavicle", "right_clavicle",
    "neck_02", "neck_03",
]
SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13],
    [6, 7], [6, 8], [8, 10], [7, 9], [9, 11], [2, 3], [1, 2], [1, 3],
    [2, 4], [3, 5], [4, 6], [5, 7],
]
SOURCE_COCO_INDICES = list(range(17))  # author's metainfo.py explicitly names indices 0–16 as COCO.


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_group(image: dict) -> str:
    source = image.get("source_file_path", "")
    return os.path.dirname(source) if source else "__no_source_path__"


def frame_number(image: dict) -> int:
    match = re.search(r"frame_(\d+)", image.get("source_file_path", ""))
    return int(match.group(1)) if match else int(image["id"])


def select_stratified(images: list[dict], target: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    groups: dict[str, list[dict]] = defaultdict(list)
    for image in images:
        groups[source_group(image)].append(image)
    group_names = list(groups)
    rng.shuffle(group_names)
    for name in group_names:
        groups[name].sort(key=lambda x: (frame_number(x), x["file_name"]))
        # Randomize the initial temporal phase so each source sequence contributes
        # examples from its full duration instead of always starting at frame zero.
        offset = rng.randrange(len(groups[name]))
        groups[name] = groups[name][offset:] + groups[name][:offset]
    selected: list[dict] = []
    cursors = Counter()
    selected_frames: dict[str, list[int]] = defaultdict(list)
    minimum_frame_gap = 5
    while len(selected) < target:
        progressed = False
        for name in group_names:
            values = groups[name]
            cursor = cursors[name]
            while cursor < len(values):
                image = values[cursor]
                cursor += 1
                current_frame = frame_number(image)
                if any(abs(current_frame - previous) < minimum_frame_gap for previous in selected_frames[name]):
                    continue
                cursors[name] = cursor
                selected_frames[name].append(current_frame)
                selected.append(image)
                progressed = True
                break
            else:
                cursors[name] = cursor
            if len(selected) == target:
                break
        if not progressed:
            raise RuntimeError(f"Only {len(selected)} images satisfy sequence-stratified spacing; need {target}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "02_datasets/source_archives/spinetrack_images.zip")
    parser.add_argument("--annotations", type=Path, default=ROOT / "02_datasets/source_archives/spinetrack_annotations.zip")
    parser.add_argument("--target", type=int, default=TARGET)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    output = ROOT / "02_datasets/canonical" / DATASET_ID
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing handoff: {output}")
    if not args.archive.is_file() or args.archive.stat().st_size != 19_407_030_602:
        raise RuntimeError("SpineTrack images.zip is missing or its size differs from the official 19,407,030,602-byte file")
    if sha256_file(args.archive) != IMAGE_ARCHIVE_SHA256:
        raise RuntimeError("SpineTrack images.zip SHA-256 does not match the pinned Hugging Face file object")
    if sha256_file(args.annotations) != ANNOTATION_ARCHIVE_SHA256:
        raise RuntimeError("SpineTrack annotations.zip SHA-256 does not match the pinned Hugging Face file object")

    with zipfile.ZipFile(args.annotations) as labels_zip:
        dataset = json.loads(labels_zip.read("annotations/person_keypoints_train-unreal.json"))
    images = dataset["images"]
    annotations_by_image = {int(a["image_id"]): a for a in dataset["annotations"]}
    if len(images) != 25_000 or len(annotations_by_image) != len(images):
        raise RuntimeError("Unexpected synthetic split size or duplicate/missing image annotation IDs")
    category = dataset["categories"][0]
    if len(category["keypoints"]) != 37 or len(SOURCE_KEYPOINT_NAMES_37) != 37 or len(COCO_NAMES) != 17:
        raise RuntimeError("Unexpected SpineTrack skeleton schema")
    if any(int(image["id"]) not in annotations_by_image for image in images):
        raise RuntimeError("Unmatched source image_id in SpineTrack annotations")

    # COCO v=2 denotes a labeled point visible in the image. Keep source labels
    # untouched, but exclude records with labeled COCO points outside the frame.
    eligible_images = []
    invalid_visible_point_images = []
    for image in images:
        points = annotations_by_image[int(image["id"])]["keypoints"]
        outside = any(
            int(points[index * 3 + 2]) > 0
            and not (0 <= points[index * 3] <= int(image["width"])
                     and 0 <= points[index * 3 + 1] <= int(image["height"]))
            for index in SOURCE_COCO_INDICES
        )
        (invalid_visible_point_images if outside else eligible_images).append(image)
    if len(eligible_images) < args.target:
        raise RuntimeError(f"Only {len(eligible_images)} images remain after COCO point-bound filtering")
    chosen = select_stratified(eligible_images, args.target, args.seed)
    archive_path = args.archive.resolve()
    archive_bytes = args.archive.stat().st_size
    output.mkdir(parents=True, exist_ok=True)
    image_dir = output / "images"
    image_dir.mkdir()
    source_members: dict[str, str] = {}
    source_basenames = {Path(x["file_name"]).name for x in chosen}
    visibility_values = Counter()
    visible_out_of_bounds = 0
    bbox_out_of_bounds = 0
    with zipfile.ZipFile(args.archive) as source_zip:
        candidates = [
            info for info in source_zip.infolist()
            if not info.is_dir()
            and "train-unreal/" in info.filename.replace("\\", "/")
            and Path(info.filename).name in source_basenames
        ]
        for info in candidates:
            base = Path(info.filename).name
            if base in source_members:
                raise RuntimeError(f"Ambiguous archive member basename: {base}")
            source_members[base] = info.filename
        missing = sorted(source_basenames - set(source_members))
        if missing:
            raise FileNotFoundError(f"{len(missing)} sampled images missing under train-unreal/ (example {missing[:3]})")

        output_images, output_annotations, sample_rows, source_annotations = [], [], [], []
        next_ann = 1
        visible_counts = Counter()
        sampled_hashes: dict[str, str] = {}
        used_hashes = set()
        accepted_frames: dict[str, list[int]] = defaultdict(list)
        accepted = 0
        # If exact duplicate RGB files exist, deterministically extend the sample
        # from the same stratified stream until 1,000 unique images are retained.
        chosen_ids = {int(y["id"]) for y in chosen}
        remaining = [x for x in eligible_images if int(x["id"]) not in chosen_ids]
        random.Random(args.seed + 1).shuffle(remaining)
        candidates_order = chosen + remaining
        # Remove accidental repeats while preserving first occurrence.
        seen_ids = set()
        candidates_order = [x for x in candidates_order if not (int(x["id"]) in seen_ids or seen_ids.add(int(x["id"]))) ]
        for source_image in candidates_order:
            sequence = source_group(source_image)
            frame = frame_number(source_image)
            if any(abs(frame - previous) < 5 for previous in accepted_frames[sequence]):
                continue
            member = source_members.get(Path(source_image["file_name"]).name)
            if member is None:
                # Members are resolved lazily for fallback candidates.
                basename = Path(source_image["file_name"]).name
                matches = [i.filename for i in source_zip.infolist() if not i.is_dir() and "train-unreal/" in i.filename.replace("\\", "/") and Path(i.filename).name == basename]
                if len(matches) != 1:
                    continue
                member = matches[0]
                source_members[basename] = member
            image_bytes = source_zip.read(member)
            digest = sha256_bytes(image_bytes)
            if digest in used_hashes:
                continue
            with Image.open(__import__("io").BytesIO(image_bytes)) as im:
                actual_size = im.size
                im.verify()
            if actual_size != (int(source_image["width"]), int(source_image["height"])):
                raise RuntimeError(f"Image size mismatch for {source_image['file_name']}: {actual_size}")
            ann = annotations_by_image[int(source_image["id"])]
            if len(ann["keypoints"]) != 111:
                raise RuntimeError(f"Expected 37 source keypoints in annotation {ann['id']}")
            output_id = accepted + 1
            file_name = f"{output_id:06d}.png"
            (image_dir / file_name).write_bytes(image_bytes)
            img_record = {
                "id": output_id, "file_name": f"images/{file_name}",
                "width": int(source_image["width"]), "height": int(source_image["height"]),
                "license": 1,
            }
            source_kpts = ann["keypoints"]
            coco_kpts = [v for idx in SOURCE_COCO_INDICES for v in source_kpts[idx * 3:idx * 3 + 3]]
            if len(coco_kpts) != 51:
                raise RuntimeError("COCO-17 projection did not produce 51 values")
            for idx in range(17):
                x, y, visibility = coco_kpts[idx * 3:idx * 3 + 3]
                visibility = int(visibility)
                if visibility not in (0, 1, 2):
                    raise RuntimeError(f"Invalid visibility value {visibility} in source annotation {ann['id']}")
                visibility_values[str(visibility)] += 1
                if visibility > 0 and not (0 <= x <= img_record["width"] and 0 <= y <= img_record["height"]):
                    visible_out_of_bounds += 1
            count = sum(int(coco_kpts[i]) > 0 for i in range(2, 51, 3))
            visible_counts[count] += 1
            bbox = [float(v) for v in ann["bbox"]]
            if bbox[0] < 0 or bbox[1] < 0 or bbox[0] + bbox[2] > img_record["width"] or bbox[1] + bbox[3] > img_record["height"]:
                bbox_out_of_bounds += 1
            out_ann = {
                "id": next_ann, "image_id": output_id, "category_id": 1,
                "keypoints": coco_kpts, "num_keypoints": count,
                "bbox": bbox, "area": float(ann.get("area", bbox[2] * bbox[3])),
                "iscrowd": 0,
            }
            output_images.append(img_record)
            output_annotations.append(out_ann)
            source_annotations.append({**ann, "sample_image_id": output_id, "source_image_id": int(source_image["id"]), "source_file_name": source_image["file_name"]})
            sample_rows.append({
                "sample_id": f"{DATASET_ID}:{output_id}", "dataset_id": DATASET_ID,
                "image_id": output_id, "image_path": f"images/{file_name}",
                "source_image_id": source_image["id"], "source_file_name": source_image["file_name"],
                "source_sequence": source_group(source_image), "width": img_record["width"], "height": img_record["height"],
                "bbox_xywh": json.dumps(bbox),
                "gt_xy": json.dumps([[coco_kpts[i], coco_kpts[i + 1]] for i in range(0, 51, 3)]),
                "visibility": json.dumps([coco_kpts[i] for i in range(2, 51, 3)]),
                "image_sha256": digest,
            })
            sampled_hashes[file_name] = digest
            used_hashes.add(digest)
            accepted_frames[sequence].append(frame)
            accepted += 1
            next_ann += 1
            if accepted == args.target:
                break
        if accepted != args.target:
            raise RuntimeError(f"Only {accepted} unique samples extracted; expected {args.target}")

    categories = [{"id": 1, "name": "person", "supercategory": "person", "keypoints": COCO_NAMES, "skeleton": SKELETON}]
    common_info = {
        "year": 2025, "source": SOURCE_URL, "revision": SOURCE_REVISION,
        "description": "SpineTrack-Unreal synthetic single-person image sample; source COCO indices 0-16 copied into standard COCO-17 order",
        "sampling_seed": args.seed, "sampling_method": "round-robin across source motion-sequence folders; minimum 5-frame gap within each sequence; exact RGB duplicates skipped",
        "source_split": "train-unreal only; no real images included",
        "sampling_quality_filter": "Exclude images where any labeled COCO-17 point (v>0) falls outside the image; keep all labels for retained images unchanged.",
    }
    licenses = [{"id": 1, "name": "CC BY-NC 4.0", "url": "https://creativecommons.org/licenses/by-nc/4.0/"}]
    coco = {"info": common_info, "licenses": licenses, "images": output_images, "annotations": output_annotations, "categories": categories}
    annotation_path = output / "annotations.json"
    annotation_path.write_text(json.dumps(coco, ensure_ascii=False), encoding="utf-8")
    source_annotation_path = output / "source_annotations_37pt.json"
    source_categories = json.loads(json.dumps(dataset["categories"]))
    source_categories[0]["keypoints"] = SOURCE_KEYPOINT_NAMES_37
    source_annotation_path.write_text(json.dumps({"info": common_info, "categories": source_categories, "annotations": source_annotations}, ensure_ascii=False), encoding="utf-8")
    index_path = output / "sample_index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(sample_rows[0]))
        writer.writeheader()
        writer.writerows(sample_rows)
    manifest = {
        "schema_version": "v4.dataset-handoff", "dataset_id": DATASET_ID,
        "source_url": SOURCE_URL, "source_revision": SOURCE_REVISION,
        "license": "CC BY-NC 4.0; attribute the dataset authors; non-commercial use only",
        "source_split": "train-unreal, synthetic only",
        "source_images": 25000, "source_annotations": 25000,
        "source_annotations_excluded_invalid_visible_coco_points": len(invalid_visible_point_images),
        "sample_count": len(output_images), "annotation_count": len(output_annotations),
        "source_archive_bytes": archive_bytes, "source_annotation_archive_bytes": args.annotations.stat().st_size,
        "source_archive_sha256": IMAGE_ARCHIVE_SHA256,
        "source_annotation_archive_sha256": ANNOTATION_ARCHIVE_SHA256,
        "sampling_seed": args.seed, "sampling_method": common_info["sampling_method"],
        "sampling_quality_filter": common_info["sampling_quality_filter"],
        "source_joint_schema": SOURCE_KEYPOINT_NAMES_37,
        "source_joint_schema_original_field": dataset["categories"][0]["keypoints"],
        "mapping_version": "spinetrack_official_metainfo_coco_indices_0_16_v1",
        "mapping_evidence": "https://raw.githubusercontent.com/dfki-av/spinepose/main/src/spinepose/metainfo.py; indices 0-16 explicitly named in COCO order",
        "annotation_contract": "full_COCO17_exact_source_coordinates_and_visibility",
        "keypoints_per_person_distribution": {str(k): visible_counts[k] for k in sorted(visible_counts)},
        "visibility_value_distribution": dict(visibility_values),
        "labeled_points_outside_image_bounds": visible_out_of_bounds,
        "person_boxes_outside_image_bounds": bbox_out_of_bounds,
        "annotation_sha256": sha256_bytes(annotation_path.read_bytes()),
        "source_annotation_sample_sha256": sha256_bytes(source_annotation_path.read_bytes()),
        "index": index_path.name, "index_sha256": sha256_bytes(index_path.read_bytes()),
        "sample_image_sha256": sampled_hashes,
        "benchmark_role": "single-person synthetic pose zero-shot benchmark; official COCOeval keypoints; all 17 COCO points mapped; 20 non-COCO source joints retained in source_annotations_37pt.json",
        "known_limitations": [
            "One Unreal Engine generation domain and one synthetic identity per frame; results do not establish general synthetic-to-real transfer.",
            "CC BY-NC 4.0 restricts use to non-commercial contexts and requires attribution; do not redistribute the source archive.",
            "The source archive also contains real subsets; only train-unreal images are extracted and used here.",
            f"The sampled annotations contain {visible_out_of_bounds} labeled COCO points outside image bounds; source values are preserved, and the count is disclosed rather than silently clipped.",
            f"The sampled annotations contain {bbox_out_of_bounds} source person boxes extending outside image bounds; source boxes are preserved.",
        ],
    }
    (output / "dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "prepared", "dataset_id": DATASET_ID, "images": accepted,
                      "sequences": len({r["source_sequence"] for r in sample_rows}),
                      "keypoints_per_person_distribution": manifest["keypoints_per_person_distribution"],
                      "index_sha256": manifest["index_sha256"], "handoff": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
