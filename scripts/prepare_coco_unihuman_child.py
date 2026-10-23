"""Build a non-overlapping, single-person apparent-child COCO-17 evaluation set.

The images are COCO val2017 images; COCO-UniHuman supplies manual apparent-age
labels. This derivative cohort is not an independent image-source benchmark.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
DATASET = "coco_unihuman_child_val134"
ANNOTATION_SOURCE = "02_datasets/source_archives/coco_unihuman_val_v2.json"
ANNOTATION_URL = "https://drive.google.com/uc?export=download&id=1G2E7J_bW7Sdnq8VSlo3ddxfZKUOY5ORh"
ANNOTATION_SHA256 = "7a27354104d014a4aa8445eecc3e9a0a918862b1e7ffb940682b642f4123fdef"
EXPECTED_IMAGES = 134
MIN_APPARENT_AGE = 3
MAX_APPARENT_AGE = 17


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(url: str, destination: Path) -> None:
    if destination.is_file() and destination.stat().st_size:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "InfantPose research data preparation"})
    temporary = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
                while block := response.read(1024 * 1024):
                    output.write(block)
            temporary.replace(destination)
            return
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def select_children(source: dict, excluded_image_ids: set[int]) -> list[tuple[dict, dict]]:
    images = {int(image["id"]): image for image in source["images"]}
    by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in source["annotations"]:
        age = annotation.get("age")
        if (annotation.get("valid", {}).get("age") != 1
                or not isinstance(age, int)
                or not MIN_APPARENT_AGE <= age <= MAX_APPARENT_AGE
                or annotation.get("valid", {}).get("keypoints_2d") != 1
                or len(annotation.get("keypoints", [])) != 51
                or int(annotation.get("num_keypoints", 0)) < 1
                or int(annotation.get("iscrowd", 0))):
            continue
        image_id = int(annotation["image_id"])
        if image_id not in excluded_image_ids:
            by_image[image_id].append(annotation)

    # One target person per source image keeps the benchmark denominator at
    # unique images and avoids counting two children in one photograph twice.
    selected = []
    for image_id, annotations in sorted(by_image.items()):
        image = images[image_id]
        annotation = max(annotations, key=lambda item: (float(item.get("area", 0)), -int(item["id"])))
        selected.append((image, annotation))
    if len(selected) != EXPECTED_IMAGES:
        raise RuntimeError(f"Expected {EXPECTED_IMAGES} non-overlapping child images, found {len(selected)}")
    return selected


def crop_box(image: dict, annotation: dict) -> tuple[int, int, int, int]:
    width, height = int(image["width"]), int(image["height"])
    box_x, box_y, box_w, box_h = map(float, annotation["bbox"])
    points = annotation["keypoints"]
    labeled = [(float(points[i]), float(points[i + 1]))
               for i in range(0, 51, 3) if int(points[i + 2]) > 0]
    xs = [box_x, box_x + box_w] + [point[0] for point in labeled]
    ys = [box_y, box_y + box_h] + [point[1] for point in labeled]
    pad = max(12.0, 0.12 * max(box_w, box_h))
    left = max(0, math.floor(min(xs) - pad))
    top = max(0, math.floor(min(ys) - pad))
    right = min(width, math.ceil(max(xs) + pad))
    bottom = min(height, math.ceil(max(ys) + pad))
    if right <= left or bottom <= top:
        raise RuntimeError(f"Invalid crop for source image {image['id']} / annotation {annotation['id']}")
    return left, top, right, bottom


def main() -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Install the isolated data-preparation environment from requirements/dataset-prep.txt") from exc

    source_path = ROOT / ANNOTATION_SOURCE
    fetch(ANNOTATION_URL, source_path)
    digest = sha256(source_path)
    if digest != ANNOTATION_SHA256:
        raise RuntimeError(f"COCO-UniHuman validation annotation SHA-256 mismatch: {digest}")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    coco_path = ROOT / "02_datasets/person_keypoints_val2017_singleperson.json"
    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    excluded_ids = {int(image["id"]) for image in coco["images"]}
    coco_raw = ROOT / "02_datasets/canonical/coco_singleperson_val2017/raw"
    supported_images = {".jpg", ".jpeg", ".png", ".bmp"}
    coco_image_files = [path for path in coco_raw.rglob("*") if path.is_file() and path.suffix.lower() in supported_images]
    if len(coco_image_files) != len(excluded_ids):
        raise RuntimeError(f"Expected {len(excluded_ids)} local COCO comparison images for byte-level deduplication, found {len(coco_image_files)}")
    excluded_content_hashes = {sha256(path) for path in coco_image_files}
    selected = select_children(source, excluded_ids)

    dataset_root = ROOT / "02_datasets/canonical" / DATASET
    raw_root = dataset_root / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    images_out, annotations_out, source_rows = [], [], []
    selected_content_hashes = set()
    for index, (source_image, source_annotation) in enumerate(selected, start=1):
        source_url = source_image.get("coco_url", "")
        parsed_url = urlsplit(source_url)
        filename = Path(source_image["file_name"]).name
        if (parsed_url.hostname != "images.cocodataset.org"
                or parsed_url.path != f"/val2017/{filename}"
                or parsed_url.scheme != "http"):
            raise RuntimeError(f"Unexpected COCO source URL for image {source_image['id']}: {source_url}")
        source_path = ROOT / "02_datasets/source_archives/coco_unihuman_images" / filename
        fetch(source_url, source_path)
        source_digest = sha256(source_path)
        if source_digest in excluded_content_hashes:
            raise RuntimeError(f"Selected child image is byte-identical to a COCO comparison image: {source_image['file_name']}")
        if source_digest in selected_content_hashes:
            raise RuntimeError(f"Duplicate child source image content detected: {source_image['file_name']}")
        selected_content_hashes.add(source_digest)
        try:
            with Image.open(source_path) as image:
                image.load()
                if image.size != (int(source_image["width"]), int(source_image["height"])):
                    raise RuntimeError(f"Source image dimensions differ from annotation: {source_image['file_name']}")
                bounds = crop_box(source_image, source_annotation)
                crop = image.convert("RGB").crop(bounds)
                filename = f"child_{index:04d}.jpg"
                crop_path = raw_root / filename
                crop.save(crop_path, format="JPEG", quality=95, subsampling=0)
        except Exception as exc:
            raise RuntimeError(f"Could not prepare {source_image['file_name']}: {exc}") from exc

        left, top, right, bottom = bounds
        new_image_id = 900000000 + index
        keypoints = list(source_annotation["keypoints"])
        for point_index in range(0, 51, 3):
            if int(keypoints[point_index + 2]) > 0:
                keypoints[point_index] = float(keypoints[point_index]) - left
                keypoints[point_index + 1] = float(keypoints[point_index + 1]) - top
            else:
                keypoints[point_index] = 0
                keypoints[point_index + 1] = 0
        source_bbox = list(map(float, source_annotation["bbox"]))
        bbox = [source_bbox[0] - left, source_bbox[1] - top, source_bbox[2], source_bbox[3]]
        annotation = dict(source_annotation)
        annotation.update({"id": index, "image_id": new_image_id, "keypoints": keypoints,
                           "bbox": bbox, "area": float(source_annotation.get("area", source_bbox[2] * source_bbox[3])),
                           "iscrowd": 0, "source_annotation_id": int(source_annotation["id"]),
                           "source_image_id": int(source_image["id"])})
        images_out.append({"id": new_image_id, "file_name": filename,
                           "width": right - left, "height": bottom - top,
                           "source_image_id": int(source_image["id"]),
                           "source_file_name": source_image["file_name"]})
        annotations_out.append(annotation)
        source_rows.append({"dataset_image_id": new_image_id,
                            "source_image_id": int(source_image["id"]),
                            "source_annotation_id": int(source_annotation["id"]),
                            "apparent_age_years": int(source_annotation["age"]),
                            "source_file_name": source_image["file_name"],
                            "source_image_sha256": source_digest,
                            "crop_file_name": filename,
                            "crop_sha256": sha256(crop_path)})
        if index % 25 == 0 or index == EXPECTED_IMAGES:
            print(f"Prepared {index}/{EXPECTED_IMAGES} child crops", flush=True)

    output = {
        "info": {"description": "COCO-UniHuman val apparent-age child cohort, cropped single-person images",
                 "source": "COCO-UniHuman v2 validation annotations + COCO val2017 RGB images",
                 "apparent_age_range_years": [MIN_APPARENT_AGE, MAX_APPARENT_AGE],
                 "keypoint_protocol": "COCO-17 identity mapping; coordinates translated into person crop",
                 "overlap_policy": "exclude COCO project image IDs and byte-identical source images; one source image hash per child sample",
                 "image_protocol": "one largest-area eligible child instance per unique source image; 12% context padding",
                 "source_annotation_sha256": digest},
        "licenses": source.get("licenses", []),
        "images": images_out,
        "annotations": annotations_out,
        "categories": source.get("categories", []),
    }
    annotation_path = ROOT / "02_datasets/person_keypoints_coco_unihuman_child_val134.json"
    annotation_path.write_text(json.dumps(output, separators=(",", ":")), encoding="utf-8")
    (dataset_root / "selection_manifest.json").write_text(
        json.dumps({"dataset_id": DATASET, "source_annotation": ANNOTATION_SOURCE,
                    "source_annotation_sha256": digest, "source_image_count": EXPECTED_IMAGES,
                    "selected_person_count": len(annotations_out), "excluded_coco_project_image_count": len(excluded_ids),
                    "apparent_age_range_years": [MIN_APPARENT_AGE, MAX_APPARENT_AGE],
                    "images": source_rows}, indent=2), encoding="utf-8")
    sys.path.insert(0, str(ROOT))
    from benchmark_core.data import build_coco_handoff
    config_path = ROOT / "config/local.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected = config["datasets"][DATASET]["expected_samples"]
    manifest = build_coco_handoff(DATASET, raw_root, annotation_path, dataset_root, expected)
    manifest.update({"source_dataset": "COCO-UniHuman v2 / COCO val2017",
                     "source_annotation_sha256": digest,
                     "apparent_age_years": [MIN_APPARENT_AGE, MAX_APPARENT_AGE],
                     "source_images_unique": EXPECTED_IMAGES,
                     "source_image_hashes_unique": len(selected_content_hashes),
                     "source_coco_overlap_images_excluded": len(excluded_ids),
                     "image_protocol": "single-person crops with 12% context padding; one largest eligible child per source image"})
    (dataset_root / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "dataset_id": DATASET, "images": len(images_out),
                      "annotations": len(annotations_out), "handoff_sha256": manifest["index_sha256"],
                      "annotation_file": str(annotation_path)}, indent=2))


if __name__ == "__main__":
    main()
