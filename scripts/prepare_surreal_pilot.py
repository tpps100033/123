"""Create a local COCO-shaped, audited 12-of-17 SURREAL pilot handoff."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import urllib.parse
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "surreal_cmu_test_run1_pilot500"
PLAN_ROOT = ROOT / "02_datasets/canonical" / DATASET_ID
SOURCE_ROOT = ROOT / "02_datasets/source_archives/surreal_cmu_test_run1_pilot500"
COCO_TO_SMPL24 = {
    5: 16, 6: 17, 7: 18, 8: 19, 9: 20, 10: 21,
    11: 1, 12: 2, 13: 4, 14: 5, 15: 7, 16: 8,
}
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
    [6, 12], [7, 13], [6, 7], [6, 8], [8, 10], [7, 9], [9, 11],
    [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7],
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_path(url: str) -> Path:
    path = urllib.parse.urlparse(url).path
    marker = "/SURREAL/"
    if marker not in path:
        raise ValueError(f"Unexpected source URL path: {url}")
    return SOURCE_ROOT / Path(path.split(marker, 1)[1])


def read_segmentation(path: Path, frame_index: int, height: int, width: int) -> np.ndarray:
    key = f"segm_{frame_index + 1}"
    mat = loadmat(path, variable_names=[key])
    if key not in mat:
        raise KeyError(f"{path} is missing expected variable {key}")
    segm = np.asarray(mat[key])
    if segm.shape != (height, width):
        raise ValueError(f"Unexpected {key} shape {segm.shape} in {path}; expected {(height, width)}")
    return segm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PLAN_ROOT / "acquisition_plan.json")
    parser.add_argument("--frames-index", type=Path, default=PLAN_ROOT / "extracted_frames.csv")
    parser.add_argument("--coordinate-offset", type=int, choices=(0, 1), default=0,
                        help="Subtract this offset from SURREAL pixel coordinates; default follows official projection values.")
    parser.add_argument("--visibility-radius", type=float, default=2.0,
                        help="Radius in pixels for matching a projected joint to the person segmentation.")
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if plan.get("dataset_id") != DATASET_ID:
        raise ValueError(f"Unexpected plan dataset_id: {plan.get('dataset_id')}")
    with args.frames_index.open(encoding="utf-8", newline="") as stream:
        extracted = list(csv.DictReader(stream))
    if len(extracted) != plan["target_images"]:
        raise ValueError(f"Expected {plan['target_images']} extracted frames, got {len(extracted)}")
    clips = {(item["sequence_folder"], item["clip_stem"]): item for item in plan["clips"]}
    extracted_by_id = {int(row["image_id"]): row for row in extracted}
    if len(extracted_by_id) != len(extracted):
        raise ValueError("Duplicate image_id in extracted frame index")

    image_root = PLAN_ROOT / "images"
    images, annotations, index_rows = [], [], []
    source_mat_hashes: dict[str, str] = {}
    source_segment_hashes: dict[str, str] = {}
    info_cache: dict[Path, dict] = {}
    source_count = 0
    visible_count = 0
    occluded_count = 0
    unsupported_count = 0

    for image_id in range(1, len(extracted) + 1):
        row = extracted_by_id[image_id]
        key = (row["sequence_folder"], row["clip_stem"])
        clip = clips[key]
        info_path = source_path(clip["info_url"])
        segm_path = source_path(clip["segm_url"])
        image_path = PLAN_ROOT / Path(row["image_path"])
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        if sha256(image_path) != row["image_sha256"]:
            raise ValueError(f"Extracted image SHA-256 changed: {image_path}")
        info_key = info_path.relative_to(SOURCE_ROOT).as_posix()
        segm_key = segm_path.relative_to(SOURCE_ROOT).as_posix()
        if info_key not in source_mat_hashes:
            source_mat_hashes[info_key] = sha256(info_path)
        if segm_key not in source_segment_hashes:
            source_segment_hashes[segm_key] = sha256(segm_path)
        if info_path not in info_cache:
            info_cache[info_path] = loadmat(info_path, variable_names=["joints2D", "sequence", "clipNo", "stride"])
        info = info_cache[info_path]
        joints = np.asarray(info["joints2D"], dtype=np.float64)
        frame_index = int(row["frame_index_zero_based"])
        if joints.ndim != 3 or joints.shape[0:2] != (2, 24):
            raise ValueError(f"Unexpected joints2D shape {joints.shape} in {info_path}")
        if frame_index >= joints.shape[2]:
            raise ValueError(f"Frame {frame_index} exceeds joints2D length {joints.shape[2]} in {info_path}")

        height, width = int(row["height"]), int(row["width"])
        segm = read_segmentation(segm_path, frame_index, height, width)
        person_mask = segm > 0
        ys, xs = np.nonzero(person_mask)
        if len(xs) == 0:
            raise ValueError(f"Empty person segmentation for {row['sample_id']}")
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        bbox = [float(x_min), float(y_min), float(x_max - x_min + 1), float(y_max - y_min + 1)]
        distance = distance_transform_edt(~person_mask)

        gt_keypoints = []
        valid_count = 0
        for coco_index in range(17):
            if coco_index not in COCO_TO_SMPL24:
                gt_keypoints.extend((0.0, 0.0, 0))
                unsupported_count += 1
                continue
            smpl_index = COCO_TO_SMPL24[coco_index]
            x = float(joints[0, smpl_index, frame_index]) - args.coordinate_offset
            y = float(joints[1, smpl_index, frame_index]) - args.coordinate_offset
            if not (math.isfinite(x) and math.isfinite(y)):
                gt_keypoints.extend((0.0, 0.0, 0))
                continue
            if not (0 <= x < width and 0 <= y < height):
                gt_keypoints.extend((0.0, 0.0, 0))
                continue
            px = min(max(int(round(x)), 0), width - 1)
            py = min(max(int(round(y)), 0), height - 1)
            visibility = 2 if float(distance[py, px]) <= args.visibility_radius else 1
            gt_keypoints.extend((x, y, visibility))
            valid_count += 1
            if visibility == 2:
                visible_count += 1
            else:
                occluded_count += 1
        if valid_count == 0:
            raise ValueError(f"No in-frame shared SMPL/COCO joints in {row['sample_id']}")
        source_count += 1

        annotations.append({
            "id": image_id,
            "image_id": image_id,
            "category_id": 1,
            "keypoints": gt_keypoints,
            "num_keypoints": valid_count,
            "bbox": bbox,
            "area": bbox[2] * bbox[3],
            "iscrowd": 0,
            "source_sequence": row["sequence_folder"],
            "source_clip": row["clip_stem"],
            "source_frame_index_zero_based": frame_index,
            "visibility_policy": "visible iff projected joint is within configured radius of SURREAL foreground mask; otherwise labeled as occluded",
        })
        images.append({
            "id": image_id,
            "file_name": row["image_path"],
            "width": width,
            "height": height,
            "license": 1,
            "source_sequence": row["sequence_folder"],
            "source_clip": row["clip_stem"],
            "source_frame_index_zero_based": frame_index,
            "sha256": row["image_sha256"],
        })
        flat_xy = [[gt_keypoints[i], gt_keypoints[i + 1]] for i in range(0, 51, 3)]
        flat_v = [gt_keypoints[i] for i in range(2, 51, 3)]
        index_rows.append({
            "sample_id": row["sample_id"],
            "dataset_id": DATASET_ID,
            "image_id": image_id,
            "image_path": row["image_path"],
            "width": width,
            "height": height,
            "bbox_xywh": json.dumps(bbox),
            "gt_xy": json.dumps(flat_xy),
            "visibility": json.dumps(flat_v),
            "source_sequence": row["sequence_folder"],
            "source_clip": row["clip_stem"],
            "source_frame_index_zero_based": frame_index,
            "image_sha256": row["image_sha256"],
        })

    annotations_path = PLAN_ROOT / "annotations.json"
    coco = {
        "info": {
            "description": "SURREAL CMU test/run1 pilot; 12 semantically shared SMPL joints mapped into COCO-17 slots",
            "year": 2017,
            "source": "https://www.di.ens.fr/willow/research/surreal/data/",
            "annotation_policy": "SURREAL source has projected joints but no COCO facial landmarks; only the 12 shared body joints are scored",
        },
        "licenses": [{"id": 1, "name": "SURREAL Code and Data License (non-commercial scientific research)",
                      "url": "https://www.di.ens.fr/willow/research/surreal/data/license.html"}],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "person", "supercategory": "person",
                        "keypoints": KEYPOINT_NAMES, "skeleton": SKELETON}],
    }
    annotations_path.write_text(json.dumps(coco, ensure_ascii=False), encoding="utf-8")
    index_path = PLAN_ROOT / "sample_index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(index_rows[0]))
        writer.writeheader()
        writer.writerows(index_rows)

    manifest = {
        "schema_version": "v4.dataset-handoff",
        "dataset_id": DATASET_ID,
        "sample_count": len(images),
        "annotation_count": len(annotations),
        "scored_keypoint_slots": 12,
        "unsupported_keypoint_slots": [0, 1, 2, 3, 4],
        "annotation_contract": "COCO-17-shaped JSON; only SMPL24-to-COCO17 shared body joints are scored; face landmarks remain unlabeled",
        "mapping_version": "surreal_smpl24_projected_2d_to_coco17_shared12_v1",
        "person_selection": "single SMPL subject in each SURREAL clip",
        "sampling": {
            "split": plan["split"], "seed": plan["seed"], "sequence_folders": plan["sequence_folders"],
            "frames_per_clip": plan["frames_per_clip"], "sample_count": len(images),
        },
        "coordinate_offset_subtracted": args.coordinate_offset,
        "visibility_policy": "v=2 when source projected joint is within configured radius of foreground segmentation; v=1 otherwise when in-frame; v=0 if outside image; source has no explicit COCO visibility labels",
        "visibility_radius_px": args.visibility_radius,
        "bbox_source": "tight bbox of nonzero SURREAL person-part segmentation",
        "license": "SURREAL Code and Data License; non-commercial scientific research only; no redistribution",
        "source_url": plan["source_url"],
        "source_indexes_sha256": plan["official_index_sha256"],
        "source_info_sha256": source_mat_hashes,
        "source_segmentation_sha256": source_segment_hashes,
        "image_index_sha256": sha256(index_path),
        "annotation_sha256": sha256(annotations_path),
        "image_sha256_manifest_sha256": hashlib.sha256("\n".join(row["image_sha256"] for row in index_rows).encode()).hexdigest(),
        "visibility_counts": {"v2_mask_aligned": visible_count, "v1_in_frame_not_mask_aligned": occluded_count,
                              "v0_unsupported_or_out_of_frame": unsupported_count},
        "evaluation_warning": "12-joint partial COCO-OKS; not directly comparable to full COCO-17 AP. Visibility is inferred from segmentation, not provided by SURREAL.",
    }
    manifest_path = PLAN_ROOT / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "prepared", "dataset_id": DATASET_ID,
                      "sample_count": len(images), "annotations": len(annotations),
                      "scored_joints": 12, "v2": visible_count, "v1": occluded_count,
                      "manifest": str(manifest_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
