"""Build the 10-vehicle SVIRO child-only official-test COCO-17 handoff.

Only the selected RGB members are downloaded. The 53-image i3 handoff stays
unchanged as a historical subset; this dataset combines all 10 official car
test splits using the same single-child selection and COCO-12 mapping.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark_core.data import build_coco_handoff, load_handoff, sha256
from scripts.prepare_sviro_child import HttpRangeFile

SOURCE = "https://sviro.kl.dfki.de"
DATASET = "sviro_children_allcars_test710_coco12"
EXPECTED_TOTAL = 710
ARCHIVES = ROOT / "02_datasets" / "source_archives"
CANONICAL = ROOT / "02_datasets" / "canonical" / DATASET
ANNOTATION_PATH = ROOT / "02_datasets" / f"person_keypoints_{DATASET}.json"
COCO_NAMES = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
              "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
              "left_knee", "right_knee", "left_ankle", "right_ankle"]
BONE_MAP = {5: "upperarm_l", 6: "upperarm_r", 7: "lowerarm_l", 8: "lowerarm_r",
            9: "hand_l", 10: "hand_r", 11: "thigh_l", 12: "thigh_r",
            13: "calf_l", 14: "calf_r", 15: "foot_l", 16: "foot_r"}
MAPPING_VERSION = "sviro_allcars_child_only_official_person_box_coco12_v1"
LABEL_PATTERN = re.compile(r"(.*_imageID_(\d+)_GT_([0-6])_([0-6])_([0-6]))_pose\.json$")

# (SVIRO folder, RGB slug/suffix/id, pose slug/suffix/id, box slug/suffix/id, count)
VEHICLES = (
    ("i3", "bmw-i3", None, 394, "bmw-i3", 6, 468, "bmw-i3", 5, 456, 53),
    ("tucson", "hyundai-tucson", 2, 397, "hyundai-tucson", 7, 472, "hyundai-tucson", 6, 460, 83),
    ("x5", "bmw-x5", 2, 395, "bmw-x5", 7, 469, "bmw-x5", 6, 457, 57),
    ("zoe", "renault-zoe", 2, 399, "renault-zoe", 7, 475, "renault-zoe", 6, 463, 69),
    ("gsf", "lexus-gs-f", None, 398, "lexus-gs-f", 6, 473, "lexus-gs-f", 5, 461, 82),
    ("hilux", "toyota-hilux", None, 401, "toyota-hilux", 6, 477, "toyota-hilux", 5, 465, 77),
    ("model3", "tesla-model-3", 2, 400, "tesla-model-3", 7, 476, "tesla-model-3", 6, 464, 65),
    ("tiguan", "vw-tiguan", 2, 402, "vw-tiguan", 7, 478, "vw-tiguan", 6, 466, 74),
    ("aclass", "mercedes-class-a", None, 385, "mercedes-class-a", 6, 474, "mercedes-class-a", 5, 462, 74),
    ("escape", "ford-escape", None, 396, "ford-escape", 6, 471, "ford-escape", 5, 459, 76),
)


def wpdm_url(slug: str, suffix: int, download_id: int) -> str:
    return f"{SOURCE}/download/{slug}-{suffix}/?wpdmdl={download_id}"


def cached_annotation(name: str, url: str) -> Path:
    path = ARCHIVES / name
    if path.is_file() and zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None:
                raise RuntimeError(f"Corrupt cached SVIRO annotation archive: {path}")
        return path
    if path.exists():
        raise RuntimeError(f"Incomplete cache exists; inspect it before replacing: {path}")
    ARCHIVES.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    subprocess.run(["curl.exe", "--fail", "--silent", "--show-error", "--location",
                    "--range", "0-9999999", "--max-time", "180", url, "--output", str(temporary)],
                   check=True, timeout=190)
    if not zipfile.is_zipfile(temporary):
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"SVIRO endpoint did not return a complete annotation ZIP: {url}")
    with zipfile.ZipFile(temporary) as archive:
        bad_entry = archive.testzip()
        if bad_entry:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"Corrupt entry in {url}: {bad_entry}")
    temporary.replace(path)
    return path


def discover_archive_size(url: str) -> int:
    """Read Content-Range from one byte, then stop the oversized WPDM stream."""
    with tempfile.NamedTemporaryFile(prefix="sviro_headers_", suffix=".txt", delete=False) as file:
        header_path = Path(file.name)
    process = subprocess.Popen(["curl.exe", "--fail", "--silent", "--show-error", "--location",
                                "--max-time", "45", "--range", "0-1",
                                "--dump-header", str(header_path), url],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        first_bytes = process.stdout.read(2)
    finally:
        process.terminate()
        process.wait(timeout=10)
    headers = header_path.read_text(encoding="latin-1", errors="replace")
    header_path.unlink(missing_ok=True)
    matches = re.findall(r"Content-Range:\s*bytes\s+\d+-\d+/(\d+)", headers, flags=re.IGNORECASE)
    if len(first_bytes) != 2 or not matches:
        raise RuntimeError(f"Could not determine SVIRO ZIP byte length from Range response: {url}")
    return int(matches[-1])


def selected_scenes(pose_zip: zipfile.ZipFile, vehicle: str) -> list[tuple[int, str, str, list[int]]]:
    prefix = f"{vehicle}/test_with_labels/pose_wholeImage/"
    names = [name for name in pose_zip.namelist() if name.startswith(prefix)]
    if len(names) != 500:
        raise RuntimeError(f"Expected 500 official test poses for {vehicle}, found {len(names)}")
    selected = []
    for name in names:
        match = LABEL_PATTERN.fullmatch(Path(name).name)
        if not match:
            raise RuntimeError(f"Unexpected SVIRO pose filename: {name}")
        labels = [int(value) for value in match.groups()[2:]]
        if labels.count(2) == 1 and not any(value in (1, 3) for value in labels):
            selected.append((int(match.group(2)), match.group(1), name, labels))
    selected.sort()
    return selected


def load_box(boxes: zipfile.ZipFile, vehicle: str, base: str) -> list[float]:
    name = f"{vehicle}/test_with_labels/boundingBoxes_wholeImage/{base}.txt"
    values = []
    for line in boxes.read(name).decode("utf-8-sig").splitlines():
        fields = [float(value) for value in line.split(",")]
        if len(fields) != 5:
            raise RuntimeError(f"Invalid SVIRO bounding box line in {name}")
        if int(fields[0]) == 3:
            values.append(fields[1:])
    if len(values) != 1:
        raise RuntimeError(f"Expected exactly one official person box in {name}, found {len(values)}")
    return values[0]


def prepare() -> None:
    if (CANONICAL / "dataset_manifest.json").is_file():
        manifest, rows = load_handoff(CANONICAL)
        if len(rows) == EXPECTED_TOTAL and manifest.get("mapping_version") == MAPPING_VERSION:
            print(json.dumps({"status": "reused_verified_handoff", "samples": len(rows),
                              "index_sha256": manifest["index_sha256"]}))
            return
        raise RuntimeError("Existing all-car child handoff differs; preserve it for review")

    raw = CANONICAL / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    images, annotations, selected_files = [], [], []
    vehicle_counts, source_archives = {}, []
    total_rgb_bytes = 0
    next_image_id = 1
    for (vehicle, rgb_slug, rgb_suffix, rgb_id, pose_slug, pose_suffix, pose_id,
         box_slug, box_suffix, box_id, expected_vehicle_count) in VEHICLES:
        pose_url = wpdm_url(pose_slug, pose_suffix, pose_id)
        box_url = wpdm_url(box_slug, box_suffix, box_id)
        pose_path = cached_annotation(f"{vehicle}_pose_wholeImage.zip", pose_url)
        box_path = cached_annotation(f"{vehicle}_bbox_wholeImage.zip", box_url)
        with zipfile.ZipFile(pose_path) as poses, zipfile.ZipFile(box_path) as boxes:
            pose_names = poses.namelist()
            box_names = boxes.namelist()
            if not any(f"{vehicle}/test_with_labels/pose_wholeImage/" in n for n in pose_names):
                raise RuntimeError(f"Pose archive is not pose labels for {vehicle}: {pose_path.name}")
            if not any(f"{vehicle}/test_with_labels/boundingBoxes_wholeImage/" in n for n in box_names):
                raise RuntimeError(f"Box archive is not official box labels for {vehicle}: {box_path.name}")
            chosen = selected_scenes(poses, vehicle)
            if len(chosen) != expected_vehicle_count:
                raise RuntimeError(f"{vehicle}: inventory expected {expected_vehicle_count}, found {len(chosen)}")

            rgb_url = f"{SOURCE}/download/{rgb_slug}{'-' + str(rgb_suffix) if rgb_suffix else ''}/?wpdmdl={rgb_id}"
            rgb_size = discover_archive_size(rgb_url)
            # Child scenes are distributed across the sorted ZIP members. Larger
            # ranges reduce per-request latency while still avoiding full archives.
            source = HttpRangeFile(rgb_url, rgb_size, block_size=4 * 1024 * 1024)
            rgb_prefix = f"{vehicle}/test_with_labels/RGB_wholeImage/"
            # Load the remote central directory once per vehicle, not once per image.
            with zipfile.ZipFile(source) as rgb:
                for source_id, base, pose_name, labels in chosen:
                    pose = json.loads(poses.read(pose_name))
                    if len(pose) != 1:
                        raise RuntimeError(f"Expected one child pose in {pose_name}")
                    person = next(iter(pose.values()))
                    expected_position = ("left", "middle", "right")[labels.index(2)]
                    if person.get("position") != expected_position:
                        raise RuntimeError(f"Pose position disagrees with labels in {pose_name}")
                    official_box = load_box(boxes, vehicle, base)
                    rgb_name = f"{rgb_prefix}{base}.png"
                    info = rgb.getinfo(rgb_name)
                    destination = raw / f"{vehicle}_{Path(base).name}.png"
                    if destination.is_file():
                        import zlib
                        payload = destination.read_bytes()
                        if len(payload) != info.file_size or zlib.crc32(payload) != info.CRC:
                            destination.unlink()
                    if not destination.is_file() and vehicle == "i3":
                        previous = (ROOT / "02_datasets" / "canonical" /
                                    "sviro_child_i3_test53_coco12" / "raw" / f"{base}.png")
                        if previous.is_file():
                            import zlib
                            cached = previous.read_bytes()
                            if len(cached) == info.file_size and zlib.crc32(cached) == info.CRC:
                                shutil.copy2(previous, destination)
                    if not destination.is_file():
                        payload = rgb.read(rgb_name)
                        import zlib
                        if len(payload) != info.file_size or zlib.crc32(payload) != info.CRC:
                            raise RuntimeError(f"SVIRO RGB CRC/size mismatch: {rgb_name}")
                        temp = destination.with_suffix(".png.part")
                        temp.write_bytes(payload)
                        temp.replace(destination)
                    else:
                        payload = destination.read_bytes()
                    if payload[:8] != b"\x89PNG\r\n\x1a\n":
                        raise RuntimeError(f"Invalid SVIRO PNG header: {rgb_name}")
                    width, height = struct.unpack(">II", payload[16:24])
                    x0, y0, x1, y1 = official_box
                    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
                        raise RuntimeError(f"Official person box outside image: {rgb_name}")
                    keypoints = [0.0] * 51
                    count = 0
                    for coco_index, bone in BONE_MAP.items():
                        if bone not in person["bones"]:
                            raise RuntimeError(f"Missing SVIRO joint {bone}: {pose_name}")
                        x, y, visibility = person["bones"][bone]
                        if visibility not in (0, 1, 2):
                            raise RuntimeError(f"Invalid visibility for {bone} in {pose_name}")
                        if visibility and 0 <= x < width and 0 <= y < height:
                            keypoints[3 * coco_index:3 * coco_index + 3] = [x, y, visibility]
                            count += 1
                    if count < 2:
                        raise RuntimeError(f"Too few visible labelled joints in {pose_name}")
                    bbox = [x0, y0, x1 - x0, y1 - y0]
                    images.append({"id": next_image_id, "file_name": destination.name,
                                   "width": width, "height": height})
                    annotations.append({"id": next_image_id, "image_id": next_image_id,
                                        "category_id": 1, "keypoints": keypoints,
                                        "num_keypoints": count, "bbox": bbox,
                                        "area": bbox[2] * bbox[3], "iscrowd": 0})
                    selected_files.append({"vehicle": vehicle, "sviro_image_id": source_id,
                                           "pose": pose_name, "rgb": rgb_name,
                                           "bbox": f"{vehicle}/test_with_labels/boundingBoxes_wholeImage/{base}.txt",
                                           "rgb_sha256": hashlib.sha256(payload).hexdigest(),
                                           "rgb_bytes": len(payload)})
                    next_image_id += 1
                    if (next_image_id - 1) % 10 == 0:
                        print(f"SVIRO progress: {next_image_id - 1}/{EXPECTED_TOTAL}; "
                              f"range bytes this car {source.bytes_downloaded / 1024**2:.1f} MiB",
                              flush=True)
            vehicle_counts[vehicle] = len(chosen)
            total_rgb_bytes += source.bytes_downloaded
            source_archives.append({"vehicle": vehicle, "rgb_url": rgb_url, "rgb_size_bytes": rgb_size,
                                    "pose_url": pose_url, "pose_sha256": sha256(pose_path),
                                    "bbox_url": box_url, "bbox_sha256": sha256(box_path),
                                    "rgb_range_bytes_downloaded": source.bytes_downloaded})
        print(f"SVIRO {vehicle}: {vehicle_counts[vehicle]}/{expected_vehicle_count} child scenes; "
              f"total RGB range bytes {total_rgb_bytes / 1024**2:.1f} MiB", flush=True)

    if next_image_id - 1 != EXPECTED_TOTAL:
        raise RuntimeError(f"Expected {EXPECTED_TOTAL} images, found {next_image_id - 1}")
    ANNOTATION_PATH.write_text(json.dumps({
        "info": {"description": "SVIRO ten-vehicle child-only official test subset",
                 "source": SOURCE, "license": "CC BY-NC-SA 4.0", "mapping": MAPPING_VERSION},
        "licenses": [{"id": 1, "name": "CC BY-NC-SA 4.0"}],
        "images": images, "annotations": annotations,
        "categories": [{"id": 1, "name": "person", "supercategory": "person",
                        "keypoints": COCO_NAMES, "skeleton": []}],
    }, indent=2), encoding="utf-8")
    manifest = build_coco_handoff(DATASET, raw, ANNOTATION_PATH, CANONICAL, EXPECTED_TOTAL)
    manifest.update({
        "mapping_version": MAPPING_VERSION, "source_site": SOURCE,
        "selection": "all ten official test splits; exactly one class-2 child and no class-1 infant or class-3 adult",
        "official_test_scenes_per_vehicle": 500, "per_vehicle_child_scene_counts": vehicle_counts,
        "source_archives": source_archives, "source_selected_files": selected_files,
        "rgb_full_archives_downloaded": False, "rgb_range_bytes_downloaded_total": total_rgb_bytes,
        "labelled_coco_indices": list(BONE_MAP), "unlabelled_coco_indices": [0, 1, 2, 3, 4],
        "bbox_area_protocol": "official SVIRO class-3 person bbox width*height",
        "independence_note": "Frames are stratified by vehicle test split; underlying synthetic human models can repeat across vehicle domains and are not claimed to be 710 independent children.",
    })
    (CANONICAL / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"status": "prepared", "dataset_id": DATASET, "samples": EXPECTED_TOTAL,
                      "per_vehicle": vehicle_counts,
                      "transferred_mib": round(total_rgb_bytes / 1024**2, 1),
                      "index_sha256": manifest["index_sha256"]}, indent=2))


if __name__ == "__main__":
    prepare()
