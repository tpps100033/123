"""Prepare a child-only RGB pose subset from the official SVIRO i3 test split.

Run with Windows Python (``py -3.9 scripts/prepare_sviro_child.py``) on this
machine. Only ZIP metadata and the 53 selected RGB PNGs are transferred.
The official pose and bounding-box ZIPs are cached under source_archives.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import subprocess
import struct
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark_core.data import build_coco_handoff, load_handoff, sha256

DATASET = "sviro_child_i3_test53_coco12"
SOURCE = "https://sviro.kl.dfki.de"
RGB_URL = SOURCE + "/download/bmw-i3/?wpdmdl=394"
POSE_URL = SOURCE + "/download/bmw-i3-6/?wpdmdl=468"
BOX_URL = SOURCE + "/download/bmw-i3-5/?wpdmdl=456"
RGB_SIZE = 2744798619
POSE_SHA256 = "4181070ddc063044b5522c436bcdb70d286191689ea1ddaa86ff61e614531339"
BOX_SHA256 = "0bfd23082683b53f8cf8fab7d2598d23bd172cbdeb1645756e211fa5162d37eb"
COCO_NAMES = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
              "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
              "left_knee", "right_knee", "left_ankle", "right_ankle"]
# SVIRO names mark the start of each bone: upperarm=shoulder,
# lowerarm=elbow, hand=wrist; thigh=hip, calf=knee, foot=ankle.
BONE_MAP = {5: "upperarm_l", 6: "upperarm_r", 7: "lowerarm_l", 8: "lowerarm_r",
            9: "hand_l", 10: "hand_r", 11: "thigh_l", 12: "thigh_r",
            13: "calf_l", 14: "calf_r", 15: "foot_l", 16: "foot_r"}
MAPPING_VERSION = "sviro_i3_child_only_official_person_box_coco12_v1"


class HttpRangeFile(io.RawIOBase):
    """Seekable HTTPS reader that rejects a server ignoring byte ranges."""

    def __init__(self, url, size, block_size=2 * 1024 * 1024):
        self.url, self.size, self.block_size = url, size, block_size
        self.pos, self.block_start, self.block = 0, -1, b""
        self.bytes_downloaded = 0

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        self.pos = offset if whence == io.SEEK_SET else self.pos + offset if whence == io.SEEK_CUR else self.size + offset
        if not 0 <= self.pos <= self.size:
            raise ValueError("ZIP seek outside source archive")
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        result = bytearray()
        while n:
            if not self.block_start <= self.pos < self.block_start + len(self.block):
                start = self.pos // self.block_size * self.block_size
                end = min(start + self.block_size, self.size) - 1
                # curl.exe uses Windows' trusted certificate store. SVIRO's
                # chunked PHP endpoint keeps connections open after a range;
                # read the requested bytes then close curl instead of waiting
                # for EOF (which would stream the complete 2.7 GB archive).
                process = subprocess.Popen(["curl.exe", "--fail", "--silent", "--show-error", "--location",
                                            "--max-time", "180", "--range", f"{start}-{end}", self.url],
                                           stdout=subprocess.PIPE)
                try:
                    data = process.stdout.read(end - start + 1)
                finally:
                    process.terminate()
                    process.wait(timeout=10)
                if len(data) != end - start + 1:
                    raise RuntimeError(f"SVIRO range was truncated: expected {end-start+1}, received {len(data)}")
                self.block_start, self.block = start, data
                self.bytes_downloaded += len(data)
            count = min(n, self.block_start + len(self.block) - self.pos)
            result.extend(self.block[self.pos - self.block_start:self.pos - self.block_start + count])
            self.pos += count
            n -= count
        return bytes(result)


def cached_zip(name, url, expected_hash):
    path = ROOT / "02_datasets" / "source_archives" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        request = urllib.request.Request(url, headers={"User-Agent": "InfantPose-Bench/4 (research)"})
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = response.read()
        temp = path.with_suffix(".part")
        temp.write_bytes(payload)
        if sha256(temp) != expected_hash:
            temp.unlink()
            raise RuntimeError(f"Unexpected official archive hash for {name}")
        temp.replace(path)
    if sha256(path) != expected_hash:
        raise RuntimeError(f"Cached official archive hash changed: {path}")
    return zipfile.ZipFile(path)


def select_children(pose_zip):
    selected = []
    for name in pose_zip.namelist():
        if not name.startswith("i3/test_with_labels/pose_wholeImage/"):
            continue
        match = re.fullmatch(r"(i3_test_imageID_(\d+)_GT_([0-6])_([0-6])_([0-6]))_pose\.json", Path(name).name)
        if match is None:
            raise RuntimeError(f"Unexpected SVIRO pose filename: {name}")
        labels = [int(x) for x in match.groups()[2:]]
        if labels.count(2) == 1 and not any(x in (1, 3) for x in labels):
            selected.append((int(match.group(2)), match.group(1), name, labels))
    selected.sort()
    if len(selected) != 53:
        raise RuntimeError(f"Expected 53 child-only i3 test scenes; found {len(selected)}")
    return selected


def prepare():
    canonical = ROOT / "02_datasets" / "canonical" / DATASET
    if (canonical / "dataset_manifest.json").is_file():
        manifest, rows = load_handoff(canonical)
        if len(rows) != 53 or manifest.get("mapping_version") != MAPPING_VERSION:
            raise RuntimeError("Existing SVIRO child handoff differs; preserve it for review")
        print(json.dumps({"status": "reused_verified_handoff", "samples": 53,
                          "index_sha256": manifest["index_sha256"]}))
        return

    with cached_zip("i3_pose_wholeImage.zip", POSE_URL, POSE_SHA256) as poses, \
         cached_zip("i3_bbox_wholeImage.zip", BOX_URL, BOX_SHA256) as boxes:
        selected = select_children(poses)
        source = HttpRangeFile(RGB_URL, RGB_SIZE)
        images, annotations, paths = [], [], []
        raw = canonical / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(source) as rgb:
            for image_id, (source_id, base, pose_name, labels) in enumerate(selected, 1):
                rgb_name = f"i3/test_with_labels/RGB_wholeImage/{base}.png"
                box_name = f"i3/test_with_labels/boundingBoxes_wholeImage/{base}.txt"
                pose = json.loads(poses.read(pose_name))
                if len(pose) != 1:
                    raise RuntimeError(f"Expected exactly one child pose: {pose_name}")
                person = next(iter(pose.values()))
                expected_position = ("left", "middle", "right")[labels.index(2)]
                if person.get("position") != expected_position:
                    raise RuntimeError(f"Child pose position disagrees with filename: {pose_name}")
                official_boxes = []
                for line in boxes.read(box_name).decode("utf-8-sig").splitlines():
                    values = [float(x) for x in line.split(",")]
                    if len(values) != 5:
                        raise RuntimeError(f"Invalid SVIRO bounding box: {box_name}")
                    if int(values[0]) == 3:
                        official_boxes.append(values[1:])
                if len(official_boxes) != 1:
                    raise RuntimeError(f"Expected one official person bounding box: {box_name}")
                destination = raw / f"{base}.png"
                entry = rgb.getinfo(rgb_name)
                payload = destination.read_bytes() if destination.is_file() else b""
                if len(payload) != entry.file_size or __import__("zlib").crc32(payload) != entry.CRC:
                    payload = rgb.read(rgb_name)
                    temp = destination.with_suffix(".part")
                    temp.write_bytes(payload)
                    temp.replace(destination)
                if payload[:8] != b"\x89PNG\r\n\x1a\n":
                    raise RuntimeError(f"Invalid SVIRO PNG: {rgb_name}")
                width, height = struct.unpack(">II", payload[16:24])
                x0, y0, x1, y1 = official_boxes[0]
                if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
                    raise RuntimeError(f"Official person bounding box outside image: {box_name}")
                keypoints = [0.0] * 51
                count = 0
                for coco_index, bone in BONE_MAP.items():
                    if bone not in person["bones"]:
                        raise RuntimeError(f"Missing SVIRO child joint {bone}: {pose_name}")
                    x, y, visibility = person["bones"][bone]
                    if visibility not in (0, 1, 2):
                        raise RuntimeError(f"Invalid visibility in {pose_name}: {bone}")
                    if visibility and 0 <= x < width and 0 <= y < height:
                        keypoints[3 * coco_index:3 * coco_index + 3] = [x, y, visibility]
                        count += 1
                if count < 2:
                    raise RuntimeError(f"Too few labeled child joints in {pose_name}")
                bbox = [x0, y0, x1 - x0, y1 - y0]
                images.append({"id": image_id, "file_name": destination.name, "width": width, "height": height})
                annotations.append({"id": image_id, "image_id": image_id, "category_id": 1,
                                    "keypoints": keypoints, "num_keypoints": count,
                                    "bbox": bbox, "area": bbox[2] * bbox[3], "iscrowd": 0})
                paths.append({"sviro_image_id": source_id, "pose": pose_name, "rgb": rgb_name,
                              "bbox": box_name, "rgb_sha256": hashlib.sha256(payload).hexdigest()})
                if image_id % 10 == 0:
                    print(f"SVIRO child prepared {image_id}/53; transferred {source.bytes_downloaded/1024**2:.1f} MiB", flush=True)

    annotation_path = ROOT / "02_datasets" / f"person_keypoints_{DATASET}.json"
    annotation_path.write_text(json.dumps({"info": {"description": "SVIRO i3 child-only test subset",
                                                     "source": SOURCE, "license": "CC BY-NC-SA 4.0",
                                                     "mapping": MAPPING_VERSION},
                                           "licenses": [{"id": 1, "name": "CC BY-NC-SA 4.0"}],
                                           "images": images, "annotations": annotations,
                                           "categories": [{"id": 1, "name": "person", "supercategory": "person",
                                                           "keypoints": COCO_NAMES, "skeleton": []}]}, indent=2), encoding="utf-8")
    manifest = build_coco_handoff(DATASET, raw, annotation_path, canonical, 53)
    manifest.update({"mapping_version": MAPPING_VERSION, "source_site": SOURCE,
                     "source_rgb_url": RGB_URL, "source_pose_url": POSE_URL, "source_bbox_url": BOX_URL,
                     "source_pose_sha256": POSE_SHA256, "source_bbox_sha256": BOX_SHA256,
                     "source_selected_files": paths, "source_rgb_full_archive_downloaded": False,
                     "rgb_range_bytes_downloaded": source.bytes_downloaded,
                     "labelled_coco_indices": list(BONE_MAP), "unlabelled_coco_indices": [0, 1, 2, 3, 4],
                     "bbox_area_protocol": "official SVIRO class-3 person bbox width*height",
                     "selection": "i3 test_with_labels; exactly one class-2 child, no class-1 infant or class-3 adult"})
    (canonical / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"status": "prepared", "samples": 53,
                      "transferred_mib": round(source.bytes_downloaded / 1024**2, 1),
                      "index_sha256": manifest["index_sha256"]}))


if __name__ == "__main__":
    prepare()
