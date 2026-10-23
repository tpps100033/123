"""Inspect and prepare the public BabyPose v2 ZIP with HTTP range reads.

The source archive is about 3 GB. This reader fetches only ZIP metadata and
selected 8-bit test frames, preserving the official train/test distinction.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import json
import re
import struct
import sys
import urllib.request
import zipfile
import zlib
import xml.etree.ElementTree as ET
from pathlib import Path

URL = "https://zenodo.org/api/records/5336836/files/BabyPose_new.zip/content"
SIZE = 3025088724
ORIGINAL_URL = "https://zenodo.org/api/records/3891404/files/BabyPose.zip/content"
ORIGINAL_SIZE = 2998236349
V2_CSV_SHA256 = "ba8d2dd3fbf65b301156cf005d3cd4c14433b07ffd2d075d181cece02968dd3a"
V1_TEST_XLSX_SHA256 = "a2885043ff814b676b31c7b3842b206de25c6813c0bcccd0e7443fb91c0f39fe"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark_core.data import build_coco_handoff, load_handoff, sha256

DATASET_ID = "babypose_test160_limb12"
JOINTS = {5: "left_shoulder", 6: "right_shoulder", 7: "left_elbow", 8: "right_elbow",
          9: "left_hand", 10: "right_hand", 11: "left_hip", 12: "right_hip",
          13: "left_knee", 14: "right_knee", 15: "left_foot", 16: "right_foot"}
OLDER_NAMES = {"left_shoulder": "LS", "right_shoulder": "RS", "left_elbow": "LE", "right_elbow": "RE",
               "left_hand": "LW", "right_hand": "RW", "left_hip": "LH", "right_hip": "RH",
               "left_knee": "LK", "right_knee": "RK", "left_foot": "LA", "right_foot": "RA"}


def read_xlsx(data):
    """Read values from the first worksheet using only the standard library."""
    ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(io.BytesIO(data)) as book:
        shared = []
        if "xl/sharedStrings.xml" in book.namelist():
            tree = ET.fromstring(book.read("xl/sharedStrings.xml"))
            shared = ["".join(t.text or "" for t in si.findall(".//x:t", ns)) for si in tree.findall("x:si", ns)]
        sheet_name = sorted(name for name in book.namelist() if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", name))[0]
        tree = ET.fromstring(book.read(sheet_name))
        result = []
        for row in tree.findall(".//x:sheetData/x:row", ns):
            values = []
            for cell in row.findall("x:c", ns):
                ref = cell.attrib["r"]
                col = 0
                for char in re.match(r"[A-Z]+", ref).group():
                    col = col * 26 + ord(char) - ord("A") + 1
                while len(values) < col:
                    values.append(None)
                value = cell.find("x:v", ns)
                inline = cell.find("x:is", ns)
                if value is not None:
                    values[col - 1] = shared[int(value.text)] if cell.attrib.get("t") == "s" else value.text
                elif inline is not None:
                    values[col - 1] = "".join(t.text or "" for t in inline.findall(".//x:t", ns))
            result.append(values)
        return result


class HttpRangeFile(io.RawIOBase):
    def __init__(self, url=URL, size=SIZE, block_size=4 * 1024 * 1024):
        self.url, self.size, self.block_size = url, size, block_size
        self.pos = 0
        self.block_start, self.block = -1, b""
        self.bytes_downloaded = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        self.pos = (offset if whence == io.SEEK_SET else self.pos + offset if whence == io.SEEK_CUR else self.size + offset)
        if not 0 <= self.pos <= self.size:
            raise ValueError(f"ZIP seek outside source archive: {self.pos}")
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
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}", "User-Agent": "InfantPose-Bench/4 (research)"})
                with urllib.request.urlopen(req, timeout=90) as response:
                    if response.status != 206:
                        raise RuntimeError(f"Range request rejected: HTTP {response.status}")
                    data = response.read()
                    content_range = response.headers.get("Content-Range", "")
                if not content_range.startswith(f"bytes {start}-{end}/") or len(data) != end - start + 1:
                    raise RuntimeError(f"Unexpected range response: {content_range}, {len(data)} bytes")
                self.block_start, self.block = start, data
                self.bytes_downloaded += len(data)
            count = min(n, self.block_start + len(self.block) - self.pos)
            result.extend(self.block[self.pos - self.block_start:self.pos - self.block_start + count])
            self.pos += count
            n -= count
        return bytes(result)


def inspect(original=False):
    source = HttpRangeFile(ORIGINAL_URL, ORIGINAL_SIZE) if original else HttpRangeFile()
    with zipfile.ZipFile(source) as archive:
        names = archive.namelist()
        annotation_bytes = archive.read("BabyPose/annotations.csv") if not original else None
        if annotation_bytes:
            cache = ROOT / "02_datasets" / "source_archives" / "BabyPose_v2_annotations.csv"
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(annotation_bytes)
        annotation_rows = list(csv.DictReader(io.StringIO(annotation_bytes.decode("utf-8-sig")), delimiter=";")) if annotation_bytes else []
        print(json.dumps({
            "entries": len(names),
            "entry_details": [{"name": item.filename, "size": item.file_size, "compressed_size": item.compress_size, "compression": item.compress_type} for item in archive.infolist()[:12]],
            "spreadsheets": [name for name in names if name.lower().endswith(".xlsx")],
            "sample_paths": names[:20],
            "eight_bit_frames": sum(bool(re.search(r"8bit\.png$", name, re.I)) for name in names),
            "annotation_preview": archive.read("BabyPose/annotations.csv")[:1200].decode("utf-8-sig", errors="replace") if "BabyPose/annotations.csv" in names else None,
            "annotation_rows": len(annotation_rows),
            "annotation_subjects": dict(collections.Counter(row["image"].split("_")[0] for row in annotation_rows)),
            "test_spreadsheet_preview": read_xlsx(archive.read("BabyPose/babyPose-test.xlsx"))[:4] if original else None,
            "range_bytes_downloaded": source.bytes_downloaded,
        }, ensure_ascii=False, indent=2))


def prepare():
    canonical = ROOT / "02_datasets" / "canonical" / DATASET_ID
    if (canonical / "dataset_manifest.json").is_file():
        manifest_path = canonical / "dataset_manifest.json"
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("mapping_version") != "babypose_v2_limb12_to_coco17_v2":
            raise RuntimeError("Existing BabyPose handoff uses a different mapping; preserve it for review.")
        with (canonical / "sample_index.csv").open(newline="", encoding="utf-8") as file:
            stored_path = next(csv.DictReader(file))["image_path"]
        if stored_path.startswith("/") or re.match(r"^[A-Za-z]:[\\/]", stored_path):
            annotation = ROOT / "02_datasets" / "person_keypoints_babypose_test160_limb12.json"
            manifest = build_coco_handoff(DATASET_ID, canonical / "raw", annotation, canonical, 160)
            manifest.update({key: value for key, value in previous.items() if key not in manifest})
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest, rows = load_handoff(canonical)
        if len(rows) == 160 and manifest.get("mapping_version") == "babypose_v2_limb12_to_coco17_v2":
            print(json.dumps({"status": "reused_verified_handoff", "count": len(rows), "index_sha256": manifest["index_sha256"]}))
            return
        raise RuntimeError("Existing BabyPose handoff does not match the declared 160-image mapping; preserve it for review.")

    v2_cache = ROOT / "02_datasets" / "source_archives" / "BabyPose_v2_annotations.csv"
    if v2_cache.is_file():
        annotation_bytes = v2_cache.read_bytes()
    else:
        v2_source = HttpRangeFile()
        with zipfile.ZipFile(v2_source) as v2_archive:
            annotation_bytes = v2_archive.read("BabyPose/annotations.csv")
        v2_cache.parent.mkdir(parents=True, exist_ok=True)
        v2_cache.write_bytes(annotation_bytes)
    if hashlib.sha256(annotation_bytes).hexdigest() != V2_CSV_SHA256:
        raise RuntimeError("BabyPose v2 annotation CSV hash differs from the reviewed source")
    annotation_rows = list(csv.DictReader(io.StringIO(annotation_bytes.decode("utf-8-sig")), delimiter=";"))
    if len(annotation_rows) != 16000:
        raise RuntimeError(f"BabyPose v2 annotation count changed: {len(annotation_rows)}")
    patients = {}
    for row in annotation_rows:
        name = row["image"].strip()
        if not re.fullmatch(r"pz\d+_\d+_8bit\.png", name):
            raise RuntimeError(f"Unexpected BabyPose v2 image path: {name!r}")
        patients.setdefault(name.split("_")[0], []).append(row)
    if len(patients) != 16 or any(len(group) != 1000 for group in patients.values()):
        raise RuntimeError(f"BabyPose v2 subject counts changed: { {key: len(group) for key, group in patients.items()} }")

    # Verify the v2 CSV's value order and hand/foot semantics against the v1
    # test workbook for every overlapping pz3 frame, not just one example.
    workbook_cache = ROOT / "02_datasets" / "source_archives" / "BabyPose_test.xlsx"
    source = HttpRangeFile(ORIGINAL_URL, ORIGINAL_SIZE)
    with zipfile.ZipFile(source) as archive:
        if workbook_cache.is_file():
            workbook_bytes = workbook_cache.read_bytes()
        else:
            workbook_bytes = archive.read("BabyPose/babyPose-test.xlsx")
            workbook_cache.parent.mkdir(parents=True, exist_ok=True)
            workbook_cache.write_bytes(workbook_bytes)
        if hashlib.sha256(workbook_bytes).hexdigest() != V1_TEST_XLSX_SHA256:
            raise RuntimeError("BabyPose v1 test workbook hash differs from the reviewed source")
        cells = read_xlsx(workbook_bytes)
        headers = [str(value).strip() if value is not None else "" for value in cells[0]]
        column = {value: i for i, value in enumerate(headers)}
        required = {"filename"} | {f"{joint}_{axis}" for joint in OLDER_NAMES.values() for axis in "xy"}
        if not required.issubset(column):
            raise RuntimeError(f"BabyPose spreadsheet columns changed: missing {required - set(column)}")
        old_by_name = {}
        for row in cells[1:]:
            name = str(row[column["filename"]]).replace("\\", "/").strip()
            if not re.fullmatch(r"pz\d+/\d+_8bit\.png", name):
                raise RuntimeError(f"Unexpected test split image path: {name!r}")
            # The v1 workbook repeats pz3 filenames 16 times for different
            # patients. Its first block belongs to pz3; retain that block.
            old_by_name.setdefault(name.replace("/", "_"), row)
        if len(cells) - 1 != 4000:
            raise RuntimeError(f"Official BabyPose test split count changed: {len(cells)-1} rows")
        overlap_count = 0
        for new in annotation_rows:
            old = old_by_name.get(new["image"])
            if old is None:
                continue
            overlap_count += 1
            for joint, old_name in OLDER_NAMES.items():
                source_y, source_x = (float(value) for value in new[joint].split(","))
                old_x, old_y = float(old[column[f"{old_name}_x"]]), float(old[column[f"{old_name}_y"]])
                if abs(source_x - old_x) > 0.02 or abs(source_y - old_y) > 0.02:
                    raise RuntimeError(f"BabyPose v1/v2 coordinate mapping disagrees: {new['image']} {joint}")
        if overlap_count < 200:
            raise RuntimeError(f"Insufficient BabyPose v1/v2 coordinate matches: {overlap_count}")

        # The paper states that the last 250 of 1,000 annotated frames for
        # each patient form the held-out test partition. Select 10 of those
        # at equally spaced midpoints, retaining all 16 patients.
        chosen = []
        for patient in sorted(patients, key=lambda value: int(value[2:])):
            group = sorted(patients[patient], key=lambda row: int(re.search(r"_(\d+)_8bit", row["image"]).group(1)))
            chosen.extend(group[750 + i * 25 + 12] for i in range(10))
        source.block_size = 128 * 1024
        source.block_start, source.block = -1, b""
        images = []
        annotations = []
        raw = canonical / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        selected_paths = []
        for image_id, row in enumerate(chosen, 1):
            flat_name = row["image"]
            name = flat_name.replace("_", "/", 1)
            archive_name = "BabyPose/" + name
            destination = raw / flat_name
            entry = archive.getinfo(archive_name)
            payload = destination.read_bytes() if destination.is_file() else b""
            if len(payload) != entry.file_size or zlib.crc32(payload) != entry.CRC:
                payload = archive.read(archive_name)
                if payload[:8] != b"\x89PNG\r\n\x1a\n":
                    raise RuntimeError(f"Invalid PNG in source ZIP: {archive_name}")
                temporary = destination.with_suffix(".part")
                temporary.write_bytes(payload)
                temporary.replace(destination)
            width, height = struct.unpack(">II", payload[16:24])
            if (width, height) != (640, 480):
                raise RuntimeError(f"Unexpected BabyPose frame dimensions: {name}: {width}x{height}")
            points = [0.0] * 51
            visible = []
            for coco_index, joint in JOINTS.items():
                y, x = (float(value) for value in row[joint].split(","))
                # The released sheet includes negative sentinels and some
                # coordinates outside the 640x480 frame. Both are unlabelled
                # for image-plane COCO evaluation; never clip GT coordinates.
                if x < 0 or y < 0 or not 0 <= x < width or not 0 <= y < height:
                    continue
                points[3 * coco_index:3 * coco_index + 3] = [x, y, 2]
                visible.append((x, y))
            if len(visible) < 2:
                raise RuntimeError(f"Insufficient visible limb joints to define COCO area: {name}")
            x0, x1 = min(x for x, _ in visible), max(x for x, _ in visible)
            y0, y1 = min(y for _, y in visible), max(y for _, y in visible)
            bbox = [x0, y0, max(1.0, x1 - x0), max(1.0, y1 - y0)]
            images.append({"id": image_id, "file_name": flat_name, "width": width, "height": height})
            annotations.append({"id": image_id, "image_id": image_id, "category_id": 1,
                                "keypoints": points, "num_keypoints": len(visible),
                                "bbox": bbox, "area": bbox[2] * bbox[3], "iscrowd": 0})
            selected_paths.append(name)
            if image_id % 20 == 0:
                print(f"BabyPose prepared {image_id}/160; downloaded {source.bytes_downloaded / 1024**2:.1f} MiB", flush=True)

    annotation = ROOT / "02_datasets" / "person_keypoints_babypose_test160_limb12.json"
    annotation.write_text(json.dumps({"info": {"description": "BabyPose v2 annotation CSV, reconstructed last-250-per-patient test partition, 10 frames per patient",
                                                "source_doi": "10.5281/zenodo.5336836", "license": "CC BY 4.0",
                                                "mapping": "12 limb joints only; COCO face joints unlabelled with v=0",
                                                "area": "tight bounding rectangle of visible labelled limb joints"},
                                      "licenses": [{"id": 1, "name": "CC BY 4.0"}],
                                      "images": images, "annotations": annotations,
                                      "categories": [{"id": 1, "name": "person", "supercategory": "person",
                                                      "keypoints": ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
                                                                    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                                                                    "left_wrist", "right_wrist", "left_hip", "right_hip",
                                                                    "left_knee", "right_knee", "left_ankle", "right_ankle"],
                                                      "skeleton": [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11]]}]},
                                     ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = build_coco_handoff(DATASET_ID, raw, annotation, canonical, 160)
    manifest.update({"mapping_version": "babypose_v2_limb12_to_coco17_v2", "source_doi": "10.5281/zenodo.5336836",
                     "source_image_doi": "10.5281/zenodo.3891404",
                     "source_subject_counts": {key: len(value) for key, value in patients.items()},
                     "source_v1_v2_joint_crosschecks": overlap_count,
                     "source_archive_md5_published": "f133418828beb6ee86ff6b4c1ccc679c",
                     "source_archive_fully_downloaded": False,
                     "source_test_xlsx_sha256": hashlib.sha256(workbook_bytes).hexdigest(),
                     "source_v2_csv_sha256": hashlib.sha256(annotation_bytes).hexdigest(),
                     "source_selected_paths": selected_paths, "range_bytes_downloaded": source.bytes_downloaded,
                     "labelled_keypoints": 12, "unlabelled_coco_indices": [0, 1, 2, 3, 4],
                     "bbox_area_protocol": "tight rectangle of visible 12 limb joints; custom subset scale"})
    (canonical / "dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "prepared", "count": 160, "downloaded_mib": round(source.bytes_downloaded / 1024**2, 1),
                      "annotation_sha256": sha256(annotation), "index_sha256": manifest["index_sha256"]}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--original", action="store_true")
    args = parser.parse_args()
    if args.inspect:
        inspect(args.original)
    else:
        prepare()


if __name__ == "__main__":
    main()
