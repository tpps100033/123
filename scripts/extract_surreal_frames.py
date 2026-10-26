"""Extract the fixed SURREAL pilot frames using the existing WSL OpenCV runtime."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import urllib.parse
from pathlib import Path

import cv2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_path(root: Path, url: str) -> Path:
    path = urllib.parse.urlparse(url).path
    marker = "/SURREAL/"
    if marker not in path:
        raise ValueError(f"Unexpected source URL path: {url}")
    return root / Path(path.split(marker, 1)[1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if plan.get("dataset_id") != "surreal_cmu_test_run1_pilot500":
        raise ValueError("Unexpected SURREAL pilot plan dataset_id")
    samples = plan["samples"]
    clips = {(item["sequence_folder"], item["clip_stem"]): item for item in plan["clips"]}
    samples_by_clip: dict[tuple[str, str], list[dict]] = {}
    for sample in samples:
        key = (sample["sequence_folder"], sample["clip_stem"])
        samples_by_clip.setdefault(key, []).append(sample)
    args.output_root.mkdir(parents=True, exist_ok=True)
    image_root = args.output_root / "images"
    image_root.mkdir(parents=True, exist_ok=True)
    extracted = []

    for clip_index, (key, clip_samples) in enumerate(sorted(samples_by_clip.items()), 1):
        clip = clips[key]
        video_path = source_path(args.source_root, clip["video_url"])
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        expected_frames = sorted(int(row["frame_index_zero_based"]) for row in clip_samples)
        wanted = set(expected_frames)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"OpenCV could not open {video_path}")
        reported_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (width, height) != (320, 240):
            capture.release()
            raise ValueError(f"Unexpected SURREAL video size {(width, height)}: {video_path}")

        sample_by_frame = {int(row["frame_index_zero_based"]): row for row in clip_samples}
        frame_index = 0
        found = set()
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index in wanted:
                sample = sample_by_frame[frame_index]
                image_id = int(sample["sample_id"].rsplit(":", 1)[1])
                image_path = image_root / f"{image_id:06d}.png"
                if not cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
                    capture.release()
                    raise RuntimeError(f"Failed writing extracted frame: {image_path}")
                extracted.append({
                    "sample_id": sample["sample_id"],
                    "image_id": image_id,
                    "sequence_folder": key[0],
                    "clip_stem": key[1],
                    "frame_index_zero_based": frame_index,
                    "video_frame_count_reported": reported_count,
                    "width": width,
                    "height": height,
                    "image_path": image_path.relative_to(args.output_root).as_posix(),
                    "image_sha256": sha256(image_path),
                    "source_video": video_path.relative_to(args.source_root).as_posix(),
                })
                found.add(frame_index)
            frame_index += 1
            if found == wanted:
                break
        capture.release()
        if found != wanted:
            missing = sorted(wanted - found)
            raise RuntimeError(f"Frame mismatch in {video_path}; missing frame indices {missing}")
        if reported_count and max(wanted) >= reported_count:
            raise RuntimeError(f"Annotation frame index exceeds video frame count: {video_path}")
        if clip_index % 10 == 0 or clip_index == len(samples_by_clip):
            print(f"Extracted {len(extracted)}/{len(samples)} frames from {clip_index}/{len(samples_by_clip)} clips", flush=True)

    extracted.sort(key=lambda row: row["image_id"])
    if len(extracted) != len(samples):
        raise RuntimeError(f"Expected {len(samples)} frames, extracted {len(extracted)}")
    if len({row["image_sha256"] for row in extracted}) != len(extracted):
        raise RuntimeError("Duplicate RGB frame hashes found; review sample selection before evaluation")

    index_path = args.output_root / "extracted_frames.csv"
    with index_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(extracted[0]))
        writer.writeheader()
        writer.writerows(extracted)
    result = {
        "status": "frames_extracted",
        "dataset_id": plan["dataset_id"],
        "image_count": len(extracted),
        "clip_count": len(samples_by_clip),
        "unique_rgb_sha256": len({row["image_sha256"] for row in extracted}),
        "opencv_version": cv2.__version__,
        "index": str(index_path),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
