"""Plan and securely download a small SURREAL CMU test/run1 sample.

The default invocation is plan-only and needs no SURREAL credentials. Use
--download only after personally accepting the official data license. The
username/password are read interactively (password hidden), used only in
memory for HTTPS Basic Auth, and never written to disk or printed.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import random
import sys
import urllib.error
import urllib.parse
import urllib.request
from getpass import getpass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_LIST = ROOT / "02_datasets/source_archives/surreal_test_mp4_list.txt"
INFO_LIST = ROOT / "02_datasets/source_archives/surreal_test_info_list.txt"
SEGM_LIST = ROOT / "02_datasets/source_archives/surreal_test_segm_list.txt"
PLAN_ROOT = ROOT / "02_datasets/canonical/surreal_cmu_test_run1_pilot500"
SOURCE_ROOT = ROOT / "02_datasets/source_archives/surreal_cmu_test_run1_pilot500"
FRAME_INDICES = (10, 30, 50, 70, 90)
RUN_MARKER = "/data/cmu/test/run1/"
OFFICIAL_BASE = "https://lsh.paris.inria.fr"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_urls(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Official file index is missing: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip().startswith("https://")]


def canonical_key(url: str, suffix: str) -> tuple[str, str]:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "lsh.paris.inria.fr":
        raise ValueError(f"Refusing non-official or non-HTTPS SURREAL URL: {url}")
    path = parsed.path
    if not path.endswith(suffix):
        raise ValueError(f"Unexpected SURREAL index URL: {url}")
    relative = path.split(RUN_MARKER, 1)
    if len(relative) != 2:
        raise ValueError(f"Not an official test/run1 URL: {url}")
    rel = Path(relative[1])
    if suffix == ".mp4":
        clip_stem = rel.stem
    else:
        clip_stem = rel.name[:-len(suffix)]
    return rel.parent.as_posix(), clip_stem


def build_plan(target: int, seed: int) -> tuple[list[dict], dict]:
    if target < 300 or target > 1000:
        raise ValueError("Choose a sample target between 300 and 1000 images.")
    if target % len(FRAME_INDICES):
        raise ValueError(f"Target must be divisible by {len(FRAME_INDICES)} frames per clip.")
    videos = read_urls(SOURCE_LIST)
    infos = read_urls(INFO_LIST)
    segmentations = read_urls(SEGM_LIST)

    def modality_map(urls: list[str], suffix: str) -> dict[tuple[str, str], str]:
        selected = {}
        for url in urls:
            if RUN_MARKER not in urllib.parse.urlparse(url).path:
                continue
            key = canonical_key(url, suffix)
            selected[key] = url
        return selected

    video_map = modality_map(videos, ".mp4")
    info_map = modality_map(infos, "_info.mat")
    segm_map = modality_map(segmentations, "_segm.mat")
    common = sorted(set(video_map) & set(info_map) & set(segm_map))
    if len(common) < target // len(FRAME_INDICES):
        raise RuntimeError("Not enough run1 clips have all three modalities in the official indexes.")

    by_sequence: dict[str, list[tuple[str, str]]] = {}
    for key in common:
        sequence = key[0]
        tokens = sequence.split("_")
        known_bad_shape = (
            sequence in {"18", "19", "143_21"}
            or (len(tokens) >= 2 and tokens[0] == "ung" and tokens[1] in {"18", "19"})
            or sequence.endswith("_143_21")
        )
        if known_bad_shape:
            continue
        by_sequence.setdefault(sequence, []).append(key)
    needed = target // len(FRAME_INDICES)
    if len(by_sequence) < needed:
        raise RuntimeError(f"Only {len(by_sequence)} clean sequence folders; need {needed}.")

    rng = random.Random(seed)
    sequence_names = sorted(by_sequence)
    chosen_sequences = sorted(rng.sample(sequence_names, needed))
    clips = []
    for sequence in chosen_sequences:
        key = rng.choice(sorted(by_sequence[sequence]))
        clips.append({
            "sequence_folder": sequence,
            "clip_stem": key[1],
            "video_url": video_map[key],
            "info_url": info_map[key],
            "segm_url": segm_map[key],
            "frame_indices_zero_based": list(FRAME_INDICES),
        })
    clips.sort(key=lambda item: (item["sequence_folder"], item["clip_stem"]))

    samples = []
    sample_index = 0
    for clip in clips:
        for frame_index in FRAME_INDICES:
            sample_index += 1
            samples.append({
                "sample_id": f"surreal_cmu_test_run1_pilot500:{sample_index:04d}",
                "sequence_folder": clip["sequence_folder"],
                "clip_stem": clip["clip_stem"],
                "frame_index_zero_based": frame_index,
            })
    plan = {
        "schema_version": "surreal-pilot-acquisition-plan-v1",
        "dataset_id": "surreal_cmu_test_run1_pilot500",
        "source_url": "https://www.di.ens.fr/willow/research/surreal/data/",
        "split": "cmu/test/run1 (30% overlap setting)",
        "target_images": target,
        "seed": seed,
        "sequence_folders": len(chosen_sequences),
        "clips": clips,
        "frames_per_clip": list(FRAME_INDICES),
        "samples": samples,
        "modalities": ["mp4", "info.mat", "segm.mat"],
        "excluded_modalities": ["depth", "gtflow"],
        "official_index_sha256": {
            path.name: sha256(path) for path in (SOURCE_LIST, INFO_LIST, SEGM_LIST)
        },
        "license": "SURREAL non-commercial scientific research license; user confirmed personal acceptance",
        "data_handling": "local-only; never commit or redistribute images, annotations, or derived image crops",
    }
    return clips, plan


def relative_destination(url: str) -> Path:
    path = urllib.parse.urlparse(url).path
    marker = "/SURREAL/"
    if marker not in path:
        raise ValueError(f"Unexpected SURREAL data host path: {url}")
    return Path(path.split(marker, 1)[1])


def authenticated_open(url: str, method: str, auth_header: str):
    request = urllib.request.Request(url, method=method, headers={"Authorization": auth_header})
    return urllib.request.urlopen(request, timeout=45)


def fetch_credentials() -> tuple[str, str]:
    username = input("SURREAL username (not stored): ").strip()
    password = getpass("SURREAL password (hidden; not stored): ")
    if not username or not password:
        raise ValueError("Both credentials are required.")
    token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
    # Avoid retaining an extra reference beyond this short function.
    del username, password
    return token, "Basic " + token


def download_plan(clips: list[dict], max_bytes: int) -> dict:
    token, auth_header = fetch_credentials()
    del token
    files = []
    for clip in clips:
        for modality in ("video_url", "info_url", "segm_url"):
            files.append(clip[modality])

    sizes = {}
    for index, url in enumerate(files, 1):
        try:
            with authenticated_open(url, "HEAD", auth_header) as response:
                size = response.headers.get("Content-Length")
                if size is None:
                    raise RuntimeError(f"Server did not report size for selected file {index}/{len(files)}")
                sizes[url] = int(size)
        except urllib.error.HTTPError as exc:
            if exc.code == 401 or exc.code == 403:
                raise PermissionError("SURREAL credentials were rejected or lack download permission.") from None
            raise RuntimeError(f"SURREAL preflight failed with HTTP {exc.code} on file {index}/{len(files)}") from None
        print(f"Preflighted {index}/{len(files)} files", flush=True)

    total_bytes = sum(sizes.values())
    if total_bytes > max_bytes:
        raise RuntimeError(
            f"Preflight size {total_bytes / 1024**3:.2f} GiB exceeds the safe download cap "
            f"{max_bytes / 1024**3:.2f} GiB; no files were downloaded."
        )
    print(f"Preflight total: {total_bytes / 1024**3:.2f} GiB; starting download.", flush=True)

    downloaded = 0
    for index, url in enumerate(files, 1):
        relative = relative_destination(url)
        destination = SOURCE_ROOT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        expected = sizes[url]
        if destination.is_file() and destination.stat().st_size == expected:
            print(f"Cached {index}/{len(files)}: {relative.as_posix()}", flush=True)
            downloaded += expected
            continue
        partial = destination.with_suffix(destination.suffix + ".part")
        try:
            with authenticated_open(url, "GET", auth_header) as response, partial.open("wb") as out:
                while block := response.read(1024 * 1024):
                    out.write(block)
            if partial.stat().st_size != expected:
                raise RuntimeError(f"Size check failed for {relative.as_posix()}")
            partial.replace(destination)
        except Exception:
            partial.unlink(missing_ok=True)
            raise
        downloaded += expected
        print(f"Downloaded {index}/{len(files)} ({downloaded / 1024**3:.2f} GiB): {relative.as_posix()}", flush=True)

    return {
        "status": "downloaded",
        "files": len(files),
        "bytes": total_bytes,
        "download_root": str(SOURCE_ROOT),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--max-download-gb", type=float, default=5.0,
                        help="Hard transfer cap for this run; leaves most of the stated daily allowance unused.")
    parser.add_argument("--download", action="store_true", help="Prompt locally for credentials and download selected clips.")
    args = parser.parse_args()
    clips, plan = build_plan(args.target, args.seed)
    PLAN_ROOT.mkdir(parents=True, exist_ok=True)
    plan_path = PLAN_ROOT / "acquisition_plan.json"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    sample_path = PLAN_ROOT / "planned_samples.csv"
    with sample_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(plan["samples"][0]))
        writer.writeheader()
        writer.writerows(plan["samples"])
    print(json.dumps({"status": "plan_ready", "target": args.target,
                      "clips": len(clips), "plan": str(plan_path),
                      "samples_csv": str(sample_path)}, ensure_ascii=False, indent=2))
    if args.download:
        result = download_plan(clips, int(args.max_download_gb * 1024**3))
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (PermissionError, RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
