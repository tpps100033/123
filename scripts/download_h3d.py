"""Download the public H3D v1.01 archive from the Berkeley source in ranges."""
from __future__ import annotations

import argparse
import concurrent.futures
import re
import subprocess
import tempfile
from pathlib import Path

URL = "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/shape/h3d/h3d_v1.01.tar.gz"
SIZE = 340_493_932


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("02_datasets/source_archives/h3d_v1.01.tar.gz"))
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise SystemExit("workers must be between 1 and 8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.is_file() and args.output.stat().st_size == SIZE:
        print(f"Already downloaded: {args.output}")
        return

    chunk_size = (SIZE + args.workers - 1) // args.workers
    work = []
    for index in range(args.workers):
        start = index * chunk_size
        end = min(start + chunk_size, SIZE) - 1
        if start >= SIZE:
            break
        chunk = args.output.with_name(f"{args.output.name}.range{index:02d}.part")
        headers = chunk.with_suffix(chunk.suffix + ".headers")
        work.append((index, start, end, chunk, headers))

    def fetch(item: tuple[int, int, int, Path, Path]) -> tuple[int, Path]:
        index, start, end, chunk, headers = item
        expected = end - start + 1
        if chunk.is_file() and chunk.stat().st_size == expected:
            return index, chunk
        chunk.unlink(missing_ok=True)
        command = ["curl.exe", "--fail", "--silent", "--show-error", "--location",
                   "--retry", "4", "--retry-delay", "2", "--max-time", "1800",
                   "--range", f"{start}-{end}", "--dump-header", str(headers),
                   "--output", str(chunk), URL]
        subprocess.run(command, check=True, timeout=1810)
        response = headers.read_text(encoding="latin-1", errors="replace")
        ranges = re.findall(r"Content-Range:\s*bytes\s+(\d+)-(\d+)/(\d+)", response, re.IGNORECASE)
        if not ranges or tuple(map(int, ranges[-1])) != (start, end, SIZE):
            chunk.unlink(missing_ok=True)
            raise RuntimeError(f"Range {index} returned incorrect Content-Range: {ranges[-1:]}")
        if chunk.stat().st_size != expected:
            chunk.unlink(missing_ok=True)
            raise RuntimeError(f"Range {index} expected {expected} bytes, got {chunk.stat().st_size}")
        print(f"Range {index + 1}/{len(work)} verified ({expected / 1024**2:.1f} MiB)", flush=True)
        return index, chunk

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        chunks = sorted(pool.map(fetch, work))
    temporary = args.output.with_suffix(args.output.suffix + ".assembling")
    with temporary.open("wb") as output:
        for _, chunk in chunks:
            with chunk.open("rb") as source:
                while block := source.read(8 * 1024 * 1024):
                    output.write(block)
    if temporary.stat().st_size != SIZE:
        raise RuntimeError(f"Assembled archive size mismatch: {temporary.stat().st_size} != {SIZE}")
    temporary.replace(args.output)
    for _, chunk in chunks:
        chunk.unlink(missing_ok=True)
        chunk.with_suffix(chunk.suffix + ".headers").unlink(missing_ok=True)
    print(f"Complete: {args.output} ({SIZE} bytes)")


if __name__ == "__main__":
    main()
