"""Inspect official SVIRO WPDM file names and sizes using one-byte ranges."""
from __future__ import annotations

import concurrent.futures
import json
import re
import subprocess
import tempfile
import urllib.request
from pathlib import Path

SOURCE = "https://sviro.kl.dfki.de"
VEHICLES = (
    ("i3", "bmw-i3"), ("tucson", "hyundai-tucson"), ("x5", "bmw-x5"),
    ("zoe", "renault-zoe"), ("gsf", "lexus-gs-f"), ("hilux", "toyota-hilux"),
    ("model3", "tesla-model-3"), ("tiguan", "vw-tiguan"),
    ("aclass", "mercedes-class-a"), ("escape", "ford-escape"),
)
SUFFIXES = (None, 2)


def probe(item: tuple[str, str, int | None]) -> dict:
    vehicle, slug, suffix = item
    page = f"{SOURCE}/download/{slug}{'-' + str(suffix) if suffix else ''}/"
    try:
        with urllib.request.urlopen(urllib.request.Request(page, headers={"User-Agent": "InfantPose-Bench/4"}),
                                    timeout=30) as response:
            html = response.read().decode("utf-8", errors="replace")
    except Exception as error:
        return {"vehicle": vehicle, "page": page, "error": str(error)}
    match = re.search(r"wpdmdl=(\d+)", html)
    if not match:
        return {"vehicle": vehicle, "page": page, "error": "no WPDM id"}
    download_id = int(match.group(1))
    url = f"{page}?wpdmdl={download_id}"
    with tempfile.NamedTemporaryFile(prefix="sviro_wpdm_", suffix=".headers", delete=False) as file:
        headers_path = Path(file.name)
    try:
        process = subprocess.Popen(["curl.exe", "--fail", "--silent", "--show-error", "--location",
                                    "--range", "0-1", "--dump-header", str(headers_path), url],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            first = process.stdout.read(2)
        finally:
            process.terminate()
            process.wait(timeout=10)
    except Exception as error:
        headers_path.unlink(missing_ok=True)
        return {"vehicle": vehicle, "page": page, "wpdmdl": download_id, "error": str(error)}
    headers = headers_path.read_text(encoding="latin-1", errors="replace")
    headers_path.unlink(missing_ok=True)
    disposition = re.findall(r"Content-Disposition:\s*attachment;\s*filename=\"?([^\"\r\n]+)",
                             headers, re.IGNORECASE)
    lengths = re.findall(r"Content-Range:\s*bytes\s+\d+-\d+/(\d+)", headers, re.IGNORECASE)
    return {"vehicle": vehicle, "suffix": suffix or 1, "wpdmdl": download_id,
            "download_name": disposition[-1] if disposition else None,
            "size_bytes": int(lengths[-1]) if lengths else None,
            "url": url, "probe_bytes": len(first)}


def main() -> None:
    items = [(vehicle, slug, suffix) for vehicle, slug in VEHICLES for suffix in SUFFIXES]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(probe, items))
    report_path = Path(__file__).resolve().parents[1] / "reports" / "sviro_wpdm_inventory.json"
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps([{"vehicle": row.get("vehicle"), "suffix": row.get("suffix"),
                       "wpdmdl": row.get("wpdmdl"), "download_name": row.get("download_name"),
                       "size_bytes": row.get("size_bytes"), "error": row.get("error")}
                      for row in results], indent=2))


if __name__ == "__main__":
    main()
