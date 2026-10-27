"""Render a contact sheet of COCO-17 handoff images with source labels."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
COCO_SKELETON = [
    (16, 14), (14, 12), (17, 15), (15, 13), (12, 13), (6, 12), (7, 13),
    (6, 7), (6, 8), (8, 10), (7, 9), (9, 11), (2, 3), (1, 2), (1, 3),
    (2, 4), (3, 5), (4, 6), (5, 7),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", type=Path, default=ROOT / "02_datasets/canonical/spinetrack_unreal_singleperson_pilot1000")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    handoff = args.handoff.resolve()
    with (handoff / "sample_index.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    coco = json.loads((handoff / "annotations.json").read_text(encoding="utf-8"))
    annotations = {int(item["image_id"]): item for item in coco["annotations"]}
    if not rows or args.count < 1 or args.columns < 1:
        raise ValueError("A nonempty handoff, positive count, and positive column count are required")
    count = min(args.count, len(rows))
    positions = [round(i * (len(rows) - 1) / max(1, count - 1)) for i in range(count)]
    first = rows[positions[0]]
    with Image.open(handoff / first["image_path"]) as first_image:
        source_w, source_h = first_image.size
    thumb_w = 320
    thumb_h = round(thumb_w * source_h / source_w)
    label_h = 24
    rows_count = (count + args.columns - 1) // args.columns
    sheet = Image.new("RGB", (args.columns * thumb_w, rows_count * (thumb_h + label_h)), (238, 238, 238))

    for tile_index, row_index in enumerate(positions):
        row = rows[row_index]
        image_path = handoff / row["image_path"]
        with Image.open(image_path) as source:
            image = source.convert("RGB").resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(image)
        ann = annotations[int(row["image_id"])]
        keypoints = ann["keypoints"]
        width, height = int(row["width"]), int(row["height"])

        def point(index: int) -> tuple[float, float]:
            x, y = keypoints[index * 3:index * 3 + 2]
            return x * thumb_w / width, y * thumb_h / height

        for one_based_a, one_based_b in COCO_SKELETON:
            a, b = one_based_a - 1, one_based_b - 1
            if int(keypoints[a * 3 + 2]) > 0 and int(keypoints[b * 3 + 2]) > 0:
                draw.line((*point(a), *point(b)), fill=(40, 220, 70), width=2)
        for index in range(17):
            if int(keypoints[index * 3 + 2]) > 0:
                x, y = point(index)
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(255, 45, 45), outline=(255, 255, 255))

        tile = Image.new("RGB", (thumb_w, thumb_h + label_h), (255, 255, 255))
        tile.paste(image, (0, 0))
        ImageDraw.Draw(tile).text((5, thumb_h + 5), f"sample {row_index + 1:04d} | {ann['num_keypoints']}/17 points", fill=(0, 0, 0))
        x0, y0 = (tile_index % args.columns) * thumb_w, (tile_index // args.columns) * (thumb_h + label_h)
        sheet.paste(tile, (x0, y0))

    output = args.output or handoff / "visual_audit_contact_sheet.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    print(json.dumps({"status": "created", "images": count, "columns": args.columns, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
