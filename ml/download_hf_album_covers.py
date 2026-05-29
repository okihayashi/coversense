#!/usr/bin/env python3
"""Download album covers from Hugging Face into a simple imagefolder layout."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

from datasets import DatasetDict, load_dataset
from PIL import Image
from tqdm import tqdm


DEFAULT_DATASET = "eong/20k-Album-Covers-within-20-Genres"
DEFAULT_OUTPUT_DIR = Path("data/album_covers_20_genres")


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "unknown"


def image_to_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image
    return image.convert("RGB")


def choose_split(dataset):
    if isinstance(dataset, DatasetDict):
        if "train" in dataset:
            return dataset["train"]
        return dataset[next(iter(dataset.keys()))]
    return dataset


def label_names_for(dataset, label_column: str) -> list[str] | None:
    feature = dataset.features.get(label_column)
    names = getattr(feature, "names", None)
    return list(names) if names else None


def row_label(row, label_column: str, label_names: list[str] | None) -> tuple[int | str, str]:
    raw_label = row[label_column]
    if isinstance(raw_label, int) and label_names:
        return raw_label, label_names[raw_label]
    return raw_label, str(raw_label)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--max-per-label", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    images_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    dataset = choose_split(load_dataset(args.dataset))
    label_names = label_names_for(dataset, args.label_column)
    counts: defaultdict[str, int] = defaultdict(int)
    records = []

    for index, row in enumerate(tqdm(dataset, desc="Exporting covers")):
        _, label_name = row_label(row, args.label_column, label_names)
        label_slug = slugify(label_name)

        if args.max_per_label is not None and counts[label_slug] >= args.max_per_label:
            continue

        image = row[args.image_column]
        if not isinstance(image, Image.Image):
            image = Image.open(image)

        image = image_to_rgb(image)
        image.thumbnail((args.image_size, args.image_size))

        label_dir = images_dir / label_slug
        label_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{label_slug}-{counts[label_slug]:05d}.jpg"
        image_path = label_dir / filename
        image.save(image_path, quality=92)

        counts[label_slug] += 1
        records.append(
            {
                "image_path": image_path.as_posix(),
                "label": label_slug,
                "label_name": label_name,
                "source_index": index,
            }
        )

    with (output_dir / "metadata.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["image_path", "label", "label_name", "source_index"])
        writer.writeheader()
        writer.writerows(records)

    label_payload = {
        "dataset": args.dataset,
        "labels": sorted(counts.keys()),
        "label_counts": dict(sorted(counts.items())),
    }
    (output_dir / "labels.json").write_text(json.dumps(label_payload, indent=2), encoding="utf-8")

    print(f"Exported {len(records)} covers to {output_dir}")


if __name__ == "__main__":
    main()
