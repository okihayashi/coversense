#!/usr/bin/env python3
"""Apply reviewed label decisions to a dataset metadata CSV.

The raw downloaded dataset stays unchanged. This script writes a derived
metadata file that can be passed to build_embeddings.py.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


DEFAULT_METADATA = Path("data/album_covers_20_genres/metadata.csv")
DEFAULT_REVIEWS = Path("data/label_reviews.csv")
DEFAULT_OUTPUT = Path("data/album_covers_20_genres_reviewed/metadata.csv")
REVIEW_FIELDNAMES = [
    "original_label",
    "review_status",
    "secondary_labels",
    "review_reason",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--reviews", type=Path, default=DEFAULT_REVIEWS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def title_label(label: str) -> str:
    return (
        label.replace("drumnbass", "drum & bass")
        .replace("deathmetal", "death metal")
        .replace("doommetal", "doom metal")
        .replace("heavymetal", "heavy metal")
        .replace("hiphop", "hip-hop")
        .replace("lofi", "lo-fi")
        .replace("psychedelicrock", "psychedelic rock")
        .title()
    )


def reviewed_label(row: dict[str, str], review: dict[str, str] | None) -> str | None:
    if review is None:
        return row["label"]
    status = review.get("review_status", "")
    if status == "exclude":
        return None
    if status in {"relabel", "multi_label"} and review.get("reviewed_label"):
        return review["reviewed_label"]
    return row["label"]


def main() -> None:
    args = parse_args()
    rows = read_csv(args.metadata)
    reviews = {row["image_path"]: row for row in read_csv(args.reviews)}
    output_rows = []
    source_fieldnames = list(rows[0].keys()) if rows else ["image_path", "label", "label_name"]
    output_fieldnames = [*source_fieldnames, *[field for field in REVIEW_FIELDNAMES if field not in source_fieldnames]]
    status_counts: Counter[str] = Counter()
    label_changes = 0

    for row in rows:
        review = reviews.get(row["image_path"])
        label = reviewed_label(row, review)
        if label is None:
            status_counts["exclude"] += 1
            continue

        original_label = row["label"]
        status = review.get("review_status", "") if review else ""
        if label != original_label:
            label_changes += 1
        if status:
            status_counts[status] += 1

        output_rows.append(
            {
                **row,
                "label": label,
                "label_name": title_label(label),
                "original_label": original_label,
                "review_status": status,
                "secondary_labels": review.get("secondary_labels", "") if review else "",
                "review_reason": review.get("reason", "") if review else "",
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=output_fieldnames)
        writer.writeheader()
        for row in output_rows:
            writer.writerow({field: row.get(field, "") for field in output_fieldnames})

    summary = {
        "source_metadata": args.metadata.as_posix(),
        "reviews": args.reviews.as_posix(),
        "output": args.output.as_posix(),
        "input_rows": len(rows),
        "output_rows": len(output_rows),
        "excluded_rows": len(rows) - len(output_rows),
        "label_changes": label_changes,
        "review_status_counts": dict(sorted(status_counts.items())),
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
