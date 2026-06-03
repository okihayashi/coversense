#!/usr/bin/env python3
"""Compare dataset-source coverage and model metrics side by side."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_DATASETS = [
    "hf-trained-hf-test:data/album_covers_20_genres:reports/model_runs/clip-mlp/metrics.json",
    "hf-trained-musicbrainz-test:data/musicbrainz_cover_art:reports/datasets/musicbrainz/hf-serving-model-eval/metrics.json",
    "musicbrainz-trained-logreg:data/musicbrainz_cover_art:reports/datasets/musicbrainz/clip-logreg/metrics.json",
    "musicbrainz-trained-linear-svc:data/musicbrainz_cover_art:reports/datasets/musicbrainz/clip-linear-svc/metrics.json",
    "musicbrainz-trained-random-forest:data/musicbrainz_cover_art:reports/datasets/musicbrainz/clip-random-forest/metrics.json",
    "musicbrainz-trained-mlp:data/musicbrainz_cover_art:reports/datasets/musicbrainz/clip-mlp/metrics.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        metavar="NAME:DATA_DIR[:METRICS_JSON]",
        help="Dataset source to compare. Can be passed multiple times.",
    )
    parser.add_argument("--output", type=Path, default=Path("reports/dataset_source_comparison.md"))
    parser.add_argument("--json-output", type=Path, default=Path("reports/dataset_source_comparison.json"))
    return parser.parse_args()


def parse_dataset_spec(spec: str) -> tuple[str, Path, Path | None]:
    parts = spec.split(":")
    if len(parts) not in {2, 3}:
        raise ValueError(f"Expected NAME:DATA_DIR[:METRICS_JSON], got: {spec}")
    name, data_dir = parts[0], Path(parts[1])
    metrics_path = Path(parts[2]) if len(parts) == 3 else None
    return name, data_dir, metrics_path


def read_metadata(data_dir: Path) -> list[dict[str, str]]:
    metadata_path = data_dir / "metadata.csv"
    if not metadata_path.exists():
        return []
    with metadata_path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def read_json(path: Path | None, default: Any) -> Any:
    if path is None or not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def existing_image_count(rows: list[dict[str, str]]) -> int:
    return sum(1 for row in rows if Path(row.get("image_path", "")).exists())


def source_summary(name: str, data_dir: Path, metrics_path: Path | None) -> dict[str, Any]:
    rows = read_metadata(data_dir)
    labels = Counter(row.get("label", "") for row in rows)
    label_counts = {label: count for label, count in sorted(labels.items()) if label}
    metrics = read_json(metrics_path, {})
    label_total = len(label_counts)
    min_count = min(label_counts.values()) if label_counts else 0
    max_count = max(label_counts.values()) if label_counts else 0
    imbalance_ratio = round(max_count / min_count, 2) if min_count else None

    return {
        "name": name,
        "data_dir": data_dir.as_posix(),
        "metadata_exists": (data_dir / "metadata.csv").exists(),
        "row_count": len(rows),
        "existing_image_count": existing_image_count(rows),
        "label_count": label_total,
        "min_per_label": min_count,
        "max_per_label": max_count,
        "imbalance_ratio": imbalance_ratio,
        "label_counts": label_counts,
        "metrics_path": metrics_path.as_posix() if metrics_path else None,
        "classifier": metrics.get("classifier"),
        "accuracy_top_1": metrics.get("accuracy_top_1"),
        "accuracy_top_3": metrics.get("accuracy_top_3"),
        "accuracy_broad_top_1": metrics.get("accuracy_broad_top_1"),
        "accuracy_broad_top_3": metrics.get("accuracy_broad_top_3"),
        "near_miss_rate": metrics.get("near_miss_rate"),
        "far_miss_rate": metrics.get("far_miss_rate"),
        "hierarchical_score": metrics.get("hierarchical_score"),
    }


def percent(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value * 100:.1f}%"


def markdown_table(summaries: list[dict[str, Any]]) -> str:
    lines = [
        "# Dataset Source Comparison",
        "",
        "| Source | Rows | Images | Labels | Min/Max Per Label | Imbalance | Top-1 | Top-3 | Broad Top-1 | Broad Top-3 | Near Miss | Far Miss | Hierarchical |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(
            "| {name} | {row_count} | {existing_image_count} | {label_count} | {min_per_label}/{max_per_label} | "
            "{imbalance} | {top1} | {top3} | {broad_top1} | {broad_top3} | {near_miss} | {far_miss} | {hierarchical} |".format(
                name=item["name"],
                row_count=item["row_count"],
                existing_image_count=item["existing_image_count"],
                label_count=item["label_count"],
                min_per_label=item["min_per_label"],
                max_per_label=item["max_per_label"],
                imbalance=item["imbalance_ratio"] if item["imbalance_ratio"] is not None else "--",
                top1=percent(item["accuracy_top_1"]),
                top3=percent(item["accuracy_top_3"]),
                broad_top1=percent(item["accuracy_broad_top_1"]),
                broad_top3=percent(item["accuracy_broad_top_3"]),
                near_miss=percent(item["near_miss_rate"]),
                far_miss=percent(item["far_miss_rate"]),
                hierarchical=percent(item["hierarchical_score"]),
            )
        )

    lines.extend(["", "## Label Counts", ""])
    for item in summaries:
        lines.append(f"### {item['name']}")
        lines.append("")
        if not item["label_counts"]:
            lines.append("_No metadata found._")
            lines.append("")
            continue
        for label, count in item["label_counts"].items():
            lines.append(f"- `{label}`: {count}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    specs = args.dataset or DEFAULT_DATASETS
    summaries = [source_summary(*parse_dataset_spec(spec)) for spec in specs]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown_table(summaries), encoding="utf-8")
    args.json_output.write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    print(args.output)
    print(args.json_output)


if __name__ == "__main__":
    main()
