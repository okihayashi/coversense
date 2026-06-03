#!/usr/bin/env python3
"""Reusable dataset-review and evaluation loop for CoverSense.

The loop keeps the raw downloaded datasets unchanged. Reviewed labels are
applied into derived metadata, cached CLIP embeddings are relabeled/filtered,
the serving classifier is retrained, and both cross-dataset evaluations are
refreshed so the far-miss objective stays visible.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from ml.genre_taxonomy import broad_genre
except ModuleNotFoundError:  # pragma: no cover - supports python ml/review_loop.py
    from genre_taxonomy import broad_genre


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


@dataclass(frozen=True)
class ReviewLoopConfig:
    raw_metadata: Path = PROJECT_ROOT / "data" / "album_covers_20_genres" / "metadata.csv"
    reviewed_metadata: Path = PROJECT_ROOT / "data" / "album_covers_20_genres_reviewed" / "metadata.csv"
    reviews: Path = PROJECT_ROOT / "data" / "label_reviews.csv"
    hf_embeddings: Path = PROJECT_ROOT / "embeddings" / "hf-reviewed-clip-vit-base-patch32.npz"
    musicbrainz_embeddings: Path = PROJECT_ROOT / "embeddings" / "musicbrainz-reviewed-clip-vit-base-patch32.npz"
    hf_model_dir: Path = PROJECT_ROOT / "models" / "clip-mlp-reviewed-hierarchical"
    hf_report_dir: Path = PROJECT_ROOT / "reports" / "model_runs" / "clip-mlp-reviewed-hierarchical"
    musicbrainz_model: Path = (
        PROJECT_ROOT
        / "models"
        / "datasets"
        / "musicbrainz"
        / "clip-mlp-reviewed-hierarchical"
        / "coversense_clip_classifier.joblib"
    )
    hf_on_musicbrainz_report_dir: Path = (
        PROJECT_ROOT
        / "reports"
        / "datasets"
        / "musicbrainz"
        / "hf-reviewed-hierarchical-on-musicbrainz-reviewed-eval"
    )
    musicbrainz_on_hf_report_dir: Path = (
        PROJECT_ROOT
        / "reports"
        / "model_runs"
        / "musicbrainz-reviewed-hierarchical-on-hf-reviewed-eval"
    )
    comparison_markdown: Path = PROJECT_ROOT / "reports" / "reviewed_dataset_cross_comparison.md"
    comparison_json: Path = PROJECT_ROOT / "reports" / "reviewed_dataset_cross_comparison.json"


def rel(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


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


def apply_reviewed_labels(config: ReviewLoopConfig) -> dict[str, Any]:
    rows = read_csv(config.raw_metadata)
    reviews = {row["image_path"]: row for row in read_csv(config.reviews)}
    source_fieldnames = list(rows[0].keys()) if rows else ["image_path", "label", "label_name"]
    review_fieldnames = ["original_label", "review_status", "secondary_labels", "review_reason"]
    output_fieldnames = [*source_fieldnames, *[field for field in review_fieldnames if field not in source_fieldnames]]

    output_rows: list[dict[str, str]] = []
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

    config.reviewed_metadata.parent.mkdir(parents=True, exist_ok=True)
    with config.reviewed_metadata.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=output_fieldnames)
        writer.writeheader()
        for row in output_rows:
            writer.writerow({field: row.get(field, "") for field in output_fieldnames})

    summary = {
        "source_metadata": rel(config.raw_metadata),
        "reviews": rel(config.reviews),
        "output": rel(config.reviewed_metadata),
        "input_rows": len(rows),
        "output_rows": len(output_rows),
        "excluded_rows": len(rows) - len(output_rows),
        "label_changes": label_changes,
        "review_status_counts": dict(sorted(status_counts.items())),
    }
    config.reviewed_metadata.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def refresh_embedding_labels(config: ReviewLoopConfig) -> dict[str, Any]:
    metadata_labels = {row["image_path"]: row["label"] for row in read_csv(config.reviewed_metadata)}
    cache = np.load(config.hf_embeddings, allow_pickle=False)
    image_paths = cache["image_paths"].astype(str)
    path_set = set(image_paths)
    keep_indices = [index for index, image_path in enumerate(image_paths) if image_path in metadata_labels]
    dropped = [image_path for image_path in image_paths if image_path not in metadata_labels]
    missing = [image_path for image_path in metadata_labels if image_path not in path_set]
    if missing:
        raise ValueError(f"{len(missing)} reviewed metadata rows are missing from the embedding cache.")

    old_labels = cache["labels"].astype(str)[keep_indices]
    new_labels = np.array([metadata_labels[image_paths[index]] for index in keep_indices])
    changed = int(np.sum(old_labels != new_labels))
    np.savez(
        config.hf_embeddings,
        embeddings=cache["embeddings"][keep_indices],
        labels=new_labels,
        image_paths=cache["image_paths"][keep_indices],
    )
    return {
        "cache": rel(config.hf_embeddings),
        "kept": len(keep_indices),
        "dropped": len(dropped),
        "changed_kept_labels": changed,
        "first_dropped": dropped[:5],
    }


def run_command(args: list[str]) -> None:
    subprocess.run(args, cwd=PROJECT_ROOT, check=True)


def train_hf_model(config: ReviewLoopConfig, max_iter: int) -> None:
    run_command(
        [
            PYTHON,
            "ml/train_classifier.py",
            "--embeddings",
            rel(config.hf_embeddings),
            "--classifier",
            "mlp",
            "--prediction-policy",
            "hierarchical",
            "--model-dir",
            rel(config.hf_model_dir),
            "--report-dir",
            rel(config.hf_report_dir),
            "--max-iter",
            str(max_iter),
        ]
    )


def evaluate_cross_dataset(config: ReviewLoopConfig) -> None:
    run_command(
        [
            PYTHON,
            "ml/evaluate_classifier.py",
            "--model",
            rel(config.hf_model_dir / "coversense_clip_classifier.joblib"),
            "--embeddings",
            rel(config.musicbrainz_embeddings),
            "--report-dir",
            rel(config.hf_on_musicbrainz_report_dir),
        ]
    )
    run_command(
        [
            PYTHON,
            "ml/evaluate_classifier.py",
            "--model",
            rel(config.musicbrainz_model),
            "--embeddings",
            rel(config.hf_embeddings),
            "--report-dir",
            rel(config.musicbrainz_on_hf_report_dir),
        ]
    )


def compare_datasets(config: ReviewLoopConfig) -> None:
    run_command(
        [
            PYTHON,
            "ml/compare_dataset_sources.py",
            "--dataset",
            f"hf-reviewed-internal:data/album_covers_20_genres_reviewed:{rel(config.hf_report_dir / 'metrics.json')}",
            "--dataset",
            (
                "hf-model-on-musicbrainz-reviewed:"
                f"data/musicbrainz_cover_art_reviewed:{rel(config.hf_on_musicbrainz_report_dir / 'metrics.json')}"
            ),
            "--dataset",
            (
                "musicbrainz-reviewed-internal:"
                "data/musicbrainz_cover_art_reviewed:"
                "reports/datasets/musicbrainz/clip-mlp-reviewed-hierarchical/metrics.json"
            ),
            "--dataset",
            (
                "musicbrainz-model-on-hf-reviewed:"
                f"data/album_covers_20_genres_reviewed:{rel(config.musicbrainz_on_hf_report_dir / 'metrics.json')}"
            ),
            "--output",
            rel(config.comparison_markdown),
            "--json-output",
            rel(config.comparison_json),
        ]
    )


def pending_top3_broad_misses(examples_path: Path, reviews_path: Path) -> int:
    reviews = {row["image_path"] for row in read_csv(reviews_path)}
    count = 0
    for example in read_json(examples_path, []):
        image_path = example.get("imagePath", "")
        if image_path in reviews:
            continue
        actual = example.get("actual", "")
        top_3_labels = [prediction.get("label") for prediction in example.get("topPredictions", [])[:3]]
        if actual not in top_3_labels and not any(broad_genre(label) == broad_genre(actual) for label in top_3_labels):
            count += 1
    return count


def queue_counts(config: ReviewLoopConfig) -> dict[str, int]:
    return {
        "hf": pending_top3_broad_misses(config.hf_report_dir / "eval_examples.json", config.reviews),
        "musicbrainz": pending_top3_broad_misses(
            PROJECT_ROOT / "reports" / "datasets" / "musicbrainz" / "clip-mlp-reviewed-hierarchical" / "eval_examples.json",
            config.reviews,
        ),
    }


def current_metrics(config: ReviewLoopConfig) -> dict[str, Any]:
    keys = [
        "accuracy_top_1",
        "accuracy_top_3",
        "accuracy_broad_top_1",
        "accuracy_broad_top_3",
        "far_miss_rate",
        "near_miss_rate",
        "hierarchical_score",
    ]
    metrics = read_json(config.hf_report_dir / "metrics.json", {})
    return {key: metrics.get(key) for key in keys}


def run_loop(args: argparse.Namespace) -> dict[str, Any]:
    config = ReviewLoopConfig()
    result: dict[str, Any] = {}
    if args.apply_reviews or args.all:
        result["reviewed_metadata"] = apply_reviewed_labels(config)
    if args.refresh_cache or args.all:
        result["embedding_cache"] = refresh_embedding_labels(config)
    if args.train or args.all:
        train_hf_model(config, args.max_iter)
    if args.cross_eval or args.all:
        evaluate_cross_dataset(config)
    if args.compare or args.all:
        compare_datasets(config)
    if args.queue_counts or args.all:
        result["pending_top3_broad_misses"] = queue_counts(config)
    result["current_hf_metrics"] = current_metrics(config)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="Run apply, cache refresh, train, cross-eval, compare, and queue counts.")
    parser.add_argument("--apply-reviews", action="store_true", help="Write reviewed HF metadata from label_reviews.csv.")
    parser.add_argument("--refresh-cache", action="store_true", help="Relabel/filter the reviewed HF embedding cache.")
    parser.add_argument("--train", action="store_true", help="Retrain the serving HF reviewed hierarchical MLP model.")
    parser.add_argument("--cross-eval", action="store_true", help="Evaluate HF-on-MusicBrainz and MusicBrainz-on-HF.")
    parser.add_argument("--compare", action="store_true", help="Refresh the reviewed dataset comparison report.")
    parser.add_argument("--queue-counts", action="store_true", help="Report pending top-3 broad-miss review queues.")
    parser.add_argument("--max-iter", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not any(
        [
            args.all,
            args.apply_reviews,
            args.refresh_cache,
            args.train,
            args.cross_eval,
            args.compare,
            args.queue_counts,
        ]
    ):
        args.queue_counts = True
    print(json.dumps(run_loop(args), indent=2))


if __name__ == "__main__":
    main()
