#!/usr/bin/env python3
"""Evaluate a trained CLIP classifier on any compatible embedding cache."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, top_k_accuracy_score

from genre_taxonomy import broad_genre, error_type, hierarchy_metrics
from train_classifier import evaluation_examples, write_confusion_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/coversense_clip_classifier.joblib"))
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    return parser.parse_args()


def classifier_name(payload: dict) -> str:
    return payload.get("classifier_name") or payload.get("classifier") or "unknown"


def write_mismatched_labels(path: Path, unknown_labels: list[str], known_labels: list[str]) -> None:
    payload = {
        "unknown_labels": unknown_labels,
        "known_labels": known_labels,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)

    payload = joblib.load(args.model)
    classifier = payload["classifier"]
    label_encoder = payload["label_encoder"]
    labels_out = label_encoder.classes_.tolist()
    label_to_index = {label: index for index, label in enumerate(labels_out)}

    cache = np.load(args.embeddings, allow_pickle=False)
    x = cache["embeddings"]
    labels = cache["labels"].astype(str)
    image_paths = cache["image_paths"].astype(str)

    unknown_labels = sorted(set(labels) - set(label_to_index))
    if unknown_labels:
        write_mismatched_labels(args.report_dir / "mismatched_labels.json", unknown_labels, labels_out)
        raise ValueError(f"Embedding cache contains labels missing from the model: {', '.join(unknown_labels)}")

    y_true = np.array([label_to_index[label] for label in labels])
    probabilities = classifier.predict_proba(x)
    predictions = np.argmax(probabilities, axis=1)

    metrics = {
        "source_model": args.model.as_posix(),
        "classifier": classifier_name(payload),
        "embedding_cache": args.embeddings.as_posix(),
        "test_size": int(len(y_true)),
        "accuracy_top_1": accuracy_score(y_true, predictions),
        "accuracy_top_3": top_k_accuracy_score(y_true, probabilities, k=3, labels=np.arange(len(labels_out))),
        "labels": labels_out,
    }
    metrics.update(hierarchy_metrics(y_true, predictions, probabilities, labels_out))

    (args.report_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.report_dir / "classification_report.txt").write_text(
        classification_report(y_true, predictions, labels=range(len(labels_out)), target_names=labels_out, zero_division=0),
        encoding="utf-8",
    )
    write_confusion_matrix(
        args.report_dir / "confusion_matrix.csv",
        labels_out,
        confusion_matrix(y_true, predictions, labels=range(len(labels_out))),
    )
    examples = evaluation_examples(image_paths, y_true, probabilities, labels_out)
    failures = [example for example in examples if not example["correct"]]
    (args.report_dir / "eval_examples.json").write_text(json.dumps(examples, indent=2), encoding="utf-8")
    (args.report_dir / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")

    with (args.report_dir / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["actual", "predicted", "actual_broad", "predicted_broad", "confidence", "correct", "error_type"])
        for example in examples:
            writer.writerow(
                [
                    example["actual"],
                    example["predicted"],
                    broad_genre(example["actual"]),
                    broad_genre(example["predicted"]),
                    example["confidence"],
                    example["correct"],
                    error_type(example["actual"], example["predicted"]),
                ]
            )

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
