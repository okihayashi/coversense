#!/usr/bin/env python3
"""Train a CLIP-embedding genre classifier for album covers.

This script keeps the original one-command behavior. For faster experiments,
prefer `build_embeddings.py` once, then `train_classifier.py` many times.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, top_k_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from embeddings import DEFAULT_CLIP_MODEL, choose_device, encode_images, load_clip, read_metadata


DEFAULT_DATA_DIR = Path("data/album_covers_20_genres")
DEFAULT_MODEL_DIR = Path("models")
DEFAULT_REPORT_DIR = Path("reports")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_DATA_DIR / "metadata.csv")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--clip-model", default=DEFAULT_CLIP_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=2000)
    return parser.parse_args()


def write_confusion_matrix(path: Path, labels: list[str], matrix: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["actual/predicted", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *row.tolist()])


def main() -> None:
    args = parse_args()
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    image_paths, labels = read_metadata(args.metadata)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    train_paths, test_paths, y_train, y_test = train_test_split(
        image_paths,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    device = choose_device()
    print(f"Using device: {device}")
    processor, clip_model = load_clip(args.clip_model, device)

    x_train = encode_images(train_paths, processor, clip_model, device, args.batch_size)
    x_test = encode_images(test_paths, processor, clip_model, device, args.batch_size)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    classifier = LogisticRegression(
        C=2.0,
        class_weight="balanced",
        max_iter=args.max_iter,
        n_jobs=-1,
        random_state=args.seed,
    )
    classifier.fit(x_train_scaled, y_train)

    predictions = classifier.predict(x_test_scaled)
    probabilities = classifier.predict_proba(x_test_scaled)
    labels_out = label_encoder.classes_.tolist()

    metrics = {
        "clip_model": args.clip_model,
        "train_size": len(train_paths),
        "test_size": len(test_paths),
        "accuracy_top_1": accuracy_score(y_test, predictions),
        "accuracy_top_3": top_k_accuracy_score(y_test, probabilities, k=3, labels=np.arange(len(labels_out))),
        "labels": labels_out,
    }

    model_payload = {
        "clip_model": args.clip_model,
        "classifier_name": "logreg",
        "label_encoder": label_encoder,
        "classifier": {
            "scaler": scaler,
            "classifier": classifier,
        },
    }
    joblib.dump(model_payload, args.model_dir / "coversense_clip_classifier.joblib")
    (args.model_dir / "coversense_labels.json").write_text(json.dumps(labels_out, indent=2), encoding="utf-8")
    (args.report_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.report_dir / "classification_report.txt").write_text(
        classification_report(y_test, predictions, target_names=labels_out),
        encoding="utf-8",
    )
    write_confusion_matrix(args.report_dir / "confusion_matrix.csv", labels_out, confusion_matrix(y_test, predictions))

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
