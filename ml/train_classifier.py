#!/usr/bin/env python3
"""Train a swappable classifier head from cached album-cover embeddings."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, top_k_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC

try:
    from ml.broad_aware_classifier import BroadAwareClassifier, HierarchicalClassifier
except ModuleNotFoundError:  # pragma: no cover - supports running as python ml/train_classifier.py
    from broad_aware_classifier import BroadAwareClassifier, HierarchicalClassifier

from embeddings import DEFAULT_CLIP_MODEL, DEFAULT_EMBEDDING_DIR
from genre_taxonomy import broad_genre, error_type, hierarchy_metrics


DEFAULT_MODEL_DIR = Path("models")
DEFAULT_REPORT_DIR = Path("reports")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDING_DIR / "clip-vit-base-patch32.npz")
    parser.add_argument(
        "--classifier",
        choices=["logreg", "linear-svc", "random-forest", "mlp"],
        default="logreg",
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--clip-model", default=DEFAULT_CLIP_MODEL)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument(
        "--prediction-policy",
        choices=["exact", "broad-weighted", "broad-first", "hierarchical"],
        default="exact",
        help="Use broad-family-aware probabilities to reduce unrelated broad-family misses.",
    )
    return parser.parse_args()


def build_classifier(name: str, seed: int, max_iter: int):
    if name == "logreg":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=2.0,
                        class_weight="balanced",
                        max_iter=max_iter,
                        random_state=seed,
                    ),
                ),
            ]
        )
    if name == "linear-svc":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    CalibratedClassifierCV(
                        LinearSVC(C=1.0, class_weight="balanced", max_iter=max_iter, random_state=seed),
                        cv=3,
                    ),
                ),
            ]
        )
    if name == "random-forest":
        return RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )
    if name == "mlp":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    MLPClassifier(
                        hidden_layer_sizes=(512, 128),
                        alpha=0.001,
                        batch_size=256,
                        early_stopping=True,
                        max_iter=max_iter,
                        random_state=seed,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unsupported classifier: {name}")


def write_confusion_matrix(path: Path, labels: list[str], matrix: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["actual/predicted", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *row.tolist()])


def evaluation_examples(
    image_paths: np.ndarray,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    labels: list[str],
    top_k: int = 5,
) -> list[dict]:
    examples = []
    for image_path, actual_index, row in zip(image_paths, y_true, probabilities):
        ranked_indices = np.argsort(row)[::-1][:top_k]
        predicted_index = int(ranked_indices[0])
        top_3_labels = [labels[int(index)] for index in ranked_indices[:3]]
        actual_label = labels[int(actual_index)]
        actual_broad = broad_genre(actual_label)
        top_3_broad_correct = any(broad_genre(label) == actual_broad for label in top_3_labels)
        examples.append(
            {
                "imagePath": str(image_path),
                "actual": actual_label,
                "predicted": labels[predicted_index],
                "actualBroad": actual_broad,
                "predictedBroad": broad_genre(labels[predicted_index]),
                "correct": predicted_index == int(actual_index),
                "top3Correct": actual_label in top_3_labels,
                "top3Miss": actual_label not in top_3_labels,
                "top3BroadCorrect": top_3_broad_correct,
                "top3BroadMiss": not top_3_broad_correct,
                "errorType": error_type(actual_label, labels[predicted_index]),
                "confidence": float(row[predicted_index]),
                "topPredictions": [
                    {"label": labels[int(index)], "probability": float(row[int(index)])}
                    for index in ranked_indices
                ],
            }
        )
    return examples


def main() -> None:
    args = parse_args()
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    cache = np.load(args.embeddings, allow_pickle=False)
    x = cache["embeddings"]
    labels = cache["labels"]
    image_paths = cache["image_paths"]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    indices = np.arange(len(y))
    train_idx, test_idx, y_train, y_test = train_test_split(
        indices,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    classifier = build_classifier(args.classifier, args.seed, args.max_iter)
    classifier.fit(x[train_idx], y_train)
    labels_out = label_encoder.classes_.tolist()
    if args.prediction_policy in {"broad-weighted", "broad-first"}:
        classifier = BroadAwareClassifier(classifier, labels_out, args.prediction_policy)
    elif args.prediction_policy == "hierarchical":
        broad_label_encoder = LabelEncoder()
        broad_labels = np.array([broad_genre(label) for label in labels])
        y_broad = broad_label_encoder.fit_transform(broad_labels)
        broad_classifier = clone(build_classifier(args.classifier, args.seed, args.max_iter))
        broad_classifier.fit(x[train_idx], y_broad[train_idx])
        classifier = HierarchicalClassifier(
            classifier,
            broad_classifier,
            labels_out,
            broad_label_encoder.classes_.tolist(),
        )

    predictions = classifier.predict(x[test_idx])
    probabilities = classifier.predict_proba(x[test_idx])
    classifier_name = args.classifier if args.prediction_policy == "exact" else f"{args.classifier}-{args.prediction_policy}"

    metrics = {
        "clip_model": args.clip_model,
        "classifier": classifier_name,
        "base_classifier": args.classifier,
        "prediction_policy": args.prediction_policy,
        "embedding_cache": args.embeddings.as_posix(),
        "train_size": int(len(train_idx)),
        "test_size": int(len(test_idx)),
        "accuracy_top_1": accuracy_score(y_test, predictions),
        "accuracy_top_3": top_k_accuracy_score(y_test, probabilities, k=3, labels=np.arange(len(labels_out))),
        "labels": labels_out,
    }
    metrics.update(hierarchy_metrics(y_test, predictions, probabilities, labels_out))

    payload = {
        "clip_model": args.clip_model,
        "classifier_name": classifier_name,
        "base_classifier": args.classifier,
        "prediction_policy": args.prediction_policy,
        "label_encoder": label_encoder,
        "classifier": classifier,
    }
    model_path = args.model_dir / "coversense_clip_classifier.joblib"
    joblib.dump(payload, model_path)

    (args.model_dir / "coversense_labels.json").write_text(json.dumps(labels_out, indent=2), encoding="utf-8")
    (args.report_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.report_dir / "classification_report.txt").write_text(
        classification_report(y_test, predictions, target_names=labels_out),
        encoding="utf-8",
    )
    write_confusion_matrix(args.report_dir / "confusion_matrix.csv", labels_out, confusion_matrix(y_test, predictions))
    examples = evaluation_examples(image_paths[test_idx], y_test, probabilities, labels_out)
    failures = [example for example in examples if not example["correct"]]
    (args.report_dir / "eval_examples.json").write_text(json.dumps(examples, indent=2), encoding="utf-8")
    (args.report_dir / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    np.savez_compressed(
        args.report_dir / "split_indices.npz",
        train_idx=train_idx,
        test_idx=test_idx,
        test_image_paths=image_paths[test_idx],
    )

    print(json.dumps(metrics, indent=2))
    print(f"Saved model: {model_path}")


if __name__ == "__main__":
    main()
