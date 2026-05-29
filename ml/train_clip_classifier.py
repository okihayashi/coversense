#!/usr/bin/env python3
"""Train a CLIP-embedding genre classifier for album covers."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, top_k_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


DEFAULT_DATA_DIR = Path("data/album_covers_20_genres")
DEFAULT_MODEL_DIR = Path("models")
DEFAULT_REPORT_DIR = Path("reports")
DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"


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


def read_metadata(path: Path) -> tuple[list[Path], list[str]]:
    paths = []
    labels = []
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            paths.append(Path(row["image_path"]))
            labels.append(row["label"])
    if not paths:
        raise ValueError(f"No rows found in {path}")
    return paths, labels


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_images(paths: list[Path]) -> list[Image.Image]:
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    return images


def tensor_from_clip_output(output):
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "image_embeds"):
        return output.image_embeds
    if hasattr(output, "pooler_output"):
        return output.pooler_output
    if isinstance(output, (tuple, list)):
        return output[0]
    raise TypeError(f"Unsupported CLIP output type: {type(output)!r}")


@torch.inference_mode()
def encode_images(
    paths: list[Path],
    processor: CLIPProcessor,
    model: CLIPModel,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    embeddings = []
    for start in tqdm(range(0, len(paths), batch_size), desc="Encoding covers"):
        batch_paths = paths[start : start + batch_size]
        images = load_images(batch_paths)
        inputs = processor(images=images, return_tensors="pt", padding=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        image_features = model.get_image_features(**inputs)
        image_features = tensor_from_clip_output(image_features)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        embeddings.append(image_features.cpu().numpy())
    return np.vstack(embeddings)


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
    processor = CLIPProcessor.from_pretrained(args.clip_model)
    clip_model = CLIPModel.from_pretrained(args.clip_model).to(device)
    clip_model.eval()

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
        "label_encoder": label_encoder,
        "scaler": scaler,
        "classifier": classifier,
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
