#!/usr/bin/env python3
"""Train a small CNN directly on album-cover pixels."""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, top_k_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from embeddings import DEFAULT_DATA_DIR, choose_device
from train_classifier import DEFAULT_MODEL_DIR, DEFAULT_REPORT_DIR, write_confusion_matrix


IMAGE_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGE_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


@dataclass
class CnnConfig:
    image_size: int
    num_classes: int
    channels: tuple[int, ...] = (32, 64, 128, 256)
    dropout: float = 0.35


class AlbumCoverDataset(Dataset):
    def __init__(
        self,
        paths: list[Path],
        targets: np.ndarray,
        image_size: int,
        augment: bool = False,
    ) -> None:
        self.paths = paths
        self.targets = targets
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        with Image.open(self.paths[index]) as image:
            image = image.convert("RGB")
            image = square_resize(image, self.image_size, self.augment)
            if self.augment:
                image = jitter_color(image)
            tensor = image_to_tensor(image)
        return tensor, torch.tensor(self.targets[index], dtype=torch.long)


class CoverCnn(nn.Module):
    def __init__(self, config: CnnConfig) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = 3
        for out_channels in config.channels:
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2),
                ]
            )
            in_channels = out_channels

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(config.dropout),
            nn.Linear(config.channels[-1], config.num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples-per-label", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
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


def limit_per_label(paths: list[Path], labels: list[str], max_per_label: int | None) -> tuple[list[Path], list[str]]:
    if max_per_label is None:
        return paths, labels
    counts: dict[str, int] = {}
    limited_paths = []
    limited_labels = []
    for path, label in zip(paths, labels):
        count = counts.get(label, 0)
        if count >= max_per_label:
            continue
        counts[label] = count + 1
        limited_paths.append(path)
        limited_labels.append(label)
    return limited_paths, limited_labels


def square_resize(image: Image.Image, image_size: int, augment: bool) -> Image.Image:
    if not augment:
        return ImageOps.fit(image, (image_size, image_size), method=Image.Resampling.BICUBIC)

    width, height = image.size
    crop_scale = random.uniform(0.86, 1.0)
    crop_size = int(min(width, height) * crop_scale)
    if crop_size < 1:
        return ImageOps.fit(image, (image_size, image_size), method=Image.Resampling.BICUBIC)

    left = random.randint(0, max(0, width - crop_size))
    top = random.randint(0, max(0, height - crop_size))
    image = image.crop((left, top, left + crop_size, top + crop_size))
    return image.resize((image_size, image_size), Image.Resampling.BICUBIC)


def jitter_color(image: Image.Image) -> Image.Image:
    for enhancer_class in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        image = enhancer_class(image).enhance(random.uniform(0.85, 1.15))
    return image


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return (tensor - IMAGE_MEAN) / IMAGE_STD


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return choose_device()
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return device


def topk_correct(logits: torch.Tensor, targets: torch.Tensor, k: int) -> int:
    topk = logits.topk(k, dim=1).indices
    return topk.eq(targets.view(-1, 1)).any(dim=1).sum().item()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_top1 = 0
    total_top3 = 0
    seen = 0
    for images, targets in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_top1 += (logits.argmax(dim=1) == targets).sum().item()
        total_top3 += topk_correct(logits, targets, min(3, logits.size(1)))
        seen += batch_size

    return {
        "loss": total_loss / seen,
        "accuracy_top_1": total_top1 / seen,
        "accuracy_top_3": total_top3 / seen,
    }


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> dict[str, object]:
    model.eval()
    total_loss = 0.0
    probabilities = []
    predictions = []
    targets_out = []
    seen = 0
    for images, targets in tqdm(loader, desc="Evaluating", leave=False):
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images)
        loss = criterion(logits, targets)
        probs = logits.softmax(dim=1)

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        probabilities.append(probs.cpu().numpy())
        predictions.append(logits.argmax(dim=1).cpu().numpy())
        targets_out.append(targets.cpu().numpy())
        seen += batch_size

    y_true = np.concatenate(targets_out)
    y_pred = np.concatenate(predictions)
    y_prob = np.vstack(probabilities)
    return {
        "loss": total_loss / seen,
        "accuracy_top_1": accuracy_score(y_true, y_pred),
        "accuracy_top_3": top_k_accuracy_score(
            y_true,
            y_prob,
            k=min(3, y_prob.shape[1]),
            labels=np.arange(y_prob.shape[1]),
        ),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    paths, labels = read_metadata(args.data_dir / "metadata.csv")
    paths, labels = limit_per_label(paths, labels, args.max_samples_per_label)

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

    train_paths = [paths[index] for index in train_idx]
    test_paths = [paths[index] for index in test_idx]
    train_loader = DataLoader(
        AlbumCoverDataset(train_paths, y_train, args.image_size, augment=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        AlbumCoverDataset(test_paths, y_test, args.image_size, augment=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    device = resolve_device(args.device)
    config = CnnConfig(image_size=args.image_size, num_classes=len(label_encoder.classes_))
    model = CoverCnn(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_state = None
    best_top1 = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        eval_metrics = evaluate(model, test_loader, criterion, device)
        epoch_metrics = {
            "epoch": epoch,
            "train": train_metrics,
            "test": {key: value for key, value in eval_metrics.items() if not key.startswith("y_")},
        }
        history.append(epoch_metrics)
        print(json.dumps(epoch_metrics, indent=2))
        if eval_metrics["accuracy_top_1"] > best_top1:
            best_top1 = float(eval_metrics["accuracy_top_1"])
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")

    model.load_state_dict(best_state)
    final_metrics = evaluate(model, test_loader, criterion, device)
    labels_out = label_encoder.classes_.tolist()
    metrics = {
        "model": "small-cnn",
        "train_size": int(len(train_idx)),
        "test_size": int(len(test_idx)),
        "image_size": args.image_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "device": device.type,
        "accuracy_top_1": final_metrics["accuracy_top_1"],
        "accuracy_top_3": final_metrics["accuracy_top_3"],
        "labels": labels_out,
        "history": history,
    }

    model_path = args.model_dir / "coversense_cnn.pt"
    torch.save(
        {
            "model_name": "small-cnn",
            "model_config": asdict(config),
            "state_dict": best_state,
            "labels": labels_out,
            "metrics": metrics,
        },
        model_path,
    )

    (args.report_dir / "cnn_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.report_dir / "cnn_classification_report.txt").write_text(
        classification_report(final_metrics["y_true"], final_metrics["y_pred"], target_names=labels_out, zero_division=0),
        encoding="utf-8",
    )
    write_confusion_matrix(
        args.report_dir / "cnn_confusion_matrix.csv",
        labels_out,
        confusion_matrix(final_metrics["y_true"], final_metrics["y_pred"]),
    )

    print(json.dumps({key: value for key, value in metrics.items() if key != "history"}, indent=2))
    print(f"Saved model: {model_path}")


if __name__ == "__main__":
    main()
