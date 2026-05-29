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
from genre_taxonomy import broad_genre, error_type, hierarchy_metrics, sibling_distribution
from train_classifier import DEFAULT_MODEL_DIR, DEFAULT_REPORT_DIR, write_confusion_matrix


IMAGE_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGE_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


@dataclass
class CnnConfig:
    image_size: int
    num_classes: int
    channels: tuple[int, ...] = (32, 64, 128, 256)
    dropout: float = 0.35


@dataclass
class TrainingSettings:
    image_size: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    dropout: float
    base_channels: int


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
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples-per-label", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--trial-epochs", type=int, default=None)
    parser.add_argument(
        "--sibling-smoothing",
        type=float,
        default=0.15,
        help="Probability mass assigned to sibling labels in the same broad genre during CNN training.",
    )
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


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return -(targets * logits.log_softmax(dim=1)).sum(dim=1).mean()


def channels_for(base_channels: int) -> tuple[int, ...]:
    return (base_channels, base_channels * 2, base_channels * 4, base_channels * 8)


def default_settings(args: argparse.Namespace) -> TrainingSettings:
    return TrainingSettings(
        image_size=args.image_size,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        base_channels=args.base_channels,
    )


def sample_settings(args: argparse.Namespace, rng: random.Random) -> TrainingSettings:
    if args.trials <= 1:
        return default_settings(args)

    image_size = rng.choice([96, 128, 160])
    batch_size = rng.choice([32, 64])
    base_channels = rng.choice([16, 24, 32])
    learning_rate = 10 ** rng.uniform(-4.2, -3.0)
    weight_decay = 10 ** rng.uniform(-5.0, -3.0)
    dropout = rng.uniform(0.2, 0.5)
    return TrainingSettings(
        image_size=image_size,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        dropout=dropout,
        base_channels=base_channels,
    )


def dataset_for_indices(
    paths: list[Path],
    targets: np.ndarray,
    indices: np.ndarray,
    image_size: int,
    augment: bool,
) -> AlbumCoverDataset:
    return AlbumCoverDataset([paths[index] for index in indices], targets[indices], image_size, augment=augment)


def loader_for_dataset(
    dataset: AlbumCoverDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    sibling_targets: torch.Tensor | None = None,
    sibling_smoothing: float = 0.0,
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
        if sibling_targets is not None and sibling_smoothing > 0:
            sibling_batch = sibling_targets[targets]
            hard_targets = torch.nn.functional.one_hot(targets, num_classes=logits.size(1)).float()
            soft_targets = hard_targets * (1 - sibling_smoothing) + sibling_batch * sibling_smoothing
            no_siblings = sibling_batch.sum(dim=1) == 0
            soft_targets[no_siblings] = hard_targets[no_siblings]
            loss = soft_cross_entropy(logits, soft_targets)
        else:
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
        "probabilities": y_prob,
    }


def train_candidate(
    paths: list[Path],
    y: np.ndarray,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    settings: TrainingSettings,
    epochs: int,
    num_classes: int,
    device: torch.device,
    args: argparse.Namespace,
    desc: str,
    sibling_targets: torch.Tensor | None,
) -> dict[str, object]:
    config = CnnConfig(
        image_size=settings.image_size,
        num_classes=num_classes,
        channels=channels_for(settings.base_channels),
        dropout=settings.dropout,
    )
    model = CoverCnn(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    train_loader = loader_for_dataset(
        dataset_for_indices(paths, y, train_idx, settings.image_size, augment=True),
        settings.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    eval_loader = loader_for_dataset(
        dataset_for_indices(paths, y, eval_idx, settings.image_size, augment=False),
        settings.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    best_state = None
    best_metrics = None
    best_top1 = -1.0
    history = []
    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            sibling_targets=sibling_targets,
            sibling_smoothing=args.sibling_smoothing,
        )
        eval_metrics = evaluate(model, eval_loader, criterion, device)
        epoch_metrics = {
            "phase": desc,
            "epoch": epoch,
            "train": train_metrics,
            "eval": {key: value for key, value in eval_metrics.items() if key not in {"y_true", "y_pred", "probabilities"}},
        }
        history.append(epoch_metrics)
        print(json.dumps(epoch_metrics, indent=2))
        if eval_metrics["accuracy_top_1"] > best_top1:
            best_top1 = float(eval_metrics["accuracy_top_1"])
            best_metrics = eval_metrics
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_state is None or best_metrics is None:
        raise RuntimeError("Training did not produce a checkpoint")

    model.load_state_dict(best_state)
    final_metrics = evaluate(model, eval_loader, criterion, device)
    return {
        "config": config,
        "settings": settings,
        "state_dict": best_state,
        "history": history,
        "metrics": final_metrics,
    }


def evaluation_examples(
    image_paths: list[Path],
    y_true: np.ndarray,
    probabilities: np.ndarray,
    labels: list[str],
    top_k: int = 5,
) -> list[dict]:
    examples = []
    for image_path, actual_index, row in zip(image_paths, y_true, probabilities):
        ranked_indices = np.argsort(row)[::-1][:top_k]
        predicted_index = int(ranked_indices[0])
        examples.append(
            {
                "imagePath": image_path.as_posix(),
                "actual": labels[int(actual_index)],
                "predicted": labels[predicted_index],
                "actualBroad": broad_genre(labels[int(actual_index)]),
                "predictedBroad": broad_genre(labels[predicted_index]),
                "correct": predicted_index == int(actual_index),
                "errorType": error_type(labels[int(actual_index)], labels[predicted_index]),
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
    train_idx, test_idx, y_train_pool, _ = train_test_split(
        indices,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )
    validation_size = max(args.validation_size, len(label_encoder.classes_) / len(train_idx))
    inner_train_idx, validation_idx, _, _ = train_test_split(
        train_idx,
        y_train_pool,
        test_size=validation_size,
        random_state=args.seed,
        stratify=y_train_pool,
    )

    device = resolve_device(args.device)
    trial_epochs = args.trial_epochs or max(1, min(args.epochs, 3))
    labels_out = label_encoder.classes_.tolist()
    sibling_targets = torch.from_numpy(sibling_distribution(labels_out)).to(device)
    rng = random.Random(args.seed)
    trials = []
    best_trial = None
    best_top1 = -1.0

    for trial in range(1, args.trials + 1):
        settings = sample_settings(args, rng)
        print(
            json.dumps(
                {
                    "trial": trial,
                    "phase": "tuning" if args.trials > 1 else "single",
                    "epochs": trial_epochs,
                    "settings": asdict(settings),
                },
                indent=2,
            )
        )
        result = train_candidate(
            paths=paths,
            y=y,
            train_idx=inner_train_idx if args.trials > 1 else train_idx,
            eval_idx=validation_idx if args.trials > 1 else test_idx,
            settings=settings,
            epochs=trial_epochs if args.trials > 1 else args.epochs,
            num_classes=len(labels_out),
            device=device,
            args=args,
            desc=f"trial-{trial}",
            sibling_targets=sibling_targets,
        )
        trial_summary = {
            "trial": trial,
            "settings": asdict(settings),
            "validation_accuracy_top_1": result["metrics"]["accuracy_top_1"],
            "validation_accuracy_top_3": result["metrics"]["accuracy_top_3"],
            "validation_loss": result["metrics"]["loss"],
        }
        trials.append(trial_summary)
        if result["metrics"]["accuracy_top_1"] > best_top1:
            best_top1 = float(result["metrics"]["accuracy_top_1"])
            best_trial = result

    if best_trial is None:
        raise RuntimeError("Training did not produce a checkpoint")

    if args.trials > 1:
        best_settings = best_trial["settings"]
        print(json.dumps({"phase": "final", "epochs": args.epochs, "settings": asdict(best_settings)}, indent=2))
        final_result = train_candidate(
            paths=paths,
            y=y,
            train_idx=train_idx,
            eval_idx=test_idx,
            settings=best_settings,
            epochs=args.epochs,
            num_classes=len(labels_out),
            device=device,
            args=args,
            desc="final",
            sibling_targets=sibling_targets,
        )
    else:
        final_result = best_trial

    config = final_result["config"]
    best_state = final_result["state_dict"]
    final_metrics = final_result["metrics"]
    best_settings = final_result["settings"]
    labels_out = label_encoder.classes_.tolist()
    metrics = {
        "model": "small-cnn",
        "tuning": {
            "trials": args.trials,
            "trial_epochs": trial_epochs,
            "validation_size": validation_size,
            "trial_results": trials,
            "best_settings": asdict(best_settings),
        },
        "train_size": int(len(train_idx)),
        "test_size": int(len(test_idx)),
        "validation_size": int(len(validation_idx)) if args.trials > 1 else 0,
        "image_size": best_settings.image_size,
        "epochs": args.epochs,
        "batch_size": best_settings.batch_size,
        "learning_rate": best_settings.learning_rate,
        "weight_decay": best_settings.weight_decay,
        "dropout": best_settings.dropout,
        "base_channels": best_settings.base_channels,
        "sibling_smoothing": args.sibling_smoothing,
        "device": device.type,
        "accuracy_top_1": final_metrics["accuracy_top_1"],
        "accuracy_top_3": final_metrics["accuracy_top_3"],
        "labels": labels_out,
        "history": final_result["history"],
    }
    metrics.update(
        hierarchy_metrics(
            final_metrics["y_true"],
            final_metrics["y_pred"],
            final_metrics["probabilities"],
            labels_out,
        )
    )

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
    examples = evaluation_examples(
        [paths[index] for index in test_idx],
        final_metrics["y_true"],
        final_metrics["probabilities"],
        labels_out,
    )
    failures = [example for example in examples if not example["correct"]]
    (args.report_dir / "cnn_eval_examples.json").write_text(json.dumps(examples, indent=2), encoding="utf-8")
    (args.report_dir / "cnn_failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")

    print(json.dumps({key: value for key, value in metrics.items() if key != "history"}, indent=2))
    print(f"Saved model: {model_path}")


if __name__ == "__main__":
    main()
