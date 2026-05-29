"""Shared CLIP embedding utilities for CoverSense."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


DEFAULT_DATA_DIR = Path("data/album_covers_20_genres")
DEFAULT_EMBEDDING_DIR = Path("embeddings")
DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"


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


def load_clip(model_name: str, device: torch.device) -> tuple[CLIPProcessor, CLIPModel]:
    processor = CLIPProcessor.from_pretrained(model_name)
    clip_model = CLIPModel.from_pretrained(model_name).to(device)
    clip_model.eval()
    return processor, clip_model


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
