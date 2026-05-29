#!/usr/bin/env python3
"""Predict genre probabilities for one album cover image."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


DEFAULT_MODEL_PATH = Path("models/coversense_clip_classifier.joblib")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    payload = joblib.load(args.model)

    device = choose_device()
    processor = CLIPProcessor.from_pretrained(payload["clip_model"])
    clip_model = CLIPModel.from_pretrained(payload["clip_model"]).to(device)
    clip_model.eval()

    with Image.open(args.image) as image:
        inputs = processor(images=[image.convert("RGB")], return_tensors="pt", padding=True)

    inputs = {key: value.to(device) for key, value in inputs.items()}
    features = clip_model.get_image_features(**inputs)
    features = features / features.norm(dim=-1, keepdim=True)
    features = payload["scaler"].transform(features.cpu().numpy())

    probabilities = payload["classifier"].predict_proba(features)[0]
    labels = payload["label_encoder"].classes_
    ranked = sorted(zip(labels, probabilities), key=lambda item: item[1], reverse=True)

    for label, probability in ranked[: args.top_k]:
        print(f"{label}: {probability:.3f}")


if __name__ == "__main__":
    main()
