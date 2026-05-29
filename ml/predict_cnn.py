#!/usr/bin/env python3
"""Predict genre probabilities with the raw-pixel CNN model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from train_cnn import CnnConfig, CoverCnn, image_to_tensor, resolve_device, square_resize


DEFAULT_MODEL_PATH = Path("models/coversense_cnn.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = torch.load(args.model, map_location=device)
    config = CnnConfig(**checkpoint["model_config"])
    model = CoverCnn(config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    with Image.open(args.image) as image:
        image = square_resize(image.convert("RGB"), config.image_size, augment=False)
        tensor = image_to_tensor(image).unsqueeze(0).to(device)

    probabilities = model(tensor).softmax(dim=1).squeeze(0).cpu()
    labels = checkpoint["labels"]
    ranked = sorted(zip(labels, probabilities.tolist()), key=lambda item: item[1], reverse=True)

    for label, probability in ranked[: args.top_k]:
        print(f"{label}: {probability:.3f}")


if __name__ == "__main__":
    main()
