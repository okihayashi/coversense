#!/usr/bin/env python3
"""Build a reusable CLIP embedding cache for album covers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from embeddings import (
    DEFAULT_CLIP_MODEL,
    DEFAULT_DATA_DIR,
    DEFAULT_EMBEDDING_DIR,
    choose_device,
    encode_images,
    load_clip,
    read_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_DATA_DIR / "metadata.csv")
    parser.add_argument("--output", type=Path, default=DEFAULT_EMBEDDING_DIR / "clip-vit-base-patch32.npz")
    parser.add_argument("--clip-model", default=DEFAULT_CLIP_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    image_paths, labels = read_metadata(args.metadata)
    device = choose_device()
    print(f"Using device: {device}")
    processor, clip_model = load_clip(args.clip_model, device)
    embeddings = encode_images(image_paths, processor, clip_model, device, args.batch_size)

    np.savez_compressed(
        args.output,
        embeddings=embeddings.astype(np.float32),
        labels=np.array(labels),
        image_paths=np.array([path.as_posix() for path in image_paths]),
    )
    metadata = {
        "clip_model": args.clip_model,
        "embedding_shape": list(embeddings.shape),
        "count": len(labels),
        "metadata_csv": args.metadata.as_posix(),
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
