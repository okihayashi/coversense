#!/usr/bin/env python3
"""Merge compatible album-cover embedding caches into one training cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, required=True, help="Embedding .npz cache to merge.")
    parser.add_argument("--output", type=Path, required=True, help="Merged embedding .npz output path.")
    return parser.parse_args()


def cache_name(path: Path) -> str:
    stem = path.stem.replace("-clip-vit-base-patch32", "")
    return stem or path.stem


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    embeddings = []
    labels = []
    image_paths = []
    sources = []
    embedding_dim = None

    for path in args.input:
        cache = np.load(path, allow_pickle=False)
        cache_embeddings = cache["embeddings"]
        cache_labels = cache["labels"].astype(str)
        cache_image_paths = cache["image_paths"].astype(str)

        if embedding_dim is None:
            embedding_dim = cache_embeddings.shape[1]
        elif cache_embeddings.shape[1] != embedding_dim:
            raise ValueError(f"{path} has embedding dimension {cache_embeddings.shape[1]}, expected {embedding_dim}")
        if len(cache_embeddings) != len(cache_labels) or len(cache_labels) != len(cache_image_paths):
            raise ValueError(f"{path} has inconsistent embeddings, labels, and image path lengths")

        source = cache_name(path)
        embeddings.append(cache_embeddings.astype(np.float32))
        labels.append(cache_labels)
        image_paths.append(cache_image_paths)
        sources.append(np.full(len(cache_labels), source))

    merged_embeddings = np.concatenate(embeddings, axis=0)
    merged_labels = np.concatenate(labels, axis=0)
    merged_image_paths = np.concatenate(image_paths, axis=0)
    merged_sources = np.concatenate(sources, axis=0)

    np.savez_compressed(
        args.output,
        embeddings=merged_embeddings,
        labels=merged_labels,
        image_paths=merged_image_paths,
        sources=merged_sources,
    )

    metadata = {
        "count": int(len(merged_labels)),
        "embedding_shape": list(merged_embeddings.shape),
        "inputs": [path.as_posix() for path in args.input],
        "source_counts": {
            source: int(np.sum(merged_sources == source)) for source in sorted(set(merged_sources.tolist()))
        },
        "label_counts": {
            label: int(np.sum(merged_labels == label)) for label in sorted(set(merged_labels.tolist()))
        },
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
