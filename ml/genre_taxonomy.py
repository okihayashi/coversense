"""Genre hierarchy helpers for evaluation and hierarchy-aware training."""

from __future__ import annotations

import numpy as np


BROAD_GENRES = {
    "blues": "roots",
    "classical": "classical",
    "country": "roots",
    "deathmetal": "metal",
    "doommetal": "metal",
    "drumnbass": "electronic",
    "electronic": "electronic",
    "folk": "roots",
    "grime": "hiphop",
    "heavymetal": "metal",
    "hiphop": "hiphop",
    "jazz": "jazz",
    "lofi": "electronic",
    "pop": "pop-soul",
    "psychedelicrock": "rock",
    "punk": "rock",
    "reggae": "roots",
    "rock": "rock",
    "soul": "pop-soul",
    "techno": "electronic",
}


def broad_genre(label: str) -> str:
    return BROAD_GENRES.get(label, label)


def error_type(actual: str, predicted: str) -> str:
    if actual == predicted:
        return "exact"
    if broad_genre(actual) == broad_genre(predicted):
        return "near_miss"
    return "far_miss"


def hierarchy_metrics(y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray, labels: list[str]) -> dict:
    actual_labels = [labels[int(index)] for index in y_true]
    predicted_labels = [labels[int(index)] for index in y_pred]
    exact = np.array([actual == predicted for actual, predicted in zip(actual_labels, predicted_labels)])
    broad = np.array(
        [broad_genre(actual) == broad_genre(predicted) for actual, predicted in zip(actual_labels, predicted_labels)]
    )
    near = np.logical_and(~exact, broad)
    far = ~broad
    top3_broad = []
    for actual, row in zip(actual_labels, probabilities):
        ranked = np.argsort(row)[::-1][: min(3, len(labels))]
        actual_broad = broad_genre(actual)
        top3_broad.append(any(broad_genre(labels[int(index)]) == actual_broad for index in ranked))

    total = len(actual_labels)
    return {
        "accuracy_broad_top_1": float(broad.mean()),
        "accuracy_broad_top_3": float(np.mean(top3_broad)),
        "near_miss_rate": float(near.mean()),
        "far_miss_rate": float(far.mean()),
        "hierarchical_score": float((exact.sum() + 0.5 * near.sum()) / total),
        "genre_hierarchy": BROAD_GENRES,
    }


def sibling_distribution(labels: list[str]) -> np.ndarray:
    distribution = np.zeros((len(labels), len(labels)), dtype=np.float32)
    broad_labels = [broad_genre(label) for label in labels]
    for row, broad in enumerate(broad_labels):
        siblings = [index for index, other_broad in enumerate(broad_labels) if other_broad == broad and index != row]
        if siblings:
            distribution[row, siblings] = 1.0 / len(siblings)
    return distribution
