"""Prediction wrappers that bias exact-genre heads toward broad-family consistency."""

from __future__ import annotations

import numpy as np

try:
    from ml.genre_taxonomy import broad_genre
except ModuleNotFoundError:  # pragma: no cover - supports running scripts from ml/
    from genre_taxonomy import broad_genre


class BroadAwareClassifier:
    """Wrap a probabilistic classifier with broad-family-aware probabilities.

    The wrapped classifier still learns exact genres. At inference time we use
    the broad genre probability mass to reduce confident jumps across unrelated
    families, which is the main source of far misses.
    """

    def __init__(self, base_classifier, labels: list[str], policy: str = "broad-weighted"):
        if policy not in {"broad-weighted", "broad-first"}:
            raise ValueError(f"Unsupported broad-aware policy: {policy}")
        self.base_classifier = base_classifier
        self.labels = list(labels)
        self.policy = policy
        self.classes_ = np.arange(len(self.labels))
        self._broad_labels = [broad_genre(label) for label in self.labels]
        self._broad_families = sorted(set(self._broad_labels))

    def predict_proba(self, x):
        probabilities = np.asarray(self.base_classifier.predict_proba(x), dtype=np.float64)
        adjusted = np.zeros_like(probabilities)

        for family in self._broad_families:
            family_indices = [index for index, broad in enumerate(self._broad_labels) if broad == family]
            family_mass = probabilities[:, family_indices].sum(axis=1, keepdims=True)
            if self.policy == "broad-weighted":
                adjusted[:, family_indices] = probabilities[:, family_indices] * family_mass

        if self.policy == "broad-first":
            broad_scores = np.column_stack(
                [
                    probabilities[:, [index for index, broad in enumerate(self._broad_labels) if broad == family]].sum(axis=1)
                    for family in self._broad_families
                ]
            )
            winning_families = np.argmax(broad_scores, axis=1)
            for row_index, family_index in enumerate(winning_families):
                family = self._broad_families[int(family_index)]
                family_indices = [index for index, broad in enumerate(self._broad_labels) if broad == family]
                adjusted[row_index, family_indices] = probabilities[row_index, family_indices]

        row_sums = adjusted.sum(axis=1, keepdims=True)
        return np.divide(adjusted, row_sums, out=np.zeros_like(adjusted), where=row_sums != 0)

    def predict(self, x):
        return np.argmax(self.predict_proba(x), axis=1)


class HierarchicalClassifier:
    """Combine a broad-family classifier with an exact-genre classifier."""

    def __init__(self, exact_classifier, broad_classifier, labels: list[str], broad_classes: list[str]):
        self.exact_classifier = exact_classifier
        self.broad_classifier = broad_classifier
        self.labels = list(labels)
        self.broad_classes = list(broad_classes)
        self.classes_ = np.arange(len(self.labels))
        self._broad_labels = [broad_genre(label) for label in self.labels]
        self._broad_to_index = {broad: index for index, broad in enumerate(self.broad_classes)}

    def predict_proba(self, x):
        exact_probabilities = np.asarray(self.exact_classifier.predict_proba(x), dtype=np.float64)
        broad_probabilities = np.asarray(self.broad_classifier.predict_proba(x), dtype=np.float64)
        adjusted = np.zeros_like(exact_probabilities)

        for family, family_index in self._broad_to_index.items():
            label_indices = [index for index, broad in enumerate(self._broad_labels) if broad == family]
            if not label_indices:
                continue
            within_family = exact_probabilities[:, label_indices]
            within_sum = within_family.sum(axis=1, keepdims=True)
            within_distribution = np.divide(
                within_family,
                within_sum,
                out=np.full_like(within_family, 1.0 / len(label_indices)),
                where=within_sum != 0,
            )
            adjusted[:, label_indices] = within_distribution * broad_probabilities[:, [family_index]]

        row_sums = adjusted.sum(axis=1, keepdims=True)
        return np.divide(adjusted, row_sums, out=np.zeros_like(adjusted), where=row_sums != 0)

    def predict(self, x):
        return np.argmax(self.predict_proba(x), axis=1)
