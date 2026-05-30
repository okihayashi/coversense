# Dataset Label Quality

Training performance is capped by label quality. If an album such as Lana Del Rey's `Honeymoon` is labeled as `electronic`, the model is rewarded for learning the wrong association.

## Correction Strategy

Keep the downloaded dataset immutable and layer corrections on top of it.

1. Store the original label from `metadata.csv`.
2. Add a reviewed-label override file.
3. Train from a resolved label: `reviewed_label` when present, otherwise `original_label`.
4. Keep every correction auditable with reviewer, reason, and timestamp.

Recommended override schema:

```csv
image_path,original_label,reviewed_label,review_status,reason,reviewer,reviewed_at
data/album_covers_20_genres/images/electronic/example.jpg,electronic,pop,approved,artist-album metadata mismatch,okihayashi,2026-05-30
```

Use `review_status=exclude` for artwork that is ambiguous, duplicated, corrupted, or outside the supported taxonomy.

## How To Find Suspicious Labels

Prioritize examples that are likely mislabeled:

1. High-confidence model disagreement: the trained model strongly predicts a different genre than the dataset label.
2. Cross-model agreement: CLIP+MLP, logistic regression, random forest, and CNN agree against the current label.
3. Broad-family contradiction: the prediction is a far miss, not just a sibling genre.
4. Metadata lookup mismatch: artist/album metadata from a trusted music catalog disagrees with the dataset label.
5. Duplicate artwork with conflicting labels.

The admin failure view should become the review queue: show artwork, original label, predicted label, broad genre, confidence, and actions for approve correction, keep original, or exclude.

## Training Rules

Use these rules when retraining:

1. Apply approved label overrides before splitting the dataset.
2. Exclude rows marked `exclude`.
3. Preserve artist/album grouping when possible, so near-duplicate artwork does not leak across train/test.
4. Report metrics for both the raw test set and the reviewed test set until the cleanup coverage is high.

## Why This Helps

Architecture changes improve how the model learns. Label cleanup improves what the model is allowed to learn. For noisy web-scale genre labels, the cleanup loop is likely the highest-leverage accuracy work.
