# Model Training

This folder contains the first real accuracy-improvement path for CoverSense.

The approach is intentionally practical:

1. Download a labeled album-cover dataset from Hugging Face.
2. Encode each cover with a frozen CLIP vision model and cache those embeddings.
3. Train a swappable classifier head on the cached embeddings.
4. Write accuracy metrics, a confusion matrix, and a reusable model artifact.

The default dataset is `eong/20k-Album-Covers-within-20-Genres`, which has 20 genres and 1,000 album covers per genre.

## Setup

```bash
cd /Users/okihayas/Documents/Codex/2026-05-28/coversense
uv sync
```

## Download Dataset

```bash
uv run python ml/download_hf_album_covers.py
```

This creates:

```text
data/album_covers_20_genres/images/
data/album_covers_20_genres/metadata.csv
data/album_covers_20_genres/labels.json
```

Use `--max-per-label` for a quick smoke test:

```bash
uv run python ml/download_hf_album_covers.py --max-per-label 50
```

## Train Classifier

For fast iteration, build embeddings once:

```bash
uv run python ml/build_embeddings.py
```

Then train any supported classifier head:

```bash
uv run python ml/train_classifier.py --classifier logreg
uv run python ml/train_classifier.py --classifier linear-svc
uv run python ml/train_classifier.py --classifier random-forest
uv run python ml/train_classifier.py --classifier mlp
```

The original one-command CLIP + logistic-regression path still works:

```bash
uv run python ml/train_clip_classifier.py
```

This creates:

```text
models/coversense_clip_classifier.joblib
models/coversense_labels.json
reports/metrics.json
reports/classification_report.txt
reports/confusion_matrix.csv
```

## Train CNN Baseline

The CNN path learns directly from resized album-cover pixels instead of using CLIP embeddings:

```bash
uv run python ml/train_cnn.py --epochs 8
```

This creates:

```text
models/coversense_cnn.pt
reports/cnn_metrics.json
reports/cnn_classification_report.txt
reports/cnn_confusion_matrix.csv
```

For a quick code-path check:

```bash
uv run python ml/train_cnn.py --max-samples-per-label 25 --epochs 1 --image-size 96
```

For a fuller CPU comparison with lightweight random hyperparameter search:

```bash
uv run python ml/train_cnn.py --trials 2 --trial-epochs 1 --epochs 3 --device cpu
```

The CNN trainer writes `reports/cnn_eval_examples.json` and `reports/cnn_failures.json` for the admin page.

## Predict One Cover

```bash
uv run python ml/predict_cover.py path/to/cover.jpg
uv run python ml/predict_cnn.py path/to/cover.jpg
```

## Notes

The Hugging Face dataset is convenient for a first model, but it does not include artist IDs. For a stronger benchmark, later datasets should split by artist or album family so the model cannot memorize artist-specific design language.
