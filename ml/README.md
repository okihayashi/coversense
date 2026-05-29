# Model Training

This folder contains the first real accuracy-improvement path for CoverSense.

The approach is intentionally practical:

1. Download a labeled album-cover dataset from Hugging Face.
2. Encode each cover with a frozen CLIP vision model.
3. Train a lightweight logistic-regression classifier on those image embeddings.
4. Write accuracy metrics, a confusion matrix, and a reusable model artifact.

The default dataset is `eong/20k-Album-Covers-within-20-Genres`, which has 20 genres and 1,000 album covers per genre.

## Setup

```bash
cd /Users/okihayas/Documents/Codex/2026-05-28/coversense
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-ml.txt
```

## Download Dataset

```bash
python ml/download_hf_album_covers.py
```

This creates:

```text
data/album_covers_20_genres/images/
data/album_covers_20_genres/metadata.csv
data/album_covers_20_genres/labels.json
```

Use `--max-per-label` for a quick smoke test:

```bash
python ml/download_hf_album_covers.py --max-per-label 50
```

## Train Classifier

```bash
python ml/train_clip_classifier.py
```

This creates:

```text
models/coversense_clip_classifier.joblib
models/coversense_labels.json
reports/metrics.json
reports/classification_report.txt
reports/confusion_matrix.csv
```

## Predict One Cover

```bash
python ml/predict_cover.py path/to/cover.jpg
```

## Notes

The Hugging Face dataset is convenient for a first model, but it does not include artist IDs. For a stronger benchmark, later datasets should split by artist or album family so the model cannot memorize artist-specific design language.
