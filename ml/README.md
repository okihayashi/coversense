# Model Training

This folder contains the first real accuracy-improvement path for CoverSense.

The approach is intentionally practical:

1. Download a labeled album-cover dataset from Hugging Face.
2. Encode each cover with a frozen CLIP vision model and cache those embeddings.
3. Train a swappable classifier head on the cached embeddings.
4. Write exact and broad-genre metrics, a confusion matrix, and a reusable model artifact.

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

## Build A Second Dataset Source

Keep the Hugging Face dataset as the baseline and build other sources into
separate folders. The MusicBrainz + Cover Art Archive path searches release
groups by genre tag, downloads front-cover thumbnails, and writes the same
`metadata.csv` shape used by the current training pipeline:

```bash
uv run python ml/download_musicbrainz_cover_art.py --max-per-label 100
```

For a tiny smoke test:

```bash
uv run python ml/download_musicbrainz_cover_art.py --labels rock,jazz,pop --max-per-label 5
```

The MusicBrainz web service expects a meaningful `User-Agent` and should be
called politely. The downloader defaults to a 1.1 second delay between
MusicBrainz search requests.

## Train Classifier

For fast iteration, build embeddings once:

```bash
uv run python ml/build_embeddings.py
```

After reviewing suspected bad labels in the admin page, create a derived
metadata file before rebuilding embeddings:

```bash
uv run python ml/apply_label_reviews.py
uv run python ml/build_embeddings.py \
  --metadata data/album_covers_20_genres_reviewed/metadata.csv \
  --output embeddings/hf-reviewed-clip-vit-base-patch32.npz
```

Then train any supported classifier head:

```bash
uv run python ml/train_classifier.py --classifier logreg
uv run python ml/train_classifier.py --classifier linear-svc
uv run python ml/train_classifier.py --classifier random-forest
uv run python ml/train_classifier.py --classifier mlp
```

To train the same classifier family on the MusicBrainz/Cover Art Archive
source, keep artifacts in dataset-specific paths:

```bash
uv run python ml/build_embeddings.py \
  --metadata data/musicbrainz_cover_art/metadata.csv \
  --output embeddings/musicbrainz-clip-vit-base-patch32.npz

uv run python ml/train_classifier.py \
  --embeddings embeddings/musicbrainz-clip-vit-base-patch32.npz \
  --classifier mlp \
  --model-dir models/datasets/musicbrainz/clip-mlp \
  --report-dir reports/datasets/musicbrainz/clip-mlp
```

For a very small pilot dataset, logistic regression is usually a better
baseline than MLP:

```bash
uv run python ml/train_classifier.py \
  --embeddings embeddings/musicbrainz-clip-vit-base-patch32.npz \
  --classifier logreg \
  --model-dir models/datasets/musicbrainz/clip-logreg \
  --report-dir reports/datasets/musicbrainz/clip-logreg
```

Evaluate the current Hugging Face-trained serving model on the MusicBrainz
embedding cache:

```bash
uv run python ml/evaluate_classifier.py \
  --model models/coversense_clip_classifier.joblib \
  --embeddings embeddings/musicbrainz-clip-vit-base-patch32.npz \
  --report-dir reports/datasets/musicbrainz/hf-serving-model-eval
```

Compare source coverage and metrics:

```bash
uv run python ml/compare_dataset_sources.py
```

This writes:

```text
reports/dataset_source_comparison.md
reports/dataset_source_comparison.json
```

## Reviewed Dataset Loop

After reviewing label-noise candidates in the admin UI, run the reusable review
loop instead of stitching together one-off commands:

```bash
uv run python ml/review_loop.py --all
```

This performs the full current cleanup cycle:

1. Apply `data/label_reviews.csv` into `data/album_covers_20_genres_reviewed/metadata.csv`.
2. Relabel and filter `embeddings/hf-reviewed-clip-vit-base-patch32.npz` without recomputing CLIP vectors.
3. Retrain the serving reviewed hierarchical MLP at `models/clip-mlp-reviewed-hierarchical`.
4. Evaluate HF-on-MusicBrainz and MusicBrainz-on-HF.
5. Refresh `reports/reviewed_dataset_cross_comparison.md`.
6. Print pending top-3 broad-family miss queue counts.

For a lighter check without retraining:

```bash
uv run python ml/review_loop.py --queue-counts
```

For a partial cycle:

```bash
uv run python ml/review_loop.py --apply-reviews --refresh-cache --train
uv run python ml/review_loop.py --cross-eval --compare --queue-counts
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

Reports include exact top-1/top-3 accuracy, broad-family top-1/top-3 accuracy, near-miss rate, far-miss rate, and a hierarchical score. A near miss means the predicted label is in the same broad family as the target, such as Doom Metal vs Death Metal.

## Train CNN Baseline

The CNN path learns directly from resized album-cover pixels instead of using CLIP embeddings:

```bash
uv run python ml/train_cnn.py --epochs 8
```

By default, CNN training uses `--sibling-smoothing 0.15`, which assigns a small amount of target probability to labels in the same broad family. Set `--sibling-smoothing 0` to train with ordinary hard labels.

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

For mislabeled examples, keep the raw dataset unchanged and apply reviewed label overrides during training. See `DATA_QUALITY.md` for the cleanup workflow.
