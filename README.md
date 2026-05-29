# CoverSense

A static prototype for inferring a likely music genre from CD or album cover art.

Open `index.html` in a browser, upload an image, or choose one of the generated samples. The current predictor is intentionally transparent: it reads brightness, saturation, contrast, color temperature, hue clusters, and edge density from the cover image, then maps those features to genre probabilities.

This is a prototype baseline, not a trained music-industry classifier. The next upgrade would be to collect labeled album covers by genre and replace `scoreGenres()` in `app.js` with a model exported from TensorFlow.js, ONNX Runtime Web, or a backend API.

## Accuracy Roadmap

The `ml/` folder adds the first real-model path:

1. Download a labeled album-cover dataset from Hugging Face.
2. Encode each cover with CLIP image embeddings.
3. Train a lightweight classifier.
4. Report top-1 and top-3 accuracy.

Start here:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-ml.txt
python ml/download_hf_album_covers.py --max-per-label 50
python ml/train_clip_classifier.py
python ml/predict_cover.py path/to/cover.jpg
```

Remove `--max-per-label 50` when you are ready to train on the full dataset.
