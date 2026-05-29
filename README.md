# CoverSense

A static prototype for inferring a likely music genre from CD or album cover art.

Open `index.html` in a browser, upload an image, or choose one of the generated samples. The current predictor is intentionally transparent: it reads brightness, saturation, contrast, color temperature, hue clusters, and edge density from the cover image, then maps those features to genre probabilities.

This is a prototype baseline, not a trained music-industry classifier. The next upgrade would be to collect labeled album covers by genre and replace `scoreGenres()` in `app.js` with a model exported from TensorFlow.js, ONNX Runtime Web, or a backend API.
