#!/usr/bin/env python3
"""Serve the CoverSense app and trained model prediction API."""

from __future__ import annotations

import json
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import joblib
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError
from transformers import CLIPModel, CLIPProcessor


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "coversense_clip_classifier.joblib"
METRICS_PATH = ROOT / "reports" / "metrics.json"
DISPLAY_LABELS = {
    "blues": "Blues",
    "classical": "Classical",
    "country": "Country",
    "deathmetal": "Death Metal",
    "doommetal": "Doom Metal",
    "drumnbass": "Drum & Bass",
    "electronic": "Electronic",
    "folk": "Folk",
    "grime": "Grime",
    "heavymetal": "Heavy Metal",
    "hiphop": "Hip-Hop",
    "jazz": "Jazz",
    "lofi": "Lo-Fi",
    "pop": "Pop",
    "psychedelicrock": "Psychedelic Rock",
    "punk": "Punk",
    "reggae": "Reggae",
    "rock": "Rock",
    "soul": "Soul",
    "techno": "Techno",
}

app = FastAPI(title="CoverSense", version="0.1.0")
NO_CACHE_HEADERS = {
    "Cache-Control": "no-store, max-age=0",
    "Pragma": "no-cache",
}


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def tensor_from_clip_output(output):
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "image_embeds"):
        return output.image_embeds
    if hasattr(output, "pooler_output"):
        return output.pooler_output
    if isinstance(output, (tuple, list)):
        return output[0]
    raise TypeError(f"Unsupported CLIP output type: {type(output)!r}")


@lru_cache(maxsize=1)
def load_predictor():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

    payload = joblib.load(MODEL_PATH)
    device = choose_device()
    processor = CLIPProcessor.from_pretrained(payload["clip_model"])
    clip_model = CLIPModel.from_pretrained(payload["clip_model"]).to(device)
    clip_model.eval()

    return {
        "payload": payload,
        "device": device,
        "processor": processor,
        "clip_model": clip_model,
    }


def read_metrics() -> dict:
    if not METRICS_PATH.exists():
        return {}
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def classifier_name(payload: dict, metrics: dict | None = None) -> str:
    if payload.get("classifier_name"):
        return payload["classifier_name"]
    if metrics and metrics.get("classifier"):
        return metrics["classifier"]
    return "logreg"


def image_from_upload(raw_bytes: bytes) -> Image.Image:
    try:
        return Image.open(BytesIO(raw_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a readable image.") from exc


@torch.inference_mode()
def predict_image(image: Image.Image, top_k: int = 6) -> list[dict]:
    predictor = load_predictor()
    payload = predictor["payload"]
    device = predictor["device"]
    processor = predictor["processor"]
    clip_model = predictor["clip_model"]

    inputs = processor(images=[image], return_tensors="pt", padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    features = clip_model.get_image_features(**inputs)
    features = tensor_from_clip_output(features)
    features = features / features.norm(dim=-1, keepdim=True)
    features = features.cpu().numpy()

    classifier = payload["classifier"]
    if isinstance(classifier, dict):
        features = classifier["scaler"].transform(features)
        probabilities = classifier["classifier"].predict_proba(features)[0]
    elif "scaler" in payload:
        features = payload["scaler"].transform(features)
        probabilities = classifier.predict_proba(features)[0]
    else:
        probabilities = classifier.predict_proba(features)[0]

    labels = payload["label_encoder"].classes_
    ranked = sorted(zip(labels, probabilities), key=lambda item: item[1], reverse=True)

    return [
        {
            "label": label,
            "genre": DISPLAY_LABELS.get(label, label.replace("-", " ").title()),
            "probability": float(probability),
        }
        for label, probability in ranked[:top_k]
    ]


@app.get("/")
def index():
    return FileResponse(ROOT / "index.html", headers=NO_CACHE_HEADERS)


@app.get("/app.js")
def app_js():
    return FileResponse(ROOT / "app.js", media_type="application/javascript", headers=NO_CACHE_HEADERS)


@app.get("/styles.css")
def styles_css():
    return FileResponse(ROOT / "styles.css", media_type="text/css", headers=NO_CACHE_HEADERS)


@app.get("/api/health")
def health():
    metrics = read_metrics()
    return {
        "ok": True,
        "modelAvailable": MODEL_PATH.exists(),
        "modelPath": str(MODEL_PATH),
        "metrics": {
            "accuracyTop1": metrics.get("accuracy_top_1"),
            "accuracyTop3": metrics.get("accuracy_top_3"),
            "trainSize": metrics.get("train_size"),
            "testSize": metrics.get("test_size"),
            "clipModel": metrics.get("clip_model"),
        },
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Trained model is not available. Run training first.")

    raw_bytes = await file.read()
    image = image_from_upload(raw_bytes)
    predictions = predict_image(image)
    metrics = read_metrics()

    return {
        "model": f"clip-{classifier_name(load_predictor()['payload'], metrics)}",
        "clipModel": metrics.get("clip_model", "openai/clip-vit-base-patch32"),
        "accuracyTop1": metrics.get("accuracy_top_1"),
        "accuracyTop3": metrics.get("accuracy_top_3"),
        "predictions": predictions,
    }
