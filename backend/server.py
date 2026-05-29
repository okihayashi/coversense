#!/usr/bin/env python3
"""Serve the CoverSense app and trained model prediction API."""

from __future__ import annotations

import json
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import joblib
import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError
from transformers import CLIPModel, CLIPProcessor

from ml.genre_taxonomy import broad_genre


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "coversense_clip_classifier.joblib"
METRICS_PATH = ROOT / "reports" / "metrics.json"
FAILURES_PATH = ROOT / "reports" / "failures.json"
EXAMPLES_PATH = ROOT / "reports" / "eval_examples.json"
CNN_MODEL_PATH = ROOT / "models" / "coversense_cnn.pt"
CNN_METRICS_PATH = ROOT / "reports" / "cnn_metrics.json"
CNN_FAILURES_PATH = ROOT / "reports" / "cnn_failures.json"
CNN_EXAMPLES_PATH = ROOT / "reports" / "cnn_eval_examples.json"
MODEL_RUNS_DIR = ROOT / "reports" / "model_runs"
MODEL_RUN_ARTIFACTS_DIR = ROOT / "models" / "model_runs"
CLIP_CLASSIFIER_RUNS = [
    ("clip-logreg", "CLIP + Logistic Regression", "logreg"),
    ("clip-linear-svc", "CLIP + Linear SVC", "linear-svc"),
    ("clip-random-forest", "CLIP + Random Forest", "random-forest"),
    ("clip-mlp", "CLIP + MLP", "mlp"),
]
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
BROAD_DISPLAY_LABELS = {
    "classical": "Classical",
    "electronic": "Electronic",
    "hiphop": "Hip-Hop",
    "jazz": "Jazz",
    "metal": "Metal",
    "pop-soul": "Pop / Soul",
    "rock": "Rock",
    "roots": "Roots",
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
    return read_json(METRICS_PATH, {})


def read_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def classifier_name(payload: dict, metrics: dict | None = None) -> str:
    if payload.get("classifier_name"):
        return payload["classifier_name"]
    if metrics and metrics.get("classifier"):
        return metrics["classifier"]
    return "logreg"


def display_label(label: str) -> str:
    return DISPLAY_LABELS.get(label, label.replace("-", " ").title())


def display_broad_label(label: str) -> str:
    broad = broad_genre(label)
    return BROAD_DISPLAY_LABELS.get(broad, display_label(broad))


def public_artwork_url(image_path: str) -> str:
    parts = [part for part in Path(image_path).parts if part not in {"", "."}]
    return "/api/artwork/" + "/".join(parts)


def normalize_examples(examples: list[dict], limit: int | None = None) -> list[dict]:
    normalized = []
    for example in examples[:limit]:
        item = dict(example)
        item["actualDisplay"] = display_label(item.get("actual", ""))
        item["predictedDisplay"] = display_label(item.get("predicted", ""))
        item["imageUrl"] = public_artwork_url(item.get("imagePath", ""))
        item["topPredictions"] = [
            {
                **prediction,
                "display": display_label(prediction.get("label", "")),
            }
            for prediction in item.get("topPredictions", [])
        ]
        normalized.append(item)
    return normalized


def model_summary(
    model_id: str,
    name: str,
    family: str,
    artifact_path: Path,
    metrics_path: Path,
    examples_path: Path,
    failures_path: Path,
) -> dict:
    metrics = read_json(metrics_path, {})
    examples = read_json(examples_path, [])
    failures = read_json(failures_path, [])
    return {
        "id": model_id,
        "name": name,
        "family": family,
        "artifactAvailable": artifact_path.exists(),
        "artifactPath": str(artifact_path),
        "metricsPath": str(metrics_path),
        "examplesAvailable": examples_path.exists(),
        "failuresAvailable": failures_path.exists(),
        "metrics": metrics,
        "evaluationCount": len(examples),
        "failureCount": len(failures),
        "failureRate": (len(failures) / len(examples)) if examples else None,
        "failurePreview": normalize_examples(failures, limit=12),
    }


def model_paths_for(model_id: str) -> tuple[Path, Path, Path, Path] | None:
    if model_id == "clip-classifier":
        return MODEL_PATH, METRICS_PATH, EXAMPLES_PATH, FAILURES_PATH
    if model_id == "small-cnn":
        return CNN_MODEL_PATH, CNN_METRICS_PATH, CNN_EXAMPLES_PATH, CNN_FAILURES_PATH
    known_run_ids = {run_id for run_id, _, _ in CLIP_CLASSIFIER_RUNS}
    if model_id in known_run_ids:
        run_dir = MODEL_RUNS_DIR / model_id
        artifact_path = MODEL_RUN_ARTIFACTS_DIR / model_id / "coversense_clip_classifier.joblib"
        return artifact_path, run_dir / "metrics.json", run_dir / "eval_examples.json", run_dir / "failures.json"
    return None


def comparison_model_summaries() -> list[dict]:
    summaries = []
    active_classifier = classifier_name({}, read_metrics())
    active_run_id = f"clip-{active_classifier}"
    active_run_found = False

    for model_id, name, classifier in CLIP_CLASSIFIER_RUNS:
        artifact_path, metrics_path, examples_path, failures_path = model_paths_for(model_id)
        is_serving = model_id == active_run_id
        active_run_found = active_run_found or is_serving
        summary = model_summary(
            model_id=model_id,
            name=f"Serving {name}" if is_serving else name,
            family="embedding-classifier",
            artifact_path=artifact_path,
            metrics_path=metrics_path,
            examples_path=examples_path,
            failures_path=failures_path,
        )
        summary["metrics"] = {"classifier": classifier, **summary["metrics"]}
        summary["serving"] = is_serving
        summaries.append(summary)

    if not active_run_found:
        active_summary = model_summary(
            model_id="clip-classifier",
            name=f"Serving CLIP + {active_classifier.upper()}",
            family="embedding-classifier",
            artifact_path=MODEL_PATH,
            metrics_path=METRICS_PATH,
            examples_path=EXAMPLES_PATH,
            failures_path=FAILURES_PATH,
        )
        active_summary["serving"] = True
        summaries.insert(0, active_summary)

    summaries.append(
        model_summary(
            model_id="small-cnn",
            name="Small CNN",
            family="raw-pixel-cnn",
            artifact_path=CNN_MODEL_PATH,
            metrics_path=CNN_METRICS_PATH,
            examples_path=CNN_EXAMPLES_PATH,
            failures_path=CNN_FAILURES_PATH,
        )
    )
    return summaries


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
            "broadGenre": broad_genre(label),
            "broadGenreDisplay": display_broad_label(label),
            "genre": DISPLAY_LABELS.get(label, label.replace("-", " ").title()),
            "probability": float(probability),
        }
        for label, probability in ranked[:top_k]
    ]


@app.get("/")
def index():
    return FileResponse(ROOT / "index.html", headers=NO_CACHE_HEADERS)


@app.get("/admin")
def admin():
    return FileResponse(ROOT / "admin.html", headers=NO_CACHE_HEADERS)


@app.get("/app.js")
def app_js():
    return FileResponse(ROOT / "app.js", media_type="application/javascript", headers=NO_CACHE_HEADERS)


@app.get("/admin.js")
def admin_js():
    return FileResponse(ROOT / "admin.js", media_type="application/javascript", headers=NO_CACHE_HEADERS)


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


@app.get("/api/models")
def models():
    return {"models": comparison_model_summaries()}


@app.get("/api/models/{model_id}/examples")
def model_examples(
    model_id: str,
    failed_only: bool = Query(default=True, alias="failedOnly"),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
):
    paths = model_paths_for(model_id)
    if paths is None:
        raise HTTPException(status_code=404, detail="Unknown model.")

    _, _, examples_path, failures_path = paths
    source_path = failures_path if failed_only else examples_path
    examples = read_json(source_path, [])
    window = examples[offset : offset + limit]
    return {
        "modelId": model_id,
        "failedOnly": failed_only,
        "offset": offset,
        "limit": limit,
        "total": len(examples),
        "examples": normalize_examples(window),
    }


@app.get("/api/artwork/{image_path:path}")
def artwork(image_path: str):
    candidate = (ROOT / image_path).resolve()
    if ROOT not in candidate.parents or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Artwork not found.")
    return FileResponse(candidate)


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
