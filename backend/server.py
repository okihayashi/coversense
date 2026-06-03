#!/usr/bin/env python3
"""Serve the CoverSense app and trained model prediction API."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import joblib
import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from transformers import CLIPModel, CLIPProcessor

from ml.genre_taxonomy import broad_genre


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_MODEL_RUN_ID = "clip-mlp-reviewed-hierarchical"
MODEL_PATH = ROOT / "models" / ACTIVE_MODEL_RUN_ID / "coversense_clip_classifier.joblib"
METRICS_PATH = ROOT / "reports" / "model_runs" / ACTIVE_MODEL_RUN_ID / "metrics.json"
FAILURES_PATH = ROOT / "reports" / "model_runs" / ACTIVE_MODEL_RUN_ID / "failures.json"
EXAMPLES_PATH = ROOT / "reports" / "model_runs" / ACTIVE_MODEL_RUN_ID / "eval_examples.json"
LEGACY_MODEL_PATH = ROOT / "models" / "coversense_clip_classifier.joblib"
LEGACY_METRICS_PATH = ROOT / "reports" / "metrics.json"
LEGACY_FAILURES_PATH = ROOT / "reports" / "failures.json"
LEGACY_EXAMPLES_PATH = ROOT / "reports" / "eval_examples.json"
CNN_MODEL_PATH = ROOT / "models" / "coversense_cnn.pt"
CNN_METRICS_PATH = ROOT / "reports" / "cnn_metrics.json"
CNN_FAILURES_PATH = ROOT / "reports" / "cnn_failures.json"
CNN_EXAMPLES_PATH = ROOT / "reports" / "cnn_eval_examples.json"
MODEL_RUNS_DIR = ROOT / "reports" / "model_runs"
MODEL_RUN_ARTIFACTS_DIR = ROOT / "models" / "model_runs"
LABEL_REVIEW_PATH = ROOT / "data" / "label_reviews.csv"
LABEL_REVIEW_FIELDNAMES = [
    "image_path",
    "dataset",
    "original_label",
    "suggested_label",
    "review_status",
    "reviewed_label",
    "secondary_labels",
    "reason",
    "reviewer",
    "reviewed_at",
]
CLIP_CLASSIFIER_RUNS = [
    ("clip-logreg", "CLIP + Logistic Regression", "logreg"),
    ("clip-linear-svc", "CLIP + Linear SVC", "linear-svc"),
    ("clip-random-forest", "CLIP + Random Forest", "random-forest"),
    ("clip-mlp", "CLIP + MLP", "mlp"),
    ("clip-mlp-reviewed", "CLIP + MLP Reviewed Labels", "mlp"),
    ("clip-mlp-reviewed-broad-weighted", "CLIP + MLP Broad Weighted", "mlp-broad-weighted"),
    ("clip-mlp-reviewed-broad-first", "CLIP + MLP Broad First", "mlp-broad-first"),
    ("clip-mlp-reviewed-hierarchical", "CLIP + MLP Hierarchical", "mlp-hierarchical"),
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


def file_mtime_iso(path: Path) -> str | None:
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()


class LabelReviewDecision(BaseModel):
    imagePath: str
    dataset: str = "hf"
    originalLabel: str
    suggestedLabel: str | None = None
    reviewStatus: str
    reviewedLabel: str | None = None
    secondaryLabels: list[str] = []
    reason: str = ""
    reviewer: str = "okihayashi"


def read_label_reviews() -> dict[str, dict[str, str]]:
    if not LABEL_REVIEW_PATH.exists():
        return {}
    with LABEL_REVIEW_PATH.open(newline="", encoding="utf-8") as file:
        return {row["image_path"]: row for row in csv.DictReader(file)}


def write_label_reviews(rows: dict[str, dict[str, str]]) -> None:
    LABEL_REVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LABEL_REVIEW_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=LABEL_REVIEW_FIELDNAMES)
        writer.writeheader()
        for row in sorted(rows.values(), key=lambda item: item["image_path"]):
            writer.writerow({field: row.get(field, "") for field in LABEL_REVIEW_FIELDNAMES})


def dataset_name_for_path(image_path: str) -> str:
    if "musicbrainz_cover_art" in image_path:
        return "musicbrainz"
    if "album_covers_20_genres" in image_path:
        return "hf"
    return "unknown"


def review_hint(example: dict) -> str:
    actual = example.get("actual", "")
    predicted = example.get("predicted", "")
    error = example.get("errorType", "")
    top_3_labels = [prediction.get("label") for prediction in example.get("topPredictions", [])[:3]]
    top_3_broad_match = any(broad_genre(label) == broad_genre(actual) for label in top_3_labels)
    if actual and actual not in top_3_labels and not top_3_broad_match:
        return "Top-3 broad miss. Strong candidate for label noise, missing secondary labels, or taxonomy mismatch."
    if error == "near_miss":
        return "Possible multi-label or sibling-genre case. Review before changing."
    if broad_genre(actual) == "rock" and broad_genre(predicted) == "metal":
        return "Rock/metal boundary candidate. Often better as multi-label unless metadata is clear."
    if {broad_genre(actual), broad_genre(predicted)} <= {"hiphop", "electronic"}:
        return "Club/electronic/hip-hop boundary candidate. Could be overlap rather than a bad label."
    return "High-confidence broad-family contradiction. Good candidate for label-noise review."


def normalize_review_candidate(example: dict, review: dict[str, str] | None = None) -> dict:
    item = normalize_examples([example])[0]
    top_3_labels = [prediction.get("label") for prediction in item.get("topPredictions", [])[:3]]
    actual = item.get("actual", "")
    top_3_broad_match = any(broad_genre(label) == broad_genre(actual) for label in top_3_labels)
    item["dataset"] = dataset_name_for_path(item.get("imagePath", ""))
    item["suggestedLabel"] = item.get("predicted", "")
    item["suggestedDisplay"] = display_label(item.get("suggestedLabel", ""))
    item["top3Labels"] = top_3_labels
    item["top3Miss"] = actual not in top_3_labels
    item["top3BroadMiss"] = not top_3_broad_match
    item["reviewHint"] = review_hint(item)
    item["review"] = review
    return item


def review_source_path(source: str) -> Path:
    if source == "musicbrainz":
        reviewed_path = (
            ROOT / "reports" / "datasets" / "musicbrainz" / "clip-mlp-reviewed-hierarchical" / "eval_examples.json"
        )
        fallback_path = ROOT / "reports" / "datasets" / "musicbrainz" / "hf-serving-model-eval" / "eval_examples.json"
    else:
        reviewed_path = MODEL_RUNS_DIR / "clip-mlp-reviewed-hierarchical" / "eval_examples.json"
        fallback_path = MODEL_RUNS_DIR / "clip-mlp" / "eval_examples.json"
    return reviewed_path if reviewed_path.exists() else fallback_path


def label_review_candidates(
    source: str,
    status: str,
    limit: int,
    offset: int,
    min_confidence: float,
    error_type_filter: str,
    issue_type: str = "top1_miss",
) -> dict:
    source_path = review_source_path(source)

    reviews = read_label_reviews()
    examples = read_json(source_path, [])
    candidates = []
    for example in examples:
        top_3_labels = [prediction.get("label") for prediction in example.get("topPredictions", [])[:3]]
        actual = example.get("actual", "")
        top_3_miss = actual not in top_3_labels
        top_3_broad_miss = not any(broad_genre(label) == broad_genre(actual) for label in top_3_labels)
        if issue_type == "top3_miss":
            if not top_3_miss or not top_3_broad_miss:
                continue
        elif example.get("correct"):
            continue
        if error_type_filter != "all" and example.get("errorType") != error_type_filter:
            continue
        if float(example.get("confidence", 0)) < min_confidence:
            continue
        image_path = example.get("imagePath", "")
        review = reviews.get(image_path)
        if status == "pending" and review:
            continue
        if status == "reviewed" and not review:
            continue
        if status not in {"all", "pending", "reviewed"} and (not review or review.get("review_status") != status):
            continue
        candidates.append(normalize_review_candidate(example, review))

    candidates.sort(key=lambda item: item.get("confidence", 0), reverse=True)
    window = candidates[offset : offset + limit]
    return {
        "source": source,
        "status": status,
        "offset": offset,
        "limit": limit,
        "issueType": issue_type,
        "total": len(candidates),
        "candidates": window,
        "reviewFile": str(LABEL_REVIEW_PATH),
    }


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
        "metricsUpdatedAt": file_mtime_iso(metrics_path),
        "artifactUpdatedAt": file_mtime_iso(artifact_path),
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
    if model_id == "legacy-clip-classifier":
        return LEGACY_MODEL_PATH, LEGACY_METRICS_PATH, LEGACY_EXAMPLES_PATH, LEGACY_FAILURES_PATH
    if model_id == "small-cnn":
        return CNN_MODEL_PATH, CNN_METRICS_PATH, CNN_EXAMPLES_PATH, CNN_FAILURES_PATH
    known_run_ids = {run_id for run_id, _, _ in CLIP_CLASSIFIER_RUNS}
    if model_id in known_run_ids:
        run_dir = MODEL_RUNS_DIR / model_id
        artifact_path = ROOT / "models" / model_id / "coversense_clip_classifier.joblib"
        if not artifact_path.exists():
            artifact_path = MODEL_RUN_ARTIFACTS_DIR / model_id / "coversense_clip_classifier.joblib"
        return artifact_path, run_dir / "metrics.json", run_dir / "eval_examples.json", run_dir / "failures.json"
    return None


def comparison_model_summaries() -> list[dict]:
    summaries = []

    for model_id, name, classifier in CLIP_CLASSIFIER_RUNS:
        artifact_path, metrics_path, examples_path, failures_path = model_paths_for(model_id)
        is_serving = model_id == ACTIVE_MODEL_RUN_ID
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
    return sorted(summaries, key=lambda summary: (not summary.get("serving", False), summary["name"]))


def genre_examples(label: str, limit: int = 6) -> list[dict]:
    examples = read_json(EXAMPLES_PATH, [])
    exact_matches = [
        example
        for example in examples
        if example.get("actual") == label and example.get("predicted") == label
    ]
    if len(exact_matches) < limit:
        exact_matches.extend(
            example
            for example in examples
            if example.get("actual") == label and example not in exact_matches
        )
    ranked = sorted(exact_matches, key=lambda example: example.get("confidence", 0), reverse=True)
    return normalize_examples(ranked[:limit])


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
        "activeModelRunId": ACTIVE_MODEL_RUN_ID,
        "modelAvailable": MODEL_PATH.exists(),
        "modelPath": str(MODEL_PATH),
        "modelUpdatedAt": file_mtime_iso(MODEL_PATH),
        "metricsPath": str(METRICS_PATH),
        "metricsUpdatedAt": file_mtime_iso(METRICS_PATH),
        "metrics": {
            "accuracyTop1": metrics.get("accuracy_top_1"),
            "accuracyTop3": metrics.get("accuracy_top_3"),
            "accuracyBroadTop1": metrics.get("accuracy_broad_top_1"),
            "accuracyBroadTop3": metrics.get("accuracy_broad_top_3"),
            "farMissRate": metrics.get("far_miss_rate"),
            "nearMissRate": metrics.get("near_miss_rate"),
            "hierarchicalScore": metrics.get("hierarchical_score"),
            "trainSize": metrics.get("train_size"),
            "testSize": metrics.get("test_size"),
            "clipModel": metrics.get("clip_model"),
            "classifier": metrics.get("classifier"),
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


@app.get("/api/review/candidates")
def review_candidates(
    source: str = Query(default="hf", pattern="^(hf|musicbrainz)$"),
    status: str = Query(default="pending", pattern="^(all|pending|reviewed|keep|relabel|multi_label|exclude|needs_metadata)$"),
    issue_type: str = Query(default="top1_miss", alias="issueType", pattern="^(top1_miss|top3_miss)$"),
    error_type: str = Query(default="far_miss", alias="errorType", pattern="^(all|near_miss|far_miss)$"),
    min_confidence: float = Query(default=0.9, alias="minConfidence", ge=0, le=1),
    limit: int = Query(default=12, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    return label_review_candidates(source, status, limit, offset, min_confidence, error_type, issue_type)


@app.post("/api/review/decision")
def save_review_decision(decision: LabelReviewDecision):
    if decision.reviewStatus not in {"keep", "relabel", "multi_label", "exclude", "needs_metadata"}:
        raise HTTPException(status_code=400, detail="Unsupported review status.")
    if decision.reviewedLabel and decision.reviewedLabel not in DISPLAY_LABELS:
        raise HTTPException(status_code=400, detail="Unknown reviewed label.")
    unknown_secondary = [label for label in decision.secondaryLabels if label not in DISPLAY_LABELS]
    if unknown_secondary:
        raise HTTPException(status_code=400, detail=f"Unknown secondary labels: {', '.join(unknown_secondary)}")

    reviews = read_label_reviews()
    reviewed_at = datetime.now(timezone.utc).isoformat()
    row = {
        "image_path": decision.imagePath,
        "dataset": decision.dataset,
        "original_label": decision.originalLabel,
        "suggested_label": decision.suggestedLabel or "",
        "review_status": decision.reviewStatus,
        "reviewed_label": decision.reviewedLabel or "",
        "secondary_labels": ",".join(decision.secondaryLabels),
        "reason": decision.reason,
        "reviewer": decision.reviewer,
        "reviewed_at": reviewed_at,
    }
    reviews[decision.imagePath] = row
    write_label_reviews(reviews)
    return {"ok": True, "review": row, "reviewFile": str(LABEL_REVIEW_PATH)}


@app.get("/api/genre-examples/{label}")
def examples_for_genre(label: str, limit: int = Query(default=6, ge=1, le=12)):
    if label not in DISPLAY_LABELS:
        raise HTTPException(status_code=404, detail="Unknown genre.")
    return {
        "label": label,
        "genre": display_label(label),
        "broadGenre": broad_genre(label),
        "broadGenreDisplay": display_broad_label(label),
        "examples": genre_examples(label, limit=limit),
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
        "modelRunId": ACTIVE_MODEL_RUN_ID,
        "clipModel": metrics.get("clip_model", "openai/clip-vit-base-patch32"),
        "accuracyTop1": metrics.get("accuracy_top_1"),
        "accuracyTop3": metrics.get("accuracy_top_3"),
        "accuracyBroadTop1": metrics.get("accuracy_broad_top_1"),
        "accuracyBroadTop3": metrics.get("accuracy_broad_top_3"),
        "farMissRate": metrics.get("far_miss_rate"),
        "nearMissRate": metrics.get("near_miss_rate"),
        "hierarchicalScore": metrics.get("hierarchical_score"),
        "predictions": predictions,
        "similarExamples": genre_examples(predictions[0]["label"]) if predictions else [],
    }
