"""Small helpers shared by training scripts and the Streamlit app."""

import json
import os
from pathlib import Path

import joblib

SAVED_DIR = Path(__file__).resolve().parent.parent / "models" / "saved"
SAVED_DIR.mkdir(parents=True, exist_ok=True)


def save_model(pipeline, name: str, metadata: dict) -> None:
    """Save a fitted sklearn Pipeline + a metadata sidecar JSON.

    name: 'clinical' or 'survey'
    """
    model_path = SAVED_DIR / f"{name}_best_model.joblib"
    meta_path = SAVED_DIR / f"{name}_metadata.json"
    joblib.dump(pipeline, model_path)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved model  -> {model_path}")
    print(f"Saved metadata -> {meta_path}")


def load_model(name: str):
    model_path = SAVED_DIR / f"{name}_best_model.joblib"
    meta_path = SAVED_DIR / f"{name}_metadata.json"
    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"No trained '{name}' model found in {SAVED_DIR}. "
            f"Run `python models/train_{name}.py` first."
        )
    pipeline = joblib.load(model_path)
    with open(meta_path) as f:
        metadata = json.load(f)
    return pipeline, metadata


def compute_threshold_presets(y_true, y_proba, thresholds, precision_floor=0.30):
    """Sweep decision thresholds and report recall/precision/F1 at each,
    then auto-pick a "recommended" threshold: the lowest threshold (i.e.
    highest recall) that still keeps precision at or above `precision_floor`,
    so the model isn't recommending something that floods users with false
    positives. Falls back to 0.50 if nothing clears the floor.

    Returns (presets: list[dict], recommended_threshold: float)
    """
    from sklearn.metrics import f1_score, precision_score, recall_score

    presets = []
    for thr in sorted(thresholds):
        y_pred = (y_proba >= thr).astype(int)
        presets.append({
            "threshold": round(float(thr), 4),
            "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
            "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        })

    recommended = next(
        (p["threshold"] for p in presets if p["precision"] >= precision_floor),
        0.50,
    )
    min_threshold = presets[0]["threshold"] if presets else None

    for p in presets:
        if p["threshold"] == recommended:
            p["note"] = "Recommended screening threshold (best recall with usable precision)"
        elif p["threshold"] == min_threshold:
            p["note"] = "Maximum recall — very low precision expected"
        elif p["threshold"] == 0.50:
            p["note"] = "Balanced / standard cutoff"
        else:
            p["note"] = ""

    return presets, recommended