from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import load_settings, resolve_path
from src.predict import predict_from_records

app = FastAPI(title="Medical ML API", version="0.1.0")


class PredictRequest(BaseModel):
    age: float
    bmi: float
    glucose: float
    systolic_bp: float
    diastolic_bp: float
    insulin: float
    hba1c: float
    gender: str = Field(min_length=1)
    smoking_status: str = Field(min_length=1)
    family_history: str = Field(min_length=1)


@app.get("/health")
def health() -> dict[str, Any]:
    settings = load_settings()
    model_path = resolve_path(settings.paths["best_model"])
    return {
        "status": "ok" if model_path.exists() else "model_missing",
        "model_version": "local-dev",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    settings = load_settings()
    metadata_path = resolve_path(settings.paths["model_metadata"])
    if not metadata_path.exists():
        raise HTTPException(status_code=404, detail="Model metadata not found.")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


@app.post("/predict")
def predict(payload: PredictRequest) -> dict[str, Any]:
    try:
        predictions = predict_from_records([payload.model_dump()])
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Prediction error: {exc}") from exc

    return {
        "request_id": str(uuid.uuid4()),
        "result": predictions[0],
    }
