from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from api.app import app
from src.config import load_settings, resolve_path
from src.preprocessing import build_preprocessor


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 60
    df = pd.DataFrame(
        {
            "patient_id": np.arange(n),
            "age": rng.integers(25, 80, n),
            "bmi": rng.normal(28, 4, n),
            "glucose": rng.normal(130, 20, n),
            "systolic_bp": rng.normal(128, 15, n),
            "diastolic_bp": rng.normal(82, 10, n),
            "insulin": rng.normal(110, 35, n),
            "hba1c": rng.normal(6.1, 1.0, n),
            "gender": rng.choice(["male", "female"], size=n),
            "smoking_status": rng.choice(["never", "former", "current"], size=n),
            "family_history": rng.choice(["yes", "no"], size=n),
            "outcome": rng.integers(0, 2, n),
        }
    )
    df.loc[0, "bmi"] = np.nan
    df.loc[1, "glucose"] = np.nan
    return df


@pytest.fixture
def api_client(tmp_path, monkeypatch, synthetic_df):
    settings = load_settings()
    model_path = tmp_path / "best_model.joblib"
    metadata_path = tmp_path / "model_metadata.json"

    X = synthetic_df.drop(columns=["outcome"])
    y = synthetic_df["outcome"]

    pipeline = Pipeline(
        steps=[
            ("preprocessing", build_preprocessor()),
            ("model", LogisticRegression(max_iter=200)),
        ]
    )
    pipeline.fit(X, y)
    joblib.dump(pipeline, model_path)
    metadata_path.write_text(
        json.dumps({"best_model_name": "logreg", "metrics": {"f1_score": 0.5}}),
        encoding="utf-8",
    )

    def fake_resolve_path(relative_path: str):
        if relative_path == settings.paths["best_model"]:
            return model_path
        if relative_path == settings.paths["model_metadata"]:
            return metadata_path
        return resolve_path(relative_path)

    monkeypatch.setattr("src.predict.resolve_path", fake_resolve_path)
    monkeypatch.setattr("api.app.resolve_path", fake_resolve_path)

    return TestClient(app)
