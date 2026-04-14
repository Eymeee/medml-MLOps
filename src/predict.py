from __future__ import annotations

import json
from typing import Any

import joblib
import pandas as pd

from src.config import load_settings, resolve_path


def load_model():
    settings = load_settings()
    model_path = resolve_path(settings.paths["best_model"])
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return joblib.load(model_path)


def predict_from_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    settings = load_settings()
    model = load_model()
    df = pd.DataFrame(records)
    probabilities = model.predict_proba(df)[:, 1]
    predictions = model.predict(df)

    return [
        {
            "prediction": int(prediction),
            "probability": round(float(probability), 4),
            "model_version": "local-dev",
            "target_name": settings.dataset["target"],
        }
        for prediction, probability in zip(predictions, probabilities)
    ]


def main() -> None:
    settings = load_settings()
    sample_path = resolve_path(settings.dataset["path"])
    df = pd.read_csv(sample_path).drop(columns=[settings.dataset["target"]]).head(2)
    outputs = predict_from_records(df.to_dict(orient="records"))
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
