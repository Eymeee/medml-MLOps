from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.evaluate import compute_classification_metrics
from src.preprocessing import build_preprocessor


def test_end_to_end_on_tiny_sample(synthetic_df):
    sample = synthetic_df.head(10).copy()
    X = sample.drop(columns=["outcome"])
    y = sample["outcome"]
    pipeline = Pipeline(
        steps=[("preprocessing", build_preprocessor()), ("model", LogisticRegression(max_iter=200))]
    )
    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    probas = pipeline.predict_proba(X)[:, 1]
    metrics = compute_classification_metrics(y, preds, probas)
    assert metrics["accuracy"] >= 0.0
