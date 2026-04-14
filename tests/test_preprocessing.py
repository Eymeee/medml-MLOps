from __future__ import annotations

import numpy as np
from sklearn.pipeline import Pipeline

from src.preprocessing import build_preprocessor


def test_preprocessor_fits_without_missing_values(synthetic_df):
    X = synthetic_df.drop(columns=["outcome"])
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(X)
    dense = transformed.toarray() if hasattr(transformed, "toarray") else transformed
    assert dense.shape[0] == len(X)
    assert not np.isnan(dense).any()


def test_pipeline_produces_binary_predictions(synthetic_df):
    from sklearn.linear_model import LogisticRegression

    X = synthetic_df.drop(columns=["outcome"])
    y = synthetic_df["outcome"]
    pipeline = Pipeline(
        steps=[("preprocessing", build_preprocessor()), ("model", LogisticRegression(max_iter=200))]
    )
    pipeline.fit(X, y)
    preds = pipeline.predict(X.head(10))
    assert set(preds).issubset({0, 1})
