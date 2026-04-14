from __future__ import annotations

import numpy as np

from src.evaluate import compute_classification_metrics


def test_metrics_are_in_unit_interval():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_proba = np.array([0.1, 0.6, 0.8, 0.9])
    metrics = compute_classification_metrics(y_true, y_pred, y_proba)
    assert set(metrics.keys()) == {"accuracy", "precision", "recall", "f1_score", "auc_roc"}
    for value in metrics.values():
        assert 0.0 <= value <= 1.0
