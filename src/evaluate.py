from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import load_settings, resolve_path


def compute_classification_metrics(y_true, y_pred, y_proba) -> dict[str, float]:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "auc_roc": round(float(roc_auc_score(y_true, y_proba)), 4),
    }


def save_evaluation_artifacts(y_true, y_pred, y_proba, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    cm_path = output_dir / "confusion_matrix.png"
    roc_path = output_dir / "roc_curve.png"
    report_path = output_dir / "classification_report.txt"

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred)).plot(ax=ax, colorbar=False)
    plt.tight_layout()
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(roc_path, dpi=150)
    plt.close(fig)

    report = classification_report(y_true, y_pred)
    report_path.write_text(report, encoding="utf-8")

    return {
        "confusion_matrix": str(cm_path),
        "roc_curve": str(roc_path),
        "classification_report": str(report_path),
    }


def evaluate_saved_model() -> dict[str, Any]:
    settings = load_settings()
    model_path = resolve_path(settings.paths["best_model"])
    test_path = resolve_path(settings.paths["processed_test"])
    report_dir = resolve_path(settings.paths["evaluation_dir"])

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = joblib.load(model_path)
    test_df = pd.read_csv(test_path)

    target = settings.dataset["target"]
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = y_pred

    metrics = compute_classification_metrics(y_test, y_pred, y_proba)
    artifacts = save_evaluation_artifacts(y_test, y_pred, y_proba, report_dir)

    payload = {"metrics": metrics, "artifacts": artifacts}
    with open(report_dir / "evaluation_summary.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    return payload


def main() -> None:
    result = evaluate_saved_model()
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
