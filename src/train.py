from __future__ import annotations

import json
import time
from dataclasses import dataclass

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.config import load_settings, resolve_path
from src.data_loader import split_features_target
from src.evaluate import compute_classification_metrics
from src.preprocessing import build_preprocessor

try:
    import mlflow
    import mlflow.sklearn
except Exception:  # pragma: no cover - fallback when mlflow is absent locally
    mlflow = None


@dataclass
class TrainingResult:
    model_name: str
    metrics: dict[str, float]
    best_params: dict
    train_seconds: float


def get_model_candidates() -> dict[str, tuple[object, dict]]:
    settings = load_settings()
    models_cfg = settings.modeling["models"]

    candidates: dict[str, tuple[object, dict]] = {}

    if models_cfg["mlp"]["enabled"]:
        candidates["mlp"] = (
            MLPClassifier(random_state=settings.dataset["random_state"]),
            models_cfg["mlp"]["params"],
        )

    if models_cfg["svm"]["enabled"]:
        candidates["svm"] = (
            SVC(random_state=settings.dataset["random_state"]),
            models_cfg["svm"]["params"],
        )

    if models_cfg["xgboost"]["enabled"]:
        candidates["xgboost"] = (
            XGBClassifier(
                random_state=settings.dataset["random_state"],
                eval_metric="logloss",
                n_jobs=1,
            ),
            models_cfg["xgboost"]["params"],
        )

    return candidates


def maybe_start_mlflow():
    settings = load_settings()
    if mlflow is None or not settings.modeling.get("track_with_mlflow", False):
        return None
    mlflow.set_experiment(settings.modeling["experiment_name"])
    return mlflow


def train_models() -> dict:
    settings = load_settings()
    train_df = pd.read_csv(resolve_path(settings.paths["processed_train"]))
    test_df = pd.read_csv(resolve_path(settings.paths["processed_test"]))

    target = settings.dataset["target"]
    X_train, y_train = split_features_target(train_df)
    X_test, y_test = split_features_target(test_df)

    results: list[TrainingResult] = []
    best_pipeline = None
    best_result = None
    best_score = -1.0

    mlflow_client = maybe_start_mlflow()

    for model_name, (base_model, param_grid) in get_model_candidates().items():
        preprocessor = build_preprocessor()
        pipeline = Pipeline(
            steps=[
                ("preprocessing", preprocessor),
                ("model", clone(base_model)),
            ]
        )
        grid = {f"model__{k}": v for k, v in param_grid.items()}

        cv = StratifiedKFold(
            n_splits=settings.modeling["cv_folds"],
            shuffle=True,
            random_state=settings.dataset["random_state"],
        )

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=grid,
            scoring=settings.modeling["primary_metric"],
            cv=cv,
            n_jobs=1,
        )

        start = time.time()
        if mlflow_client is not None:
            with mlflow_client.start_run(run_name=model_name):
                search.fit(X_train, y_train)
                duration = time.time() - start
                y_pred = search.best_estimator_.predict(X_test)
                y_proba = search.best_estimator_.predict_proba(X_test)[:, 1]
                metrics = compute_classification_metrics(y_test, y_pred, y_proba)
                mlflow_client.log_params(search.best_params_)
                mlflow_client.log_metrics(metrics)
        else:
            search.fit(X_train, y_train)
            duration = time.time() - start
            y_pred = search.best_estimator_.predict(X_test)
            y_proba = search.best_estimator_.predict_proba(X_test)[:, 1]
            metrics = compute_classification_metrics(y_test, y_pred, y_proba)

        result = TrainingResult(
            model_name=model_name,
            metrics=metrics,
            best_params=search.best_params_,
            train_seconds=round(duration, 3),
        )
        results.append(result)

        metric_key = settings.modeling["primary_metric"]
        score = metrics["f1_score" if metric_key == "f1" else metric_key]
        if score > best_score:
            best_score = score
            best_pipeline = search.best_estimator_
            best_result = result

    if best_pipeline is None or best_result is None:
        raise RuntimeError("No model was trained.")

    model_path = resolve_path(settings.paths["best_model"])
    metadata_path = resolve_path(settings.paths["model_metadata"])
    metrics_csv_path = resolve_path(settings.paths["metrics_csv"])

    comparison_df = pd.DataFrame(
        [
            {
                "model": r.model_name,
                **r.metrics,
                "train_seconds": r.train_seconds,
                "best_params": json.dumps(r.best_params),
            }
            for r in results
        ]
    )
    comparison_df.to_csv(metrics_csv_path, index=False)

    joblib.dump(best_pipeline, model_path)
    metadata = {
        "best_model_name": best_result.model_name,
        "best_params": best_result.best_params,
        "metrics": best_result.metrics,
        "comparison_table": comparison_df.to_dict(orient="records"),
    }
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return metadata


def main() -> None:
    settings = load_settings()
    processed_train = resolve_path(settings.paths["processed_train"])
    processed_test = resolve_path(settings.paths["processed_test"])
    if not processed_train.exists() or not processed_test.exists():
        from src.preprocessing import save_processed_splits

        save_processed_splits()

    metadata = train_models()
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
