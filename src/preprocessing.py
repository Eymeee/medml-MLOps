from __future__ import annotations

import json

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import load_settings, resolve_path
from src.data_loader import basic_cleaning, load_dataset, train_test_from_dataframe


def build_preprocessor() -> ColumnTransformer:
    settings = load_settings()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=settings.preprocessing["numeric_imputer"])),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=settings.preprocessing["categorical_imputer"])),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, settings.dataset["numeric_features"]),
            ("cat", categorical_pipeline, settings.dataset["categorical_features"]),
        ]
    )


def save_processed_splits() -> None:
    settings = load_settings()
    df = basic_cleaning(load_dataset())
    X_train, X_test, y_train, y_test = train_test_from_dataframe(df)

    train_df = X_train.copy()
    train_df[settings.dataset["target"]] = y_train.values

    test_df = X_test.copy()
    test_df[settings.dataset["target"]] = y_test.values

    train_path = resolve_path(settings.paths["processed_train"])
    test_path = resolve_path(settings.paths["processed_test"])
    schema_path = resolve_path(settings.paths["feature_schema"])

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    schema = {
        "numeric_features": settings.dataset["numeric_features"],
        "categorical_features": settings.dataset["categorical_features"],
        "target": settings.dataset["target"],
    }
    with open(schema_path, "w", encoding="utf-8") as handle:
        json.dump(schema, handle, indent=2)


def main() -> None:
    save_processed_splits()
    print("Processed train/test splits saved.")


if __name__ == "__main__":
    main()
