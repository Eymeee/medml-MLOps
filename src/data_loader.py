from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import load_settings, resolve_path


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    settings = load_settings()
    dataset_path = resolve_path(path or settings.dataset["path"])
    df = pd.read_csv(dataset_path)
    return validate_schema(df)


def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    settings = load_settings()
    expected = (
        settings.dataset["numeric_features"]
        + settings.dataset["categorical_features"]
        + [settings.dataset["target"]]
    )
    missing = [column for column in expected if column not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes: {missing}")
    return df.copy()


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    settings = load_settings()
    cleaned = df.copy()

    if settings.preprocessing.get("drop_duplicates", True):
        cleaned = cleaned.drop_duplicates()

    numeric_cols = settings.dataset["numeric_features"]
    for col in numeric_cols:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    categorical_cols = settings.dataset["categorical_features"]
    for col in categorical_cols:
        cleaned[col] = cleaned[col].astype("object")

    if "age" in cleaned.columns:
        cleaned.loc[cleaned["age"] < 0, "age"] = pd.NA
    if "bmi" in cleaned.columns:
        cleaned.loc[cleaned["bmi"] <= 0, "bmi"] = pd.NA
    if "glucose" in cleaned.columns:
        cleaned.loc[cleaned["glucose"] <= 0, "glucose"] = pd.NA
    if "systolic_bp" in cleaned.columns:
        cleaned.loc[cleaned["systolic_bp"] <= 0, "systolic_bp"] = pd.NA
    if "diastolic_bp" in cleaned.columns:
        cleaned.loc[cleaned["diastolic_bp"] <= 0, "diastolic_bp"] = pd.NA

    return cleaned


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    settings = load_settings()
    target = settings.dataset["target"]
    if target not in df.columns:
        raise ValueError(f"Colonne cible absente: {target}")
    X = df.drop(columns=[target])
    y = df[target].astype(int)
    return X, y


def train_test_from_dataframe(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    settings = load_settings()
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=settings.dataset["test_size"],
        random_state=settings.dataset["random_state"],
        stratify=y,
    )
    return X_train, X_test, y_train, y_test
