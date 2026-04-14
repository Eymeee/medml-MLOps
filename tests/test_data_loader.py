from __future__ import annotations

import pandas as pd
import pytest

from src.data_loader import basic_cleaning, split_features_target, validate_schema


def test_validate_schema_accepts_expected_columns(synthetic_df):
    validated = validate_schema(synthetic_df)
    assert isinstance(validated, pd.DataFrame)
    assert "outcome" in validated.columns


def test_validate_schema_rejects_missing_target(synthetic_df):
    broken = synthetic_df.drop(columns=["outcome"])
    with pytest.raises(ValueError):
        validate_schema(broken)


def test_basic_cleaning_handles_invalid_numeric_values(synthetic_df):
    synthetic_df.loc[0, "age"] = -5
    synthetic_df.loc[1, "glucose"] = 0
    cleaned = basic_cleaning(synthetic_df)
    assert pd.isna(cleaned.loc[0, "age"])
    assert pd.isna(cleaned.loc[1, "glucose"])


def test_split_features_target_returns_binary_target(synthetic_df):
    X, y = split_features_target(synthetic_df)
    assert "outcome" not in X.columns
    assert set(y.unique()).issubset({0, 1})
