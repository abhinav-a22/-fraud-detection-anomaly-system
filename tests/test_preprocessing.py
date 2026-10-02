"""Unit tests for preprocessing, chronological splitting, and train-only imputation."""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    FEATURE_COLS,
    TARGET_COL,
    chronological_split,
    handle_imputation,
)


@pytest.fixture
def synthetic_split_data():
    """Create a toy synthetic dataset spanning 10 consecutive timestamps."""
    dates = pd.date_range("2026-01-01", periods=10, freq="1D")
    df = pd.DataFrame({
        "TRANSACTION_ID": np.arange(10),
        "TX_DATETIME": dates,
        "CUSTOMER_ID": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        "TERMINAL_ID": [10, 10, 20, 20, 30, 30, 40, 40, 50, 50],
        "TX_AMOUNT": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
        "CUSTOMER_AVG_AMOUNT_1D": [np.nan, 10.0, np.nan, 30.0, np.nan, 50.0, np.nan, 70.0, np.nan, 90.0],
        "CUSTOMER_AVG_AMOUNT_7D": [np.nan, 10.0, np.nan, 30.0, np.nan, 50.0, np.nan, 70.0, np.nan, 90.0],
        "TERMINAL_AVG_AMOUNT_1D": [np.nan, 15.0, np.nan, 35.0, np.nan, 55.0, np.nan, 75.0, np.nan, 95.0],
        "TX_FRAUD": [0, 0, 0, 0, 0, 1, 0, 0, 1, 0]
    })
    return df


def test_chronological_ordering(synthetic_split_data):
    """Verify that chronological splits guarantee PAST -> VALIDATION -> FUTURE ordering."""
    train_df, val_df, test_df = chronological_split(synthetic_split_data, train_ratio=0.6, val_ratio=0.2)

    # 1. Size checks (6 train, 2 val, 2 test)
    assert len(train_df) == 6
    assert len(val_df) == 2
    assert len(test_df) == 2

    # 2. Strict chronological boundaries: max(train) < min(val) and max(val) < min(test)
    assert train_df["TX_DATETIME"].max() < val_df["TX_DATETIME"].min()
    assert val_df["TX_DATETIME"].max() < test_df["TX_DATETIME"].min()


def test_train_only_imputation(synthetic_split_data):
    """Verify that imputation values are derived solely from Train partition."""
    train_df, val_df, test_df = chronological_split(synthetic_split_data, train_ratio=0.6, val_ratio=0.2)

    # Calculate expected train median manually
    train_cust_median = float(train_df["CUSTOMER_AVG_AMOUNT_7D"].dropna().median())

    train_imp, val_imp, test_imp, impute_vals = handle_imputation(train_df, val_df, test_df)

    # 1. Imputation dict matches train median
    assert impute_vals["CUSTOMER_AVG_AMOUNT_7D"] == train_cust_median

    # 2. No remaining NaNs in any split
    assert train_imp["CUSTOMER_AVG_AMOUNT_7D"].isnull().sum() == 0
    assert val_imp["CUSTOMER_AVG_AMOUNT_7D"].isnull().sum() == 0
    assert test_imp["CUSTOMER_AVG_AMOUNT_7D"].isnull().sum() == 0

    # 3. Check that the imputed value applied to Val equals train median
    assert val_imp.loc[0, "CUSTOMER_AVG_AMOUNT_7D"] == train_cust_median


def test_processed_files_exist():
    """Verify that processed data files exist on disk with zero NaNs."""
    for split in ["train", "val", "test"]:
        path = f"data/processed/{split}.csv"
        assert os.path.exists(path), f"Processed split {path} missing"
        df = pd.read_csv(path)
        assert len(df) > 0
        assert df[FEATURE_COLS].isnull().sum().sum() == 0, f"Found unexpected NaNs in {split}.csv"

