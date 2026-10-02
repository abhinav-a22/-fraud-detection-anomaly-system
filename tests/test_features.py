"""Unit tests for feature engineering and data leakage integrity."""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import pytest
from src.features import (
    add_transaction_features,
    add_time_features,
    add_customer_behavioral_features,
    add_terminal_behavioral_features,
    engineer_all_features,
    verify_no_leakage,
)


@pytest.fixture
def sample_raw_data():
    """Create a minimal synthetic transaction sequence for unit testing."""
    records = [
        {"TRANSACTION_ID": 0, "TX_DATETIME": "2026-01-01 01:00:00", "CUSTOMER_ID": 1, "TERMINAL_ID": 10, "TX_AMOUNT": 10.0, "TX_FRAUD": 0},
        {"TRANSACTION_ID": 1, "TX_DATETIME": "2026-01-01 01:30:00", "CUSTOMER_ID": 1, "TERMINAL_ID": 10, "TX_AMOUNT": 20.0, "TX_FRAUD": 0},
        {"TRANSACTION_ID": 2, "TX_DATETIME": "2026-01-01 03:00:00", "CUSTOMER_ID": 1, "TERMINAL_ID": 11, "TX_AMOUNT": 30.0, "TX_FRAUD": 0},
        {"TRANSACTION_ID": 3, "TX_DATETIME": "2026-01-01 05:00:00", "CUSTOMER_ID": 2, "TERMINAL_ID": 10, "TX_AMOUNT": 50.0, "TX_FRAUD": 0},
        {"TRANSACTION_ID": 4, "TX_DATETIME": "2026-01-01 12:00:00", "CUSTOMER_ID": 1, "TERMINAL_ID": 10, "TX_AMOUNT": 200.0, "TX_FRAUD": 1},
    ]
    df = pd.DataFrame(records)
    df["TX_DATETIME"] = pd.to_datetime(df["TX_DATETIME"])
    return df


def test_transaction_features(sample_raw_data):
    """Test log transformation math."""
    res = add_transaction_features(sample_raw_data)
    assert "TX_AMOUNT_LOG" in res.columns
    expected = np.log1p(sample_raw_data["TX_AMOUNT"].values)
    np.testing.assert_allclose(res["TX_AMOUNT_LOG"].values, expected, rtol=1e-5)


def test_time_features(sample_raw_data):
    """Test diurnal flags and hour extractions."""
    res = add_time_features(sample_raw_data)
    assert res.loc[0, "TX_HOUR"] == 1
    assert res.loc[0, "TX_IS_NIGHT"] == 1  # 01:00 is night
    assert res.loc[4, "TX_HOUR"] == 12
    assert res.loc[4, "TX_IS_NIGHT"] == 0  # 12:00 is day


def test_customer_behavioral_rolling(sample_raw_data):
    """Verify closed='left' behavior: first transaction must have zero prior count."""
    res = add_customer_behavioral_features(sample_raw_data)
    # At tx 0 (01:00), customer 1 has 0 prior transactions
    assert res.loc[0, "CUSTOMER_TX_COUNT_1H"] == 0
    assert res.loc[0, "CUSTOMER_TX_COUNT_24H"] == 0

    # At tx 1 (01:30), customer 1 had 1 prior transaction within 1h (at 01:00)
    assert res.loc[1, "CUSTOMER_TX_COUNT_1H"] == 1
    assert res.loc[1, "CUSTOMER_TX_COUNT_24H"] == 1

    # At tx 2 (03:00), customer 1 had 0 prior transactions within 1h (previous was 01:30), but 2 within 24h
    assert res.loc[2, "CUSTOMER_TX_COUNT_1H"] == 0
    assert res.loc[2, "CUSTOMER_TX_COUNT_24H"] == 2


def test_leakage_verification_proof():
    """Verify that verify_no_leakage passes on a slice of generated data."""
    raw_path = "data/raw/transactions.csv"
    if os.path.exists(raw_path):
        df_slice = pd.read_csv(raw_path, nrows=5000)
        df_slice["TX_DATETIME"] = pd.to_datetime(df_slice["TX_DATETIME"])
        featured = engineer_all_features(df_slice)
        is_safe = verify_no_leakage(featured, sample_idx=2500)
        assert is_safe is True
