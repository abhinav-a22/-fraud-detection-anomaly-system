"""Unit tests for online prediction, FraudRiskEngine, and output validation."""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import pytest

from src.predict import FraudRiskEngine
from src.preprocessing import FEATURE_COLS


@pytest.fixture(scope="module")
def engine():
    """Load pre-trained FraudRiskEngine artifact."""
    return FraudRiskEngine(
        supervised_model_path="models/final/final_supervised_model.joblib",
        anomaly_model_path="models/final/final_isolation_forest.joblib"
    )


@pytest.fixture
def sample_features():
    """Create a valid transaction feature dictionary."""
    return {
        "TX_AMOUNT": 45.0,
        "TX_AMOUNT_LOG": float(np.log1p(45.0)),
        "TX_HOUR": 14,
        "TX_DAY_OF_WEEK": 2,
        "TX_IS_WEEKEND": 0,
        "TX_IS_NIGHT": 0,
        "CUSTOMER_TX_COUNT_1H": 0.0,
        "CUSTOMER_TX_COUNT_24H": 1.0,
        "CUSTOMER_TX_COUNT_7D": 5.0,
        "CUSTOMER_AVG_AMOUNT_1D": 40.0,
        "CUSTOMER_AVG_AMOUNT_7D": 42.0,
        "CUSTOMER_AMOUNT_RATIO": 1.07,
        "TERMINAL_TX_COUNT_1D": 1.0,
        "TERMINAL_AVG_AMOUNT_1D": 50.0
    }


def test_score_ranges(engine, sample_features):
    """Verify that all scores fall strictly in [0.0, 1.0]."""
    res = engine.score_transaction(sample_features)

    assert 0.0 <= res["fraud_probability"] <= 1.0, "Fraud probability outside [0, 1]"
    assert 0.0 <= res["anomaly_score"] <= 1.0, "Anomaly score outside [0, 1]"
    assert 0.0 <= res["combined_risk_score"] <= 1.0, "Combined risk score outside [0, 1]"


def test_decision_tier_consistency(engine, sample_features):
    """Verify that risk tier maps strictly to defined business action."""
    res = engine.score_transaction(sample_features)

    assert res["risk_category"] in ["LOW", "MEDIUM", "HIGH"]
    assert res["action"] in ["APPROVE", "CHALLENGE_2FA", "DECLINE"]

    if res["risk_category"] == "LOW":
        assert res["action"] == "APPROVE"
    elif res["risk_category"] == "MEDIUM":
        assert res["action"] == "CHALLENGE_2FA"
    elif res["risk_category"] == "HIGH":
        assert res["action"] == "DECLINE"


def test_explainability_reason_structure(engine, sample_features):
    """Verify that explainability reasons contain required metadata."""
    res = engine.score_transaction(sample_features)
    reasons = res["explanation_reasons"]

    assert isinstance(reasons, list)
    for r in reasons:
        assert "feature" in r
        assert "value" in r
        assert "impact" in r
        assert "direction" in r
        assert r["feature"] in FEATURE_COLS


def test_missing_feature_validation(engine, sample_features):
    """Verify that missing required features raises KeyError."""
    incomplete = sample_features.copy()
    del incomplete["TX_AMOUNT"]

    with pytest.raises(KeyError):
        engine.score_transaction(incomplete)

