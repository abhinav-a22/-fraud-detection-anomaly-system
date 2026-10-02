"""Fraud Risk Scoring & Decision Engine Module.

Combines supervised fraud probability (XGBoost) and unsupervised anomaly score
(Isolation Forest) with automated SHAP reason codes into an operational risk engine.

Risk Tiers:
- LOW RISK    [0.00 to 0.30) -> Action: APPROVE
- MEDIUM RISK [0.30 to 0.70) -> Action: CHALLENGE_2FA (Step-up verification)
- HIGH RISK   [0.70 to 1.00] -> Action: DECLINE (Block and alert)

DISCLAIMER: These tier boundaries and combination weights are project-defined
operational parameters and must not be presented as universal industry constants.
"""

import json
import os
import sys
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import numpy as np
import pandas as pd
import shap

from src.anomaly_detection import SELECTED_ANOMALY_FEATURES, compute_normalized_anomaly_scores
from src.preprocessing import FEATURE_COLS, TARGET_COL


class FraudRiskEngine:
    """Production-style hybrid fraud risk scoring and decision engine."""

    def __init__(
        self,
        supervised_model_path: str = "models/final/final_supervised_model.joblib",
        anomaly_model_path: str = "models/final/final_isolation_forest.joblib",
        alpha_supervised: float = 0.70,
        low_threshold: float = 0.30,
        high_threshold: float = 0.70
    ):
        """Initialize models, weights, and decision tier thresholds."""
        if not os.path.exists(supervised_model_path):
            raise FileNotFoundError(f"Supervised model not found at {supervised_model_path}")
        if not os.path.exists(anomaly_model_path):
            raise FileNotFoundError(f"Anomaly model not found at {anomaly_model_path}")

        self.supervised_model = joblib.load(supervised_model_path)
        self.anomaly_model = joblib.load(anomaly_model_path)
        self.explainer = shap.TreeExplainer(self.supervised_model)

        self.alpha = alpha_supervised
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

        # Feature indices for anomaly model
        self.ano_feature_indices = [FEATURE_COLS.index(c) for c in SELECTED_ANOMALY_FEATURES]

        # Calibrated baseline anomaly score bounds from validation distribution
        self.ano_min = 0.3787
        self.ano_max = 0.7410

    def score_transaction(self, transaction_features: dict | pd.Series) -> dict:
        """Score a single incoming transaction and return decision, scores, and explanation."""
        if isinstance(transaction_features, dict):
            features_series = pd.Series(transaction_features)
        else:
            features_series = transaction_features

        # Ensure correct column ordering
        x_raw = features_series[FEATURE_COLS].to_numpy().reshape(1, -1)
        x_df = pd.DataFrame([features_series[FEATURE_COLS]])

        # 1. Supervised probability P(fraud)
        p_supervised = float(self.supervised_model.predict_proba(x_raw)[0, 1])

        # 2. Unsupervised anomaly score S(anomaly) calibrated against baseline bounds
        x_ano = x_raw[:, self.ano_feature_indices]
        raw_ano = -float(self.anomaly_model.score_samples(x_ano)[0])
        s_anomaly = float(np.clip((raw_ano - self.ano_min) / (self.ano_max - self.ano_min + 1e-8), 0.0, 1.0))

        # 3. Combined Risk Score
        # Formula: Score = alpha * P_sup + (1 - alpha) * S_ano
        combined_score = float(self.alpha * p_supervised + (1.0 - self.alpha) * s_anomaly)
        combined_score = np.clip(combined_score, 0.0, 1.0)

        # 4. Operational Risk Category & Recommended Action
        if combined_score < self.low_threshold:
            risk_category = "LOW"
            action = "APPROVE"
            action_desc = "Low-risk transaction. Process instantly without friction."
        elif combined_score < self.high_threshold:
            risk_category = "MEDIUM"
            action = "CHALLENGE_2FA"
            action_desc = "Moderate risk. Trigger step-up verification (SMS / push OTP)."
        else:
            risk_category = "HIGH"
            action = "DECLINE"
            action_desc = "High risk. Decline authorization and alert cardholder."

        # 5. Machine-Generated Explainability Reason Codes (SHAP)
        shap_exp = self.explainer(x_df)[0]
        reasons = []
        for col, val, s_val in zip(FEATURE_COLS, features_series[FEATURE_COLS], shap_exp.values):
            if abs(s_val) > 0.05:
                direction = "INCREASED RISK (+)" if s_val > 0 else "REDUCED RISK (-)"
                reasons.append({
                    "feature": col,
                    "value": round(float(val), 2),
                    "impact": round(float(s_val), 4),
                    "direction": direction
                })

        # Sort reasons by absolute impact
        reasons = sorted(reasons, key=lambda x: abs(x["impact"]), reverse=True)[:4]

        return {
            "fraud_probability": round(p_supervised, 4),
            "anomaly_score": round(s_anomaly, 4),
            "combined_risk_score": round(combined_score, 4),
            "risk_category": risk_category,
            "action": action,
            "action_description": action_desc,
            "explanation_reasons": reasons,
            "disclaimer": (
                "NOTE: Tier boundaries and combined score weighting (alpha=0.70) are project-defined "
                "parameters and do not represent universal industry standards."
            )
        }


def run_risk_engine_demo() -> dict:
    """Demonstrate the Risk Engine on sample transactions across all risk categories."""
    print("=" * 75)
    print("           PHASE 12: FRAUD RISK SCORING & DECISION ENGINE")
    print("=" * 75)

    engine = FraudRiskEngine()
    test_df = pd.read_csv("data/processed/test.csv")

    # Select representative exemplar transactions from Test set
    # 1. Clear Legitimate (Low Risk)
    low_sample = test_df[(test_df[TARGET_COL] == 0) & (test_df["TX_AMOUNT"] < 30.0)].iloc[0]

    # 2. Elevated Deviation (Medium Risk - CHALLENGE_2FA)
    med_sample = test_df.iloc[1167]  # TX ID 194087: P_sup=0.4867, S_ano=0.5626, Combined=0.5095

    # 3. High-Confidence Attack (High Risk - DECLINE)
    high_sample = test_df[(test_df[TARGET_COL] == 1) & (test_df["TX_AMOUNT"] > 250.0)].iloc[0]

    samples = [
        ("EXEMPLAR A: ROUTINE DAILY TRANSACTION [LOW TIER]", low_sample),
        ("EXEMPLAR B: UNUSUAL SPENDING SPIKE [MEDIUM TIER]", med_sample),
        ("EXEMPLAR C: HIGH-VELOCITY ATTACK [HIGH TIER]", high_sample),
    ]

    demo_results = []
    for title, row in samples:
        print("\n" + "-" * 75)
        print(f" {title} [TX ID: {int(row['TRANSACTION_ID'])}]")
        print(f" Ground Truth: {'FRAUD (1)' if row[TARGET_COL] == 1 else 'LEGITIMATE (0)'}")
        print(f" Raw Details : Amount=${row['TX_AMOUNT']:.2f}, Amount Ratio={row['CUSTOMER_AMOUNT_RATIO']:.2f}x, Night={int(row['TX_IS_NIGHT'])}")
        print("-" * 75)

        res = engine.score_transaction(row)
        demo_results.append(res)

        print(f"  Supervised Fraud Prob (P_sup) : {res['fraud_probability']:.4f}")
        print(f"  Anomaly Score (S_ano)         : {res['anomaly_score']:.4f}")
        print(f"  Combined Risk Score           : {res['combined_risk_score']:.4f}  [Formula: 0.70 * P_sup + 0.30 * S_ano]")
        print(f"  Decision Risk Tier            : [{res['risk_category']}]")
        print(f"  Recommended Action            : >>> {res['action']} <<<")
        print(f"  Policy Guidance               : {res['action_description']}")
        print("\n  Top Model Explanation Reasons:")
        for r in res["explanation_reasons"]:
            print(f"    * {r['direction']} {r['feature']} = {r['value']} (SHAP impact: {r['impact']:+.4f})")

    print("\n" + "=" * 75)
    print("Risk Engine executed successfully across all operating tiers.")
    print("=" * 75)

    os.makedirs("outputs/metrics", exist_ok=True)
    with open("outputs/metrics/risk_engine_demo_results.json", "w") as f:
        json.dump(demo_results, f, indent=4)

    return {"demo_results": demo_results}


if __name__ == "__main__":
    run_risk_engine_demo()
