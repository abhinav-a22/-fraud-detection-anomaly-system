"""Lightweight Streamlit Dashboard for Fraud Detection & Anomaly Analytics System.

Provides an interactive user interface for:
1. Real-time hybrid transaction risk scoring (XGBoost + Isolation Forest)
2. Interactive decision routing (APPROVE, CHALLENGE_2FA, DECLINE)
3. Instant model explainability (SHAP reason codes)
4. Interactive benchmark exploration (Experiments E0 to E8 & diagnostic plots)
"""

import json
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.predict import FraudRiskEngine
from src.preprocessing import FEATURE_COLS, TARGET_COL


st.set_page_config(
    page_title="Fraud Detection & Anomaly Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def get_risk_engine():
    """Load cached risk engine artifact."""
    return FraudRiskEngine()


@st.cache_data
def load_test_samples():
    """Load test dataset for interactive exploration."""
    test_path = "data/processed/test.csv"
    if os.path.exists(test_path):
        return pd.read_csv(test_path)
    return None


@st.cache_data
def load_experiment_metrics():
    """Load experiment comparison metrics."""
    metrics_path = "outputs/metrics/experiments.csv"
    if os.path.exists(metrics_path):
        return pd.read_csv(metrics_path, keep_default_na=False)
    return None


def main():
    st.title("🛡️ Fraud Detection & Anomaly Analytics System")
    st.markdown(
        "**Hybrid Human-in-the-Loop Machine Learning Pipeline**: "
        "Supervised Gradient Boosted Trees (XGBoost) + Unsupervised Outlier Detection (Isolation Forest) "
        "+ Game-Theoretic Explainability (SHAP)."
    )
    st.divider()

    engine = get_risk_engine()
    test_df = load_test_samples()
    exp_df = load_experiment_metrics()

    # Tabs for main navigation
    tab_scoring, tab_experiments, tab_diagnostics = st.tabs([
        "⚡ Live Risk Scoring Engine",
        "📊 Experiment Matrix & Benchmarks",
        "📈 Diagnostic Visualizations"
    ])

    # ---------------- TAB 1: LIVE RISK SCORING ----------------
    with tab_scoring:
        col_ctrl, col_display = st.columns([1, 2])

        with col_ctrl:
            st.subheader("1. Transaction Input")
            input_mode = st.radio(
                "Choose Input Mode:",
                ["Preset Test Cases", "Manual Parameter Simulation"],
                horizontal=True
            )

            if input_mode == "Preset Test Cases" and test_df is not None:
                preset = st.selectbox(
                    "Select a Preset Exemplar from Test Set:",
                    [
                        "Exemplar A: Routine Small Purchase (Low Risk)",
                        "Exemplar B: Unusually Elevated Spend (Medium Risk)",
                        "Exemplar C: Suspicious High-Velocity Attack (High Risk)"
                    ]
                )
                if "Exemplar A" in preset:
                    sample_row = test_df[(test_df[TARGET_COL] == 0) & (test_df["TX_AMOUNT"] < 25.0)].iloc[0]
                elif "Exemplar B" in preset:
                    sample_row = test_df.iloc[1167]
                else:
                    sample_row = test_df[(test_df[TARGET_COL] == 1) & (test_df["TX_AMOUNT"] > 250.0)].iloc[0]

                tx_data = sample_row.to_dict()

            else:
                st.markdown("**Simulate Transaction Features:**")
                tx_amt = st.slider("Transaction Amount ($)", min_value=1.0, max_value=800.0, value=125.0, step=1.0)
                tx_ratio = st.slider("Customer Amount Ratio (vs 7d avg)", min_value=0.1, max_value=10.0, value=2.5, step=0.1)
                tx_night = st.selectbox("Transaction Time Window", ["Daytime (06:00 - 23:59)", "Night Window (00:00 - 06:00)"])
                tx_count_1h = st.slider("Customer 1-Hour Velocity (#)", 0, 10, 0)
                tx_count_24h = st.slider("Customer 24-Hour Velocity (#)", 0, 20, 2)

                tx_data = {
                    "TX_AMOUNT": tx_amt,
                    "TX_AMOUNT_LOG": float(np.log1p(tx_amt)),
                    "TX_HOUR": 2 if "Night" in tx_night else 14,
                    "TX_DAY_OF_WEEK": 2,
                    "TX_IS_WEEKEND": 0,
                    "TX_IS_NIGHT": 1 if "Night" in tx_night else 0,
                    "CUSTOMER_TX_COUNT_1H": float(tx_count_1h),
                    "CUSTOMER_TX_COUNT_24H": float(tx_count_24h),
                    "CUSTOMER_TX_COUNT_7D": 7.0,
                    "CUSTOMER_AVG_AMOUNT_1D": 45.0,
                    "CUSTOMER_AVG_AMOUNT_7D": tx_amt / max(0.01, tx_ratio),
                    "CUSTOMER_AMOUNT_RATIO": float(tx_ratio),
                    "TERMINAL_TX_COUNT_1D": 1.0,
                    "TERMINAL_AVG_AMOUNT_1D": 50.0,
                }

            st.subheader("2. Policy Weighting")
            alpha = st.slider("Supervised Weight (alpha)", 0.0, 1.0, 0.70, 0.05,
                              help="Alpha balances Supervised XGBoost (alpha) vs. Unsupervised Isolation Forest (1 - alpha).")
            engine.alpha = alpha

        # Scoring execution
        with col_display:
            st.subheader("2. Evaluation & Decision Routing")
            res = engine.score_transaction(tx_data)

            # Metric Cards
            m1, m2, m3 = st.columns(3)
            m1.metric("Supervised Prob (P_sup)", f"{res['fraud_probability']:.4f}")
            m2.metric("Anomaly Score (S_ano)", f"{res['anomaly_score']:.4f}")
            m3.metric("Combined Risk Score", f"{res['combined_risk_score']:.4f}")

            # Visual Decision Banner
            category = res["risk_category"]
            action = res["action"]

            if category == "LOW":
                st.success(f"### DECISION: {action} (LOW RISK)")
            elif category == "MEDIUM":
                st.warning(f"### DECISION: {action} (MEDIUM RISK - 2FA CHALLENGE)")
            else:
                st.error(f"### DECISION: {action} (HIGH RISK - DECLINED)")

            st.info(f"**Policy Action:** {res['action_description']}")

            # SHAP Reason Codes
            st.subheader("3. Model Adverse Action & Explainability (SHAP)")
            st.markdown("Factors driving this specific transaction prediction:")
            for reason in res["explanation_reasons"]:
                badge = "🔴" if "+" in reason["direction"] else "🟢"
                st.write(
                    f"{badge} **{reason['direction']}** `{reason['feature']}` = `{reason['value']}` "
                    f"(SHAP Impact: `{reason['impact']:+.4f}`)"
                )

            st.caption(res["disclaimer"])

    # ---------------- TAB 2: EXPERIMENT MATRIX ----------------
    with tab_experiments:
        st.subheader("Full Experimental Evaluation Matrix (Chronological Validation Set)")
        st.markdown(
            "Every metric represents actual verified model execution across experiments E0 through E8. "
            "Notice how accuracy remains misleadingly high (~98.7%) across all models while PR-AUC exposes the true ranking capability."
        )
        if exp_df is not None:
            st.dataframe(exp_df, use_container_width=True)
        else:
            st.info("Run `python src/evaluate.py` to view experiment table.")

        # Final Test Set Card
        test_metrics_path = "outputs/metrics/final_test_metrics.json"
        if os.path.exists(test_metrics_path):
            with open(test_metrics_path) as f:
                test_summary = json.load(f)
            st.subheader("Final Unbiased Test Set Generalization (Touched Once)")
            tm1, tm2, tm3, tm4 = st.columns(4)
            tm1.metric("Test PR-AUC", f"{test_summary['test_metrics']['PR-AUC']:.4f}")
            tm2.metric("Tuned Threshold", f"theta* = {test_summary['optimal_threshold']:.2f}")
            tm3.metric("Test Precision", f"{test_summary['test_metrics']['tuned_threshold']['precision']:.4f}")
            tm4.metric("Test Recall", f"{test_summary['test_metrics']['tuned_threshold']['recall']:.4f}")

    # ---------------- TAB 3: DIAGNOSTIC VISUALIZATIONS ----------------
    with tab_diagnostics:
        st.subheader("Diagnostic Visualizations & Performance Curves")
        diag_choice = st.selectbox(
            "Select Diagnostic Figure to Inspect:",
            [
                "Precision-Recall & ROC Curves (Phase 6)",
                "Decision Threshold Tuning Curves (Phase 8)",
                "Cross-Experiment Metrics Comparison (Phase 7)",
                "Supervised vs. Anomaly Quadrants (Phase 9)",
                "Error Analysis: False Positives & Negatives (Phase 10)",
                "SHAP Global Feature Importance Beeswarm (Phase 11)",
                "SHAP Waterfall Explanation Example (Phase 11)",
            ]
        )

        plot_mapping = {
            "Precision-Recall & ROC Curves (Phase 6)": "outputs/plots/curves_phase6.png",
            "Decision Threshold Tuning Curves (Phase 8)": "outputs/plots/threshold_tuning_curve.png",
            "Cross-Experiment Metrics Comparison (Phase 7)": "outputs/plots/experiment_metrics_comparison.png",
            "Supervised vs. Anomaly Quadrants (Phase 9)": "outputs/plots/supervised_vs_unsupervised_quadrants.png",
            "Error Analysis: False Positives & Negatives (Phase 10)": "outputs/plots/error_analysis_amounts_and_ratios.png",
            "SHAP Global Feature Importance Beeswarm (Phase 11)": "outputs/plots/shap_summary_beeswarm.png",
            "SHAP Waterfall Explanation Example (Phase 11)": "outputs/plots/shap_waterfall_tp.png",
        }

        target_file = plot_mapping.get(diag_choice)
        if target_file and os.path.exists(target_file):
            st.image(target_file, use_container_width=True)
        else:
            st.warning(f"Plot file not found at {target_file}. Run the corresponding phase script first.")


if __name__ == "__main__":
    main()

