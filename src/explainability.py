"""SHAP Model Explainability Module.

Computes exact Shapley values using TreeExplainer on the final XGBoost model.
Generates:
1. Global feature importance (Mean |SHAP| bar chart & Beeswarm plot)
2. Local individual transaction explanations (Waterfalls & reason codes)
3. Quantitative breakdown of positive vs negative forces pushing fraud probability
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.preprocessing import FEATURE_COLS, TARGET_COL


def load_model_and_data(
    model_path: str = "models/final/final_supervised_model.joblib",
    val_path: str = "data/processed/val.csv"
) -> tuple:
    """Load final model and validation dataset."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run src/threshold_tuning.py first.")
    model = joblib.load(model_path)
    val_df = pd.read_csv(val_path)
    return model, val_df


def compute_global_shap(
    model,
    X_sample: pd.DataFrame,
    output_dir: str = "outputs/plots"
) -> shap.Explanation:
    """Compute and export global SHAP beeswarm and bar summary plots."""
    print("Computing TreeSHAP values for global explanation sample (2,000 transactions)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)

    os.makedirs(output_dir, exist_ok=True)

    # 1. Global Bar Plot (Mean |SHAP| ranking)
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.bar(shap_values, max_display=12, show=False)
    plt.title("Global Feature Importance (Mean |SHAP Value| Across Validation Set)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    bar_path = os.path.join(output_dir, "shap_global_bar.png")
    plt.savefig(bar_path, dpi=200)
    plt.close()
    print(f"[Plot Saved] Global SHAP feature ranking -> {bar_path}")

    # 2. Beeswarm Plot (Directional feature value impact)
    fig, ax = plt.subplots(figsize=(11, 6.5))
    shap.plots.beeswarm(shap_values, max_display=12, show=False)
    plt.title("SHAP Summary Beeswarm Plot (Feature Values vs. Fraud Impact)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    beeswarm_path = os.path.join(output_dir, "shap_summary_beeswarm.png")
    plt.savefig(beeswarm_path, dpi=200)
    plt.close()
    print(f"[Plot Saved] SHAP beeswarm plot -> {beeswarm_path}")

    return shap_values


def explain_individual_transaction(
    explainer: shap.TreeExplainer,
    model,
    row_features: pd.Series,
    tx_id: int,
    true_label: int,
    cohort_name: str,
    output_plot_path: str | None = None
) -> dict:
    """Deconstruct exact local Shapley forces for a single transaction."""
    x_df = pd.DataFrame([row_features])
    prob_fraud = float(model.predict_proba(x_df)[0, 1])

    # Compute single-row SHAP Explanation
    shap_exp = explainer(x_df)[0]
    base_val = float(shap_exp.base_values)
    values = shap_exp.values

    # Partition features into forces pushing TOWARD fraud (+) and TOWARD legitimate (-)
    forces = []
    for col, val, shap_val in zip(FEATURE_COLS, row_features.values, values):
        forces.append({
            "feature": col,
            "value": float(val),
            "shap_value": round(float(shap_val), 4),
            "direction": "Pushes toward FRAUD (+)" if shap_val > 0 else "Pushes toward LEGIT (-)"
        })

    forces_df = pd.DataFrame(forces)
    fraud_push = forces_df[forces_df["shap_value"] > 0].sort_values(by="shap_value", ascending=False)
    legit_push = forces_df[forces_df["shap_value"] < 0].sort_values(by="shap_value", ascending=True)

    print("\n" + "=" * 70)
    print(f" SHAP INDIVIDUAL TRANSACTION EXPLANATION [TX ID: {tx_id}]")
    print(f" Cohort: {cohort_name} | True Label: {true_label} | Predicted Fraud Prob: {prob_fraud:.4f}")
    print(f" Model Base Log-Odds: {base_val:.4f} (Base Fraud Prevalence: {1 / (1 + np.exp(-base_val)):.2%})")
    print("=" * 70)

    print("Top Factors Pushing Prediction TOWARD FRAUD (+):")
    if len(fraud_push) > 0:
        for _, r in fraud_push.head(4).iterrows():
            print(f"  [+] {r['feature']:<24} = {r['value']:>8.2f} (SHAP: +{r['shap_value']:.4f})")
    else:
        print("  None")

    print("\nTop Factors Pushing Prediction TOWARD LEGITIMATE (-):")
    if len(legit_push) > 0:
        for _, r in legit_push.head(4).iterrows():
            print(f"  [-] {r['feature']:<24} = {r['value']:>8.2f} (SHAP: {r['shap_value']:.4f})")
    else:
        print("  None")
    print("=" * 70)

    # Optional local waterfall plot
    if output_plot_path:
        fig, ax = plt.subplots(figsize=(9, 5))
        shap.plots.waterfall(shap_exp, max_display=10, show=False)
        plt.title(f"SHAP Waterfall Explanation [TX {tx_id} - {cohort_name}] (P={prob_fraud:.3f})",
                  fontsize=11, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_plot_path, dpi=200)
        plt.close()
        print(f"[Plot Saved] Individual waterfall plot -> {output_plot_path}")

    return {
        "transaction_id": int(tx_id),
        "cohort": cohort_name,
        "true_label": int(true_label),
        "predicted_probability": round(prob_fraud, 4),
        "base_log_odds": round(base_val, 4),
        "top_fraud_factors": fraud_push.head(4).to_dict(orient="records"),
        "top_legit_factors": legit_push.head(4).to_dict(orient="records")
    }


def run_explainability_pipeline() -> dict:
    """Execute full SHAP explainability pipeline on global sample and error cohorts."""
    print("=" * 70)
    print("        PHASE 11: SHAP (SHapley Additive exPlanations) PIPELINE")
    print("=" * 70)

    model, val_df = load_model_and_data()
    X_val = val_df[FEATURE_COLS]
    y_val = val_df[TARGET_COL]

    # Predict probabilities to identify cohort exemplars
    val_probas = model.predict_proba(X_val.to_numpy())[:, 1]
    val_df["P_FRAUD"] = val_probas
    val_df["PRED"] = (val_probas >= 0.20).astype(int)

    # 1. Global Explanations (Sample of 2,000 transactions)
    np.random.seed(42)
    sample_indices = np.random.choice(len(val_df), size=min(2000, len(val_df)), replace=False)
    X_sample = X_val.iloc[sample_indices]
    compute_global_shap(model, X_sample)

    # 2. Individual Explanations on Distinct Cohorts
    explainer = shap.TreeExplainer(model)
    explanations = []

    # Case A: True Positive (Caught Fraud)
    tp_candidates = val_df[(val_df[TARGET_COL] == 1) & (val_df["PRED"] == 1)].sort_values(by="P_FRAUD", ascending=False)
    if len(tp_candidates) > 0:
        tp_row = tp_candidates.iloc[0]
        res_tp = explain_individual_transaction(
            explainer=explainer,
            model=model,
            row_features=tp_row[FEATURE_COLS],
            tx_id=int(tp_row["TRANSACTION_ID"]),
            true_label=int(tp_row[TARGET_COL]),
            cohort_name="True Positive (Caught Fraud)",
            output_plot_path="outputs/plots/shap_waterfall_tp.png"
        )
        explanations.append(res_tp)

    # Case B: False Positive (Innocent Cardholder Flagged)
    fp_candidates = val_df[(val_df[TARGET_COL] == 0) & (val_df["PRED"] == 1)].sort_values(by="P_FRAUD", ascending=False)
    if len(fp_candidates) > 0:
        fp_row = fp_candidates.iloc[0]
        res_fp = explain_individual_transaction(
            explainer=explainer,
            model=model,
            row_features=fp_row[FEATURE_COLS],
            tx_id=int(fp_row["TRANSACTION_ID"]),
            true_label=int(fp_row[TARGET_COL]),
            cohort_name="False Positive (Innocent Flagged)",
            output_plot_path="outputs/plots/shap_waterfall_fp.png"
        )
        explanations.append(res_fp)

    # Case C: False Negative (Subtle Fraud Missed)
    fn_candidates = val_df[(val_df[TARGET_COL] == 1) & (val_df["PRED"] == 0)].sort_values(by="P_FRAUD", ascending=True)
    if len(fn_candidates) > 0:
        fn_row = fn_candidates.iloc[0]
        res_fn = explain_individual_transaction(
            explainer=explainer,
            model=model,
            row_features=fn_row[FEATURE_COLS],
            tx_id=int(fn_row["TRANSACTION_ID"]),
            true_label=int(fn_row[TARGET_COL]),
            cohort_name="False Negative (Fraud Missed)",
            output_plot_path="outputs/plots/shap_waterfall_fn.png"
        )
        explanations.append(res_fn)

    # Save to metrics JSON
    out_json = "outputs/metrics/shap_explanation_examples.json"
    with open(out_json, "w") as f:
        json.dump(explanations, f, indent=4)
    print(f"Saved verified SHAP explanations to: {out_json}")
    print("=" * 70)

    return {"explanations": explanations}


if __name__ == "__main__":
    run_explainability_pipeline()

