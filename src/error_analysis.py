"""Deep Error Analysis Module for Fraud Detection System.

Performs statistical and behavioral investigation of:
1. False Positives (Innocent cardholders flagged -> customer friction)
2. False Negatives (Fraud transactions missed -> financial loss)
3. Cohort profiling across amount, velocity, ratio, and timing
4. Ground-truth scenario vulnerability analysis (Scenario 1 vs 2 vs 3)
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def perform_error_analysis(
    data_path: str = "data/processed/val.csv",
    probas_path: str = "outputs/predictions/val_probas_E3.npy",
    optimal_threshold: float = 0.20,
    output_dir: str = "outputs/plots",
    metrics_dir: str = "outputs/metrics"
) -> dict:
    """Analyze error cohorts, investigate behavioral root causes, and export plots."""
    print("=" * 70)
    print("         PHASE 10: DEEP ERROR ANALYSIS & COHORT PROFILING")
    print("=" * 70)

    val_df = pd.read_csv(data_path)
    probas = np.load(probas_path)
    val_df["P_FRAUD"] = probas
    val_df["PRED"] = (probas >= optimal_threshold).astype(int)

    # Classify transactions into 4 distinct cohorts
    val_df["COHORT"] = "TN"
    val_df.loc[(val_df["TX_FRAUD"] == 0) & (val_df["PRED"] == 1), "COHORT"] = "FP"
    val_df.loc[(val_df["TX_FRAUD"] == 1) & (val_df["PRED"] == 0), "COHORT"] = "FN"
    val_df.loc[(val_df["TX_FRAUD"] == 1) & (val_df["PRED"] == 1), "COHORT"] = "TP"

    cohort_order = ["TN", "FP", "FN", "TP"]
    cohort_counts = val_df["COHORT"].value_counts().reindex(cohort_order)

    print("Prediction Cohort Sizes:")
    for cohort, count in cohort_counts.items():
        pct = count / len(val_df) * 100
        desc = {
            "TN": "True Negatives (Legitimate Approved)",
            "FP": "False Positives (Innocent Flagged)",
            "FN": "False Negatives (Fraud Missed)",
            "TP": "True Positives (Fraud Caught)",
        }[cohort]
        print(f"  {cohort} ({desc:<35}): {count:>6,} ({pct:>5.2f}%)")
    print("-" * 70)

    # 1. Behavioral Profiles Across Cohorts
    features_to_analyze = [
        "TX_AMOUNT",
        "CUSTOMER_AMOUNT_RATIO",
        "CUSTOMER_TX_COUNT_1H",
        "CUSTOMER_TX_COUNT_24H",
        "TERMINAL_TX_COUNT_1D",
        "TX_IS_NIGHT"
    ]
    cohort_stats = val_df.groupby("COHORT")[features_to_analyze].agg(["mean", "median", "std"])

    print("\nBehavioral Attribute Comparison (Mean Values):")
    print(f"{'Cohort':<6} | {'TX_AMOUNT':<10} | {'AMOUNT_RATIO':<12} | {'TX_1H':<6} | {'TX_24H':<7} | {'TERM_1D':<8} | {'IS_NIGHT'}")
    print("-" * 70)
    for c in cohort_order:
        row = val_df[val_df["COHORT"] == c]
        print(
            f"{c:<6} | "
            f"${row['TX_AMOUNT'].mean():<9.2f} | "
            f"{row['CUSTOMER_AMOUNT_RATIO'].mean():<12.2f} | "
            f"{row['CUSTOMER_TX_COUNT_1H'].mean():<6.2f} | "
            f"{row['CUSTOMER_TX_COUNT_24H'].mean():<7.2f} | "
            f"{row['TERMINAL_TX_COUNT_1D'].mean():<8.2f} | "
            f"{row['TX_IS_NIGHT'].mean() * 100:.1f}%"
        )
    print("-" * 70)

    # 2. Fraud Scenario Vulnerability Breakdown
    print("\nGround-Truth Fraud Scenarios: Caught (TP) vs. Missed (FN):")
    scenarios = {
        1: "Scenario 1: High Amount (> 220)",
        2: "Scenario 2: Compromised Terminal Window",
        3: "Scenario 3: Account Takeover (5x Amount)"
    }
    scenario_counts = val_df[val_df["TX_FRAUD"] == 1].groupby(["TX_FRAUD_SCENARIO", "COHORT"]).size().unstack(fill_value=0)
    if "TP" not in scenario_counts.columns:
        scenario_counts["TP"] = 0
    if "FN" not in scenario_counts.columns:
        scenario_counts["FN"] = 0
    scenario_counts["Total"] = scenario_counts["TP"] + scenario_counts["FN"]
    scenario_counts["Recall_%"] = (scenario_counts["TP"] / scenario_counts["Total"] * 100).round(1)

    print(f"{'Scenario':<42} | {'Total':<6} | {'TP (Caught)':<12} | {'FN (Missed)':<12} | {'Detection Recall'}")
    print("-" * 85)
    for sc_id, row in scenario_counts.iterrows():
        sc_name = scenarios.get(sc_id, f"Scenario {sc_id}")
        print(f"[{sc_id}] {sc_name:<38} | {int(row['Total']):<6} | {int(row['TP']):<12} | {int(row['FN']):<12} | {row['Recall_%']}%")
    print("-" * 85)

    # 3. Export Visualizations
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")

    # Plot 1: TX_AMOUNT Distribution by Cohort
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    palette = {"TN": "#2b5c8f", "FP": "#e7298a", "FN": "#d95f02", "TP": "#1b9e77"}

    sns.boxplot(data=val_df, x="COHORT", y="TX_AMOUNT", order=cohort_order, palette=palette, ax=axes[0], showfliers=False)
    axes[0].set_title("Transaction Amount Distribution Across Cohorts (Excl. Outliers)", fontweight="bold")
    axes[0].set_ylabel("TX_AMOUNT ($)")

    # Plot 2: CUSTOMER_AMOUNT_RATIO by Cohort
    sns.boxplot(data=val_df, x="COHORT", y="CUSTOMER_AMOUNT_RATIO", order=cohort_order, palette=palette, ax=axes[1], showfliers=False)
    axes[1].set_title("Customer Amount Ratio (Current vs. 7-Day History)", fontweight="bold")
    axes[1].set_ylabel("CUSTOMER_AMOUNT_RATIO")

    plt.tight_layout()
    p1 = os.path.join(output_dir, "error_analysis_amounts_and_ratios.png")
    fig.savefig(p1, dpi=200)
    plt.close(fig)
    print(f"[Plot Saved] Amount & ratio cohort boxplots -> {p1}")

    # Plot 3: Scenario Detection Rates
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sc_labels = [scenarios.get(idx, f"Scenario {idx}") for idx in scenario_counts.index]
    bars = ax.bar(sc_labels, scenario_counts["Recall_%"], color=["#1b9e77", "#d95f02", "#2b5c8f"], edgecolor="black")
    ax.set_ylabel("Detection Recall Rate (%)", fontweight="bold")
    ax.set_title("Model Detection Rate by Fraud Attack Scenario", fontweight="bold")
    ax.set_ylim(0, 110)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 2, f"{h:.1f}%", ha="center", fontweight="bold")
    plt.tight_layout()
    p2 = os.path.join(output_dir, "error_analysis_scenarios_caught.png")
    fig.savefig(p2, dpi=200)
    plt.close(fig)
    print(f"[Plot Saved] Scenario detection rate -> {p2}")

    # 4. Save JSON summary
    summary_data = {
        "cohort_counts": {k: int(v) for k, v in cohort_counts.items()},
        "cohort_means": {
            c: {col: round(float(val_df[val_df["COHORT"] == c][col].mean()), 2) for col in features_to_analyze}
            for c in cohort_order
        },
        "scenario_detection": {
            int(idx): {
                "name": scenarios.get(idx, "Unknown"),
                "total": int(row["Total"]),
                "caught": int(row["TP"]),
                "missed": int(row["FN"]),
                "recall_pct": float(row["Recall_%"])
            }
            for idx, row in scenario_counts.iterrows()
        }
    }
    json_path = os.path.join(metrics_dir, "error_analysis_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary_data, f, indent=4)
    print(f"Saved detailed error report to: {json_path}")
    print("=" * 70)

    return summary_data


if __name__ == "__main__":
    perform_error_analysis()

