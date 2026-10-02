"""Model Evaluation, Experiment Tracking, and Comprehensive Comparison Module.

Computes imbalanced classification metrics:
- PR-AUC (Average Precision) -> Primary metric
- ROC-AUC
- Precision, Recall, F1 Score
- Confusion Matrix (TN, FP, FN, TP)
- Accuracy (included explicitly to demonstrate its flaw in imbalanced domains)
"""

import json
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
    experiment_id: str = "E0",
    model_name: str = "Model",
    imbalance_strategy: str = "None"
) -> dict:
    """Calculate comprehensive evaluation metrics at a specified decision threshold."""
    y_true = np.asarray(y_true).astype(int)
    y_pred_proba = np.asarray(y_pred_proba).astype(float)
    y_pred = (y_pred_proba >= threshold).astype(int)

    # 1. Ranking metrics (threshold-independent)
    pr_auc = float(average_precision_score(y_true, y_pred_proba))
    try:
        roc_auc = float(roc_auc_score(y_true, y_pred_proba))
    except ValueError:
        roc_auc = 0.5

    # 2. Threshold-dependent classification metrics
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    accuracy = float(np.mean(y_true == y_pred))

    # 3. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "Experiment": experiment_id,
        "Model": model_name,
        "Imbalance Strategy": imbalance_strategy,
        "Threshold": round(threshold, 2),
        "PR-AUC": round(pr_auc, 4),
        "Recall": round(recall, 4),
        "Precision": round(precision, 4),
        "F1": round(f1, 4),
        "ROC-AUC": round(roc_auc, 4),
        "Accuracy": round(accuracy, 4),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def print_metrics_summary(metrics: dict) -> None:
    """Pretty-print formatted metrics table and confusion matrix."""
    print("=" * 65)
    print(f" EXPERIMENT RESULTS: {metrics['Experiment']} | {metrics['Model']}")
    print(f" Imbalance Strategy: {metrics['Imbalance Strategy']} | Threshold: {metrics['Threshold']}")
    print("=" * 65)
    print(f"  PR-AUC (Primary Metric) : {metrics['PR-AUC']:.4f}")
    print(f"  Recall                  : {metrics['Recall']:.4f}")
    print(f"  Precision               : {metrics['Precision']:.4f}")
    print(f"  F1 Score                : {metrics['F1']:.4f}")
    print(f"  ROC-AUC                 : {metrics['ROC-AUC']:.4f}")
    print(f"  Accuracy                : {metrics['Accuracy']:.4f}  <-- Notice how misleading this is!")
    print("-" * 65)
    print("  Confusion Matrix:")
    print(f"    True Negatives  (Legitimate Correct) : {metrics['TN']:,}")
    print(f"    False Positives (Legitimate Flagged) : {metrics['FP']:,}")
    print(f"    False Negatives (Fraud Missed)       : {metrics['FN']:,}")
    print(f"    True Positives  (Fraud Caught)       : {metrics['TP']:,}")
    print("=" * 65)


def log_experiment_result(
    metrics: dict,
    results_path: str = "outputs/metrics/experiments.csv"
) -> pd.DataFrame:
    """Append or update experiment metrics in persistent experiments.csv log."""
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    row_df = pd.DataFrame([metrics])

    if os.path.exists(results_path):
        existing_df = pd.read_csv(results_path)
        if metrics["Experiment"] in existing_df["Experiment"].values:
            existing_df = existing_df[existing_df["Experiment"] != metrics["Experiment"]]
        updated_df = pd.concat([existing_df, row_df], ignore_index=True)
    else:
        updated_df = row_df

    # Sort by experiment ID
    updated_df = updated_df.sort_values(by="Experiment").reset_index(drop=True)
    updated_df.to_csv(results_path, index=False)
    return updated_df


def plot_curves(
    y_true: np.ndarray,
    models_dict: dict[str, np.ndarray],
    output_path: str = "outputs/plots/experiment_pr_roc_curves.png"
) -> None:
    """Plot Precision-Recall and ROC curves comparing multiple models."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    base_rate = float(np.mean(y_true))
    axes[0].axhline(y=base_rate, color="grey", linestyle="--", label=f"Random Baseline ({base_rate:.4f})")

    for model_name, probas in models_dict.items():
        prec, rec, _ = precision_recall_curve(y_true, probas)
        ap = average_precision_score(y_true, probas)
        axes[0].plot(rec, prec, label=f"{model_name} (PR-AUC={ap:.4f})")

        fpr, tpr, _ = roc_curve(y_true, probas)
        auc = roc_auc_score(y_true, probas)
        axes[1].plot(fpr, tpr, label=f"{model_name} (ROC-AUC={auc:.4f})")

    axes[0].set_xlabel("Recall", fontsize=11)
    axes[0].set_ylabel("Precision", fontsize=11)
    axes[0].set_title("Precision-Recall Curves (Primary Comparison)", fontsize=12, fontweight="bold")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot([0, 1], [0, 1], color="grey", linestyle="--", label="Random Guess (0.5000)")
    axes[1].set_xlabel("False Positive Rate", fontsize=11)
    axes[1].set_ylabel("True Positive Rate (Recall)", fontsize=11)
    axes[1].set_title("Receiver Operating Characteristic (ROC)", fontsize=12, fontweight="bold")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[Plot Saved] Curves comparison -> {output_path}")


def generate_experiment_comparison_report(
    results_path: str = "outputs/metrics/experiments.csv",
    plot_path: str = "outputs/plots/experiment_metrics_comparison.png"
) -> pd.DataFrame:
    """Generate comprehensive comparison table and multi-metric diagnostic bar chart."""
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Experiment log not found at {results_path}")

    df = pd.read_csv(results_path)
    df = df.sort_values(by="Experiment").reset_index(drop=True)

    print("\n" + "=" * 80)
    print("                PHASE 7: COMPREHENSIVE EXPERIMENT COMPARISON TABLE")
    print("=" * 80)
    display_cols = ["Experiment", "Model", "Imbalance Strategy", "PR-AUC", "Recall", "Precision", "F1", "ROC-AUC", "Accuracy"]
    print(df[display_cols].to_string(index=False))
    print("=" * 80)

    # Multi-metric comparison bar plot
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 6))

    metrics_to_plot = ["PR-AUC", "F1", "Recall", "Precision"]
    x = np.arange(len(df))
    width = 0.2

    colors = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77"]
    for i, metric in enumerate(metrics_to_plot):
        ax.bar(x + (i - 1.5) * width, df[metric], width, label=metric, color=colors[i], alpha=0.85, edgecolor="black")

    ax.set_xticks(x)
    exp_labels = [f"{row['Experiment']}\n{row['Model']}" for _, row in df.iterrows()]
    ax.set_xticklabels(exp_labels, fontsize=10)
    ax.set_ylabel("Score (0.0 to 1.0)", fontsize=11, fontweight="bold")
    ax.set_title("Cross-Experiment Performance Comparison (Validation Set)", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f"[Plot Saved] Experiment metrics comparison -> {plot_path}")

    return df


if __name__ == "__main__":
    generate_experiment_comparison_report()
