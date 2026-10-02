"""Systematic Decision Threshold Tuning & Final Test Evaluation Module.

Sweeps classification thresholds on the VALIDATION set to find the optimal operating
point balancing Precision, Recall, and F1. Only after locking the threshold on
Validation is the untouched TEST set evaluated once for unbiased final performance.
"""

import json
import os
import sys
# Ensure utf-8 encoding for Windows console
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
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import joblib

from src.train import load_data_splits
from src.preprocessing import FEATURE_COLS, TARGET_COL


def evaluate_threshold_grid(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    thresholds: list[float] | None = None
) -> pd.DataFrame:
    """Evaluate precision, recall, F1, FP, FN across a range of decision thresholds."""
    if thresholds is None:
        thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

    records = []
    for t in thresholds:
        preds = (y_pred_proba >= t).astype(int)
        p = float(precision_score(y_true, preds, zero_division=0))
        r = float(recall_score(y_true, preds, zero_division=0))
        f1 = float(f1_score(y_true, preds, zero_division=0))
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()

        records.append({
            "Threshold": round(t, 2),
            "Precision": round(p, 4),
            "Recall": round(r, 4),
            "F1": round(f1, 4),
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "TN": int(tn)
        })

    return pd.DataFrame(records)


def plot_threshold_curves(
    tuning_df: pd.DataFrame,
    optimal_threshold: float,
    model_name: str = "XGBoost",
    output_path: str = "outputs/plots/threshold_tuning_curve.png"
) -> None:
    """Plot Threshold vs Precision, Recall, and F1 curve with optimal operating point."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))

    ax.plot(tuning_df["Threshold"], tuning_df["Precision"], label="Precision", color="#1b9e77", linewidth=2.2, marker="o")
    ax.plot(tuning_df["Threshold"], tuning_df["Recall"], label="Recall", color="#d95f02", linewidth=2.2, marker="s")
    ax.plot(tuning_df["Threshold"], tuning_df["F1"], label="F1 Score", color="#2b5c8f", linewidth=2.5, marker="^")

    # Mark selected optimal threshold
    ax.axvline(x=optimal_threshold, color="#7570b3", linestyle="--", linewidth=2.0,
               label=f"Selected Threshold theta* = {optimal_threshold:.2f}")

    ax.set_xlabel("Decision Probability Threshold (theta)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Score (0.0 to 1.0)", fontsize=11, fontweight="bold")
    ax.set_title(f"Threshold vs Precision, Recall, & F1 Tuning ({model_name} on Validation Set)",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="center right", frameon=True)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(0.0, 0.95)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[Plot Saved] Threshold tuning curve -> {output_path}")


def run_threshold_tuning_and_test_eval() -> dict:
    """End-to-end threshold optimization and final unbiased test evaluation."""
    print("=" * 70)
    print("       PHASE 8: THRESHOLD TUNING & FINAL TEST EVALUATION")
    print("=" * 70)

    # 1. Load data splits
    data = load_data_splits()
    y_val = data["y_val"]
    y_test = data["y_test"]

    # 2. Select final candidate model: XGBoost Baseline (E3)
    model_path = "models/baseline/xgboost_baseline.joblib"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run src/train.py first.")
    model = joblib.load(model_path)

    # Predict validation probabilities
    val_probas = model.predict_proba(data["X_val_raw"])[:, 1]

    # 3. Sweep thresholds on VALIDATION data only
    threshold_grid = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    tuning_df = evaluate_threshold_grid(y_val, val_probas, threshold_grid)

    print("\n--- VALIDATION SET THRESHOLD TUNING GRID ---")
    print(tuning_df.to_string(index=False))

    # 4. Programmatically select threshold maximizing F1 on Validation
    best_row = tuning_df.loc[tuning_df["F1"].idxmax()]
    optimal_threshold = float(best_row["Threshold"])
    print(f"\n[OPTIMAL OPERATING POINT SELECTED ON VALIDATION SET]")
    print(f"  Selected Threshold (theta*) : {optimal_threshold:.2f}")
    print(f"  Validation F1 Score         : {best_row['F1']:.4f}")
    print(f"  Validation Precision        : {best_row['Precision']:.4f} (TP={best_row['TP']}, FP={best_row['FP']})")
    print(f"  Validation Recall           : {best_row['Recall']:.4f} (FN={best_row['FN']})")

    # Plot tuning curves
    plot_threshold_curves(tuning_df, optimal_threshold=optimal_threshold, model_name="XGBoost")

    # 5. NOW AND ONLY NOW: Evaluate once on untouched TEST SET
    print("\n" + "=" * 70)
    print("     FINAL UNBIASED TEST EVALUATION (TOUCHED ONCE WITH theta*)")
    print("=" * 70)

    test_probas = model.predict_proba(data["X_test_raw"])[:, 1]
    test_preds_default = (test_probas >= 0.50).astype(int)
    test_preds_tuned = (test_probas >= optimal_threshold).astype(int)

    # Compute ranking metrics on test set
    test_pr_auc = float(average_precision_score(y_test, test_probas))
    test_roc_auc = float(roc_auc_score(y_test, test_probas))

    # Metrics at default 0.50
    p_def = float(precision_score(y_test, test_preds_default, zero_division=0))
    r_def = float(recall_score(y_test, test_preds_default, zero_division=0))
    f1_def = float(f1_score(y_test, test_preds_default, zero_division=0))
    tn_def, fp_def, fn_def, tp_def = confusion_matrix(y_test, test_preds_default, labels=[0, 1]).ravel()

    # Metrics at tuned theta* = 0.20
    p_opt = float(precision_score(y_test, test_preds_tuned, zero_division=0))
    r_opt = float(recall_score(y_test, test_preds_tuned, zero_division=0))
    f1_opt = float(f1_score(y_test, test_preds_tuned, zero_division=0))
    tn_opt, fp_opt, fn_opt, tp_opt = confusion_matrix(y_test, test_preds_tuned, labels=[0, 1]).ravel()

    print(f"Test Set PR-AUC (Primary Metric) : {test_pr_auc:.4f}")
    print(f"Test Set ROC-AUC                 : {test_roc_auc:.4f}")
    print("-" * 70)
    print(f"Comparison at Default (0.50) vs. Tuned ({optimal_threshold:.2f}):")
    print(f"  Default Threshold (theta = 0.50)   : Recall={r_def:.4f} | Precision={p_def:.4f} | F1={f1_def:.4f} | TP={tp_def} | FP={fp_def}")
    print(f"  Tuned Threshold   (theta* = {optimal_threshold:.2f})  : Recall={r_opt:.4f} | Precision={p_opt:.4f} | F1={f1_opt:.4f} | TP={tp_opt} | FP={fp_opt}")
    print("-" * 70)
    print(f"Impact: Threshold tuning increased Recall from {r_def*100:.1f}% to {r_opt*100:.1f}% (+{tp_opt - tp_def} frauds caught) on unseen future test transactions!")
    print("=" * 70)

    # Save artifacts
    os.makedirs("models/final", exist_ok=True)
    joblib.dump(model, "models/final/final_supervised_model.joblib")
    np.save("outputs/predictions/test_probas_final.npy", test_probas)

    results_dict = {
        "model": "XGBoost",
        "optimal_threshold": optimal_threshold,
        "test_metrics": {
            "PR-AUC": round(test_pr_auc, 4),
            "ROC-AUC": round(test_roc_auc, 4),
            "default_threshold_0.50": {
                "precision": round(p_def, 4),
                "recall": round(r_def, 4),
                "f1": round(f1_def, 4),
                "TP": int(tp_def), "FP": int(fp_def), "FN": int(fn_def), "TN": int(tn_def)
            },
            "tuned_threshold": {
                "threshold": optimal_threshold,
                "precision": round(p_opt, 4),
                "recall": round(r_opt, 4),
                "f1": round(f1_opt, 4),
                "TP": int(tp_opt), "FP": int(fp_opt), "FN": int(fn_opt), "TN": int(tn_opt)
            }
        }
    }

    with open("outputs/metrics/final_test_metrics.json", "w") as f:
        json.dump(results_dict, f, indent=4)
    print("Saved final test evaluation metrics to: outputs/metrics/final_test_metrics.json")
    print("Saved final model artifact to: models/final/final_supervised_model.joblib")

    return results_dict


if __name__ == "__main__":
    run_threshold_tuning_and_test_eval()
