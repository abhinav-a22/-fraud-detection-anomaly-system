"""Unsupervised Anomaly Detection Module using Isolation Forest.

Implements:
- E6: Baseline Isolation Forest (all features, trained strictly without labels)
- E8: Feature-Selected Isolation Forest (focused behavioral & deviation features)
- Anomaly score normalization [0, 1]
- 4-Way Agreement Analysis: Supervised (XGBoost) vs. Unsupervised (Isolation Forest)
- Diagnostic quadrant visualization
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
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.evaluate import log_experiment_result, print_metrics_summary
from src.preprocessing import FEATURE_COLS, TARGET_COL
from src.train import load_data_splits


SELECTED_ANOMALY_FEATURES = [
    "TX_AMOUNT",
    "TX_AMOUNT_LOG",
    "CUSTOMER_AMOUNT_RATIO",
    "CUSTOMER_TX_COUNT_1H",
    "CUSTOMER_TX_COUNT_24H",
    "TERMINAL_TX_COUNT_1D",
    "TX_IS_NIGHT",
]


def fit_isolation_forest(
    X_train: np.ndarray,
    contamination: float = 0.015,
    random_state: int = 42
) -> IsolationForest:
    """Train Isolation Forest on unlabeled training data."""
    iso = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        max_samples=256,
        random_state=random_state,
        n_jobs=-1
    )
    # UNLABELED: y_train is never passed
    iso.fit(X_train)
    return iso


def compute_normalized_anomaly_scores(
    model: IsolationForest,
    X: np.ndarray
) -> np.ndarray:
    """Convert raw score_samples into normalized [0, 1] anomaly scores (1 = high anomaly)."""
    # score_samples returns negative values (lower = more anomalous)
    raw_scores = -model.score_samples(X)
    # Min-max normalization for interpretable risk scoring
    s_min = raw_scores.min()
    s_max = raw_scores.max()
    normalized = (raw_scores - s_min) / (s_max - s_min + 1e-8)
    return normalized.astype(np.float32)


def evaluate_anomaly_detector(
    y_true: np.ndarray,
    anomaly_scores: np.ndarray,
    contamination: float = 0.015,
    experiment_id: str = "E6",
    model_name: str = "Isolation Forest",
    feature_desc: str = "All Features"
) -> dict:
    """Evaluate anomaly detector against ground-truth labels (POST-HOC ONLY)."""
    # Threshold at top percentile corresponding to contamination
    cutoff = np.percentile(anomaly_scores, 100 * (1 - contamination))
    preds = (anomaly_scores >= cutoff).astype(int)

    pr_auc = float(average_precision_score(y_true, anomaly_scores))
    roc_auc = float(roc_auc_score(y_true, anomaly_scores))
    p = float(precision_score(y_true, preds, zero_division=0))
    r = float(recall_score(y_true, preds, zero_division=0))
    f1 = float(f1_score(y_true, preds, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    acc = float(np.mean(y_true == preds))

    metrics = {
        "Experiment": experiment_id,
        "Model": model_name,
        "Imbalance Strategy": f"Unsupervised ({feature_desc})",
        "Threshold": round(float(cutoff), 4),
        "PR-AUC": round(pr_auc, 4),
        "Recall": round(r, 4),
        "Precision": round(p, 4),
        "F1": round(f1, 4),
        "ROC-AUC": round(roc_auc, 4),
        "Accuracy": round(acc, 4),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }

    print_metrics_summary(metrics)
    log_experiment_result(metrics)
    return metrics


def analyze_agreement(
    y_true: np.ndarray,
    sup_probas: np.ndarray,
    ano_scores: np.ndarray,
    sup_threshold: float = 0.20,
    ano_percentile: float = 98.5,
    output_plot: str = "outputs/plots/supervised_vs_unsupervised_quadrants.png"
) -> dict:
    """Compare predictions between Supervised (XGBoost) and Unsupervised (Isolation Forest).

    Analyzes 4 distinct quadrants:
    1. Both Agree Fraud: Known pattern with high behavioral deviation.
    2. XGBoost Flagged, IsoForest Missed: Subtler fraud matching supervised training conjunctions.
    3. IsoForest Flagged, XGBoost Missed: Outlier behavior or novel anomaly pattern.
    4. Both Agree Legitimate: Routine low-risk traffic.
    """
    ano_threshold = np.percentile(ano_scores, ano_percentile)

    sup_flag = (sup_probas >= sup_threshold).astype(int)
    ano_flag = (ano_scores >= ano_threshold).astype(int)

    both_fraud = int(np.sum((sup_flag == 1) & (ano_flag == 1)))
    xgb_only = int(np.sum((sup_flag == 1) & (ano_flag == 0)))
    iso_only = int(np.sum((sup_flag == 0) & (ano_flag == 1)))
    both_legit = int(np.sum((sup_flag == 0) & (ano_flag == 0)))

    # Count actual frauds in each quadrant
    frauds_both = int(np.sum((sup_flag == 1) & (ano_flag == 1) & (y_true == 1)))
    frauds_xgb_only = int(np.sum((sup_flag == 1) & (ano_flag == 0) & (y_true == 1)))
    frauds_iso_only = int(np.sum((sup_flag == 0) & (ano_flag == 1) & (y_true == 1)))
    frauds_both_legit = int(np.sum((sup_flag == 0) & (ano_flag == 0) & (y_true == 1)))

    print("\n" + "=" * 70)
    print("      4-WAY AGREEMENT ANALYSIS: SUPERVISED (XGB) vs. ANOMALY (ISO)")
    print("=" * 70)
    print(f"Quadrant 1: Both Agree Fraud       : {both_fraud:>6} alerts  | Actual Frauds: {frauds_both:>3} ({frauds_both/max(1, both_fraud)*100:.1f}% hit rate)")
    print(f"Quadrant 2: XGBoost Only (Iso Miss): {xgb_only:>6} alerts  | Actual Frauds: {frauds_xgb_only:>3} ({frauds_xgb_only/max(1, xgb_only)*100:.1f}% hit rate)")
    print(f"Quadrant 3: IsoForest Only (Novel) : {iso_only:>6} alerts  | Actual Frauds: {frauds_iso_only:>3} ({frauds_iso_only/max(1, iso_only)*100:.1f}% hit rate)")
    print(f"Quadrant 4: Both Agree Legitimate  : {both_legit:>6} trans.  | Undetected Frauds: {frauds_both_legit:>3}")
    print("=" * 70)

    # Diagnostic Quadrant Plot
    os.makedirs(os.path.dirname(output_plot), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6.5))

    # Sample for visual clarity
    indices = np.random.choice(len(y_true), size=min(10000, len(y_true)), replace=False)
    sub_y = y_true[indices]
    sub_sup = sup_probas[indices]
    sub_ano = ano_scores[indices]

    scatter = ax.scatter(
        sub_sup[sub_y == 0], sub_ano[sub_y == 0],
        c="#2b5c8f", alpha=0.3, s=15, label="Legitimate (0)"
    )
    scatter_fraud = ax.scatter(
        sub_sup[sub_y == 1], sub_ano[sub_y == 1],
        c="#d95f02", alpha=0.9, s=40, edgecolors="black", label="Fraud (1)"
    )

    ax.axvline(x=sup_threshold, color="black", linestyle="--", linewidth=1.5, label=f"XGB Threshold ({sup_threshold})")
    ax.axhline(y=ano_threshold, color="purple", linestyle="--", linewidth=1.5, label=f"Anomaly Threshold ({ano_threshold:.3f})")

    ax.set_xlabel("Supervised Fraud Probability (XGBoost)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Normalized Anomaly Score (Isolation Forest)", fontsize=11, fontweight="bold")
    ax.set_title("Supervised Risk vs. Anomaly Detection Quadrants", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_plot, dpi=200)
    plt.close(fig)
    print(f"[Plot Saved] Quadrant agreement chart -> {output_plot}")

    return {
        "both_fraud": both_fraud,
        "xgb_only": xgb_only,
        "iso_only": iso_only,
        "both_legit": both_legit,
        "frauds_both": frauds_both,
        "frauds_xgb_only": frauds_xgb_only,
        "frauds_iso_only": frauds_iso_only
    }


def run_anomaly_detection_pipeline() -> dict:
    """Run E6, E8, and 4-way agreement analysis."""
    print("=" * 70)
    print("      PHASE 9: UNSUPERVISED ANOMALY DETECTION (ISOLATION FOREST)    ")
    print("=" * 70)

    data = load_data_splits()
    X_train_raw = data["X_train_raw"]
    X_val_raw = data["X_val_raw"]
    y_val = data["y_val"]

    # ---------------- EXPERIMENT E6: Baseline Isolation Forest (All Features) ----------------
    print("\n--- Training Experiment E6: Isolation Forest (All 14 Features) ---")
    iso_e6 = fit_isolation_forest(X_train_raw, contamination=0.015, random_state=42)
    scores_e6_val = compute_normalized_anomaly_scores(iso_e6, X_val_raw)

    metrics_e6 = evaluate_anomaly_detector(
        y_true=y_val,
        anomaly_scores=scores_e6_val,
        contamination=0.015,
        experiment_id="E6",
        model_name="Isolation Forest",
        feature_desc="All Features"
    )
    joblib.dump(iso_e6, "models/baseline/isolation_forest_e6.joblib")
    np.save("outputs/predictions/val_anomaly_scores_E6.npy", scores_e6_val)

    # ---------------- EXPERIMENT E8: Feature-Selected Isolation Forest ----------------
    print("\n--- Training Experiment E8: Isolation Forest (Selected Behavioral Features) ---")
    feat_indices = [FEATURE_COLS.index(col) for col in SELECTED_ANOMALY_FEATURES]
    X_train_sel = X_train_raw[:, feat_indices]
    X_val_sel = X_val_raw[:, feat_indices]

    iso_e8 = fit_isolation_forest(X_train_sel, contamination=0.015, random_state=42)
    scores_e8_val = compute_normalized_anomaly_scores(iso_e8, X_val_sel)

    metrics_e8 = evaluate_anomaly_detector(
        y_true=y_val,
        anomaly_scores=scores_e8_val,
        contamination=0.015,
        experiment_id="E8",
        model_name="Isolation Forest",
        feature_desc="Behavioral Subset"
    )
    os.makedirs("models/final", exist_ok=True)
    joblib.dump(iso_e8, "models/final/final_isolation_forest.joblib")
    np.save("outputs/predictions/val_anomaly_scores_E8.npy", scores_e8_val)

    # ---------------- 4-Way Agreement Analysis ----------------
    sup_probas_val = np.load("outputs/predictions/val_probas_E3.npy")
    agreement = analyze_agreement(
        y_true=y_val,
        sup_probas=sup_probas_val,
        ano_scores=scores_e8_val,
        sup_threshold=0.20,
        ano_percentile=98.5
    )

    return {"metrics_e6": metrics_e6, "metrics_e8": metrics_e8, "agreement": agreement}


if __name__ == "__main__":
    run_anomaly_detection_pipeline()

