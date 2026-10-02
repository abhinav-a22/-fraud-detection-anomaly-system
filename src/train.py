"""Model Training & Experiment Execution Module.

Implements the experimental matrix:
- E0: Dummy Baseline (Predict most frequent class)
- E1: Logistic Regression (Interpretable linear baseline)
- E2: Random Forest (Nonlinear tree ensemble baseline)
- E3: XGBoost (Gradient boosted decision trees baseline)
- E4: XGBoost + Class Weighting (Cost-sensitive scale_pos_weight)
- E5: XGBoost + SMOTE (Synthetic minority over-sampling on Train only)
"""

import argparse
import json
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from imblearn.over_sampling import SMOTE
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.evaluate import (
    compute_metrics,
    log_experiment_result,
    plot_curves,
    print_metrics_summary,
)
from src.preprocessing import FEATURE_COLS, TARGET_COL


def load_data_splits(
    data_dir: str = "data/processed",
    scaler_path: str = "models/baseline/scaler.joblib"
) -> dict:
    """Load train, val, and test data along with fitted scaler."""
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")
    test_path = os.path.join(data_dir, "test.csv")

    if not all(os.path.exists(p) for p in [train_path, val_path, test_path]):
        raise FileNotFoundError(f"Processed splits not found in {data_dir}. Run src/preprocessing.py first.")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

    X_train_raw = train_df[FEATURE_COLS].to_numpy()
    y_train = train_df[TARGET_COL].to_numpy()

    X_val_raw = val_df[FEATURE_COLS].to_numpy()
    y_val = val_df[TARGET_COL].to_numpy()

    X_test_raw = test_df[FEATURE_COLS].to_numpy()
    y_test = test_df[TARGET_COL].to_numpy()

    # Pre-scale for linear models
    X_train_scaled = scaler.transform(train_df[FEATURE_COLS]) if scaler else X_train_raw
    X_val_scaled = scaler.transform(val_df[FEATURE_COLS]) if scaler else X_val_raw
    X_test_scaled = scaler.transform(test_df[FEATURE_COLS]) if scaler else X_test_raw

    return {
        "train_df": train_df, "val_df": val_df, "test_df": test_df,
        "X_train_raw": X_train_raw, "X_train_scaled": X_train_scaled, "y_train": y_train,
        "X_val_raw": X_val_raw, "X_val_scaled": X_val_scaled, "y_val": y_val,
        "X_test_raw": X_test_raw, "X_test_scaled": X_test_scaled, "y_test": y_test,
        "scaler": scaler
    }


def train_e0_dummy(data: dict) -> dict:
    """E0: Dummy baseline predicting majority class (legitimate)."""
    print("\n--- Training Experiment E0: Dummy Baseline ---")
    model = DummyClassifier(strategy="most_frequent")
    model.fit(data["X_train_raw"], data["y_train"])

    val_probas = model.predict_proba(data["X_val_raw"])[:, 1]

    metrics = compute_metrics(
        y_true=data["y_val"],
        y_pred_proba=val_probas,
        threshold=0.5,
        experiment_id="E0",
        model_name="Dummy Baseline",
        imbalance_strategy="None"
    )

    print_metrics_summary(metrics)
    log_experiment_result(metrics)

    os.makedirs("outputs/predictions", exist_ok=True)
    np.save("outputs/predictions/val_probas_E0.npy", val_probas)

    return {"model": model, "metrics": metrics, "val_probas": val_probas}


def train_e1_logistic_regression(data: dict) -> dict:
    """E1: Logistic Regression using standardized features."""
    print("\n--- Training Experiment E1: Logistic Regression ---")
    model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42,
        solver="lbfgs"
    )
    model.fit(data["X_train_scaled"], data["y_train"])

    val_probas = model.predict_proba(data["X_val_scaled"])[:, 1]

    metrics = compute_metrics(
        y_true=data["y_val"],
        y_pred_proba=val_probas,
        threshold=0.5,
        experiment_id="E1",
        model_name="Logistic Regression",
        imbalance_strategy="None"
    )

    print_metrics_summary(metrics)
    log_experiment_result(metrics)

    os.makedirs("models/baseline", exist_ok=True)
    joblib.dump(model, "models/baseline/logistic_regression.joblib")

    os.makedirs("outputs/predictions", exist_ok=True)
    np.save("outputs/predictions/val_probas_E1.npy", val_probas)

    coef_df = pd.DataFrame({
        "Feature": FEATURE_COLS,
        "Coefficient": model.coef_[0]
    }).sort_values(by="Coefficient", ascending=False)

    return {"model": model, "metrics": metrics, "val_probas": val_probas, "coefficients": coef_df}


def train_e2_random_forest(data: dict) -> dict:
    """E2: Random Forest baseline (bagging ensemble of deep decision trees)."""
    print("\n--- Training Experiment E2: Random Forest ---")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )
    model.fit(data["X_train_raw"], data["y_train"])

    val_probas = model.predict_proba(data["X_val_raw"])[:, 1]

    metrics = compute_metrics(
        y_true=data["y_val"],
        y_pred_proba=val_probas,
        threshold=0.5,
        experiment_id="E2",
        model_name="Random Forest",
        imbalance_strategy="None"
    )

    print_metrics_summary(metrics)
    log_experiment_result(metrics)

    os.makedirs("models/baseline", exist_ok=True)
    joblib.dump(model, "models/baseline/random_forest.joblib")

    os.makedirs("outputs/predictions", exist_ok=True)
    np.save("outputs/predictions/val_probas_E2.npy", val_probas)

    importances_df = pd.DataFrame({
        "Feature": FEATURE_COLS,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    return {"model": model, "metrics": metrics, "val_probas": val_probas, "importances": importances_df}


def train_e3_xgboost(data: dict) -> dict:
    """E3: XGBoost baseline (gradient boosted decision trees, unweighted)."""
    print("\n--- Training Experiment E3: XGBoost ---")
    model = XGBClassifier(
        n_estimators=120,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
    model.fit(data["X_train_raw"], data["y_train"])

    val_probas = model.predict_proba(data["X_val_raw"])[:, 1]

    metrics = compute_metrics(
        y_true=data["y_val"],
        y_pred_proba=val_probas,
        threshold=0.5,
        experiment_id="E3",
        model_name="XGBoost",
        imbalance_strategy="None"
    )

    print_metrics_summary(metrics)
    log_experiment_result(metrics)

    os.makedirs("models/baseline", exist_ok=True)
    joblib.dump(model, "models/baseline/xgboost_baseline.joblib")

    os.makedirs("outputs/predictions", exist_ok=True)
    np.save("outputs/predictions/val_probas_E3.npy", val_probas)

    return {"model": model, "metrics": metrics, "val_probas": val_probas}


def train_e4_xgboost_class_weight(data: dict) -> dict:
    """E4: XGBoost with Cost-Sensitive Class Weighting (scale_pos_weight)."""
    print("\n--- Training Experiment E4: XGBoost + Class Weighting ---")
    n_neg = int(np.sum(data["y_train"] == 0))
    n_pos = int(np.sum(data["y_train"] == 1))
    scale_weight = n_neg / n_pos
    print(f"Calculated scale_pos_weight: {scale_weight:.2f} ({n_neg:,} neg / {n_pos:,} pos)")

    model = XGBClassifier(
        n_estimators=120,
        max_depth=5,
        learning_rate=0.08,
        scale_pos_weight=scale_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
    model.fit(data["X_train_raw"], data["y_train"])

    val_probas = model.predict_proba(data["X_val_raw"])[:, 1]

    metrics = compute_metrics(
        y_true=data["y_val"],
        y_pred_proba=val_probas,
        threshold=0.5,
        experiment_id="E4",
        model_name="XGBoost",
        imbalance_strategy=f"Class Weighting (scale_pos_weight={scale_weight:.1f})"
    )

    print_metrics_summary(metrics)
    log_experiment_result(metrics)

    os.makedirs("models/baseline", exist_ok=True)
    joblib.dump(model, "models/baseline/xgboost_class_weight.joblib")

    os.makedirs("outputs/predictions", exist_ok=True)
    np.save("outputs/predictions/val_probas_E4.npy", val_probas)

    return {"model": model, "metrics": metrics, "val_probas": val_probas}


def train_e5_xgboost_smote(data: dict) -> dict:
    """E5: XGBoost with SMOTE (Synthetic Minority Over-sampling on TRAIN ONLY)."""
    print("\n--- Training Experiment E5: XGBoost + SMOTE ---")
    print("Applying SMOTE strictly to training data (k_neighbors=5, sampling_strategy=0.10)...")
    smote = SMOTE(sampling_strategy=0.10, k_neighbors=5, random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(data["X_train_raw"], data["y_train"])

    n_res_pos = int(np.sum(y_train_resampled == 1))
    print(f"Train size after SMOTE: {len(X_train_resampled):,} | Frauds: {n_res_pos:,} ({n_res_pos / len(X_train_resampled) * 100:.2f}%)")
    print("CRITICAL: Validation and test sets remain completely untouched!")

    model = XGBClassifier(
        n_estimators=120,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
    # Fit on resampled training data
    model.fit(X_train_resampled, y_train_resampled)

    # Evaluate on UNTOUCHED raw validation data
    val_probas = model.predict_proba(data["X_val_raw"])[:, 1]

    metrics = compute_metrics(
        y_true=data["y_val"],
        y_pred_proba=val_probas,
        threshold=0.5,
        experiment_id="E5",
        model_name="XGBoost",
        imbalance_strategy="SMOTE (10% on Train)"
    )

    print_metrics_summary(metrics)
    log_experiment_result(metrics)

    os.makedirs("models/baseline", exist_ok=True)
    joblib.dump(model, "models/baseline/xgboost_smote.joblib")

    os.makedirs("outputs/predictions", exist_ok=True)
    np.save("outputs/predictions/val_probas_E5.npy", val_probas)

    return {"model": model, "metrics": metrics, "val_probas": val_probas}


def main():
    parser = argparse.ArgumentParser(description="Execute Fraud Detection Experiments")
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["E0", "E1", "E2", "E3", "E4", "E5", "all_phase4", "all_phase5", "all_phase6"],
        default="all_phase6",
        help="Experiment identifier to run"
    )
    args = parser.parse_args()

    print("Loading preprocessed chronological splits...")
    data = load_data_splits()

    if args.experiment in ["E0", "all_phase4", "all_phase5", "all_phase6"]:
        res_e0 = train_e0_dummy(data)

    if args.experiment in ["E1", "all_phase4", "all_phase5", "all_phase6"]:
        res_e1 = train_e1_logistic_regression(data)

    if args.experiment in ["E2", "all_phase5", "all_phase6"]:
        res_e2 = train_e2_random_forest(data)

    if args.experiment in ["E3", "all_phase5", "all_phase6"]:
        res_e3 = train_e3_xgboost(data)

    if args.experiment in ["E4", "all_phase6"]:
        res_e4 = train_e4_xgboost_class_weight(data)

    if args.experiment in ["E5", "all_phase6"]:
        res_e5 = train_e5_xgboost_smote(data)

    if args.experiment == "all_phase6":
        # Load validation predictions from disk to plot all 6 curves
        probas_dict = {
            "E0 Dummy Baseline": np.load("outputs/predictions/val_probas_E0.npy"),
            "E1 Logistic Regression": np.load("outputs/predictions/val_probas_E1.npy"),
            "E2 Random Forest": np.load("outputs/predictions/val_probas_E2.npy"),
            "E3 XGBoost Baseline": np.load("outputs/predictions/val_probas_E3.npy"),
            "E4 XGBoost + Class Weight": np.load("outputs/predictions/val_probas_E4.npy"),
            "E5 XGBoost + SMOTE": np.load("outputs/predictions/val_probas_E5.npy"),
        }
        plot_curves(
            y_true=data["y_val"],
            models_dict=probas_dict,
            output_path="outputs/plots/curves_phase6.png"
        )


if __name__ == "__main__":
    main()

