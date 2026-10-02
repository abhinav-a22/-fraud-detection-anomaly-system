"""Data Preprocessing & Chronological Splitting Module.

Performs:
1. Feature engineering orchestration (leakage-safe).
2. Strict chronological splitting (60% Train, 20% Validation, 20% Test).
3. Train-only imputation: calculates median baselines on Train and applies to Val/Test.
4. Train-only standardization: fits StandardScaler on Train and transforms Val/Test.
5. Saves processed splits to data/processed/ and metadata to outputs/metrics/.
"""

import json
import os
import sys
# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.features import engineer_all_features

FEATURE_COLS = [
    # Transactional features
    "TX_AMOUNT",
    "TX_AMOUNT_LOG",
    # Temporal features
    "TX_HOUR",
    "TX_DAY_OF_WEEK",
    "TX_IS_WEEKEND",
    "TX_IS_NIGHT",
    # Customer behavioral features
    "CUSTOMER_TX_COUNT_1H",
    "CUSTOMER_TX_COUNT_24H",
    "CUSTOMER_TX_COUNT_7D",
    "CUSTOMER_AVG_AMOUNT_1D",
    "CUSTOMER_AVG_AMOUNT_7D",
    "CUSTOMER_AMOUNT_RATIO",
    # Terminal behavioral features
    "TERMINAL_TX_COUNT_1D",
    "TERMINAL_AVG_AMOUNT_1D",
]

TARGET_COL = "TX_FRAUD"

METADATA_COLS = [
    "TRANSACTION_ID",
    "TX_DATETIME",
    "CUSTOMER_ID",
    "TERMINAL_ID",
    "TX_FRAUD_SCENARIO"
]


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.60,
    val_ratio: float = 0.20
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset chronologically into Train (60%), Validation (20%), and Test (20%)."""
    df = df.sort_values("TX_DATETIME").reset_index(drop=True)
    n_total = len(df)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)

    train_df = df.iloc[:n_train].copy().reset_index(drop=True)
    val_df = df.iloc[n_train:n_train + n_val].copy().reset_index(drop=True)
    test_df = df.iloc[n_train + n_val:].copy().reset_index(drop=True)

    return train_df, val_df, test_df


def handle_imputation(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Impute missing rolling values using TRAIN-ONLY statistics to prevent leakage."""
    print("Computing train-only imputation statistics...")
    imputation_values = {
        "CUSTOMER_AVG_AMOUNT_1D": float(train_df["CUSTOMER_AVG_AMOUNT_1D"].dropna().median()),
        "CUSTOMER_AVG_AMOUNT_7D": float(train_df["CUSTOMER_AVG_AMOUNT_7D"].dropna().median()),
        "TERMINAL_AVG_AMOUNT_1D": float(train_df["TERMINAL_AVG_AMOUNT_1D"].dropna().median()),
    }

    for col, fill_val in imputation_values.items():
        train_df[col] = train_df[col].fillna(fill_val)
        val_df[col] = val_df[col].fillna(fill_val)
        test_df[col] = test_df[col].fillna(fill_val)

    return train_df, val_df, test_df, imputation_values


def fit_and_apply_scaler(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    scaler_save_path: str = "models/baseline/scaler.joblib"
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Fit StandardScaler ONLY on training data, then transform Train, Val, and Test."""
    os.makedirs(os.path.dirname(scaler_save_path), exist_ok=True)
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(train_df[feature_cols])
    X_val_scaled = scaler.transform(val_df[feature_cols])
    X_test_scaled = scaler.transform(test_df[feature_cols])

    joblib.dump(scaler, scaler_save_path)
    print(f"StandardScaler fitted on Train and saved to {scaler_save_path}")

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def run_preprocessing_pipeline(
    raw_csv_path: str = "data/raw/transactions.csv",
    output_dir: str = "data/processed",
    metrics_dir: str = "outputs/metrics"
) -> dict:
    """Execute complete data pipeline: load, feature engineer, split, impute, scale, and save."""
    print("=" * 65)
    print("      DATA PREPROCESSING & CHRONOLOGICAL SPLITTING PIPELINE    ")
    print("=" * 65)

    # 1. Load raw data
    print(f"[1/5] Loading raw transactions from {raw_csv_path}...")
    df_raw = pd.read_csv(raw_csv_path)

    # 2. Engineer features
    print("[2/5] Engineering leakage-safe features...")
    df_feat = engineer_all_features(df_raw)

    # 3. Chronological split
    print("[3/5] Performing chronological 60% / 20% / 20% split...")
    train_df, val_df, test_df = chronological_split(df_feat, train_ratio=0.60, val_ratio=0.20)

    # 4. Train-only imputation
    print("[4/5] Applying train-only imputation...")
    train_df, val_df, test_df, impute_vals = handle_imputation(train_df, val_df, test_df)

    # 5. Fit scaler on train only
    fit_and_apply_scaler(train_df, val_df, test_df, FEATURE_COLS)

    # 6. Save processed splits to disk
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    cols_to_save = METADATA_COLS + FEATURE_COLS + [TARGET_COL]
    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df[cols_to_save].to_csv(train_path, index=False)
    val_df[cols_to_save].to_csv(val_path, index=False)
    test_df[cols_to_save].to_csv(test_path, index=False)

    # Summary statistics
    summary = {
        "total_records": len(df_raw),
        "feature_count": len(FEATURE_COLS),
        "feature_names": FEATURE_COLS,
        "imputation_values": impute_vals,
        "splits": {
            "train": {
                "count": len(train_df),
                "fraud_count": int(train_df[TARGET_COL].sum()),
                "fraud_rate_pct": round(float(train_df[TARGET_COL].mean() * 100), 2),
                "start_time": str(train_df["TX_DATETIME"].min()),
                "end_time": str(train_df["TX_DATETIME"].max()),
            },
            "validation": {
                "count": len(val_df),
                "fraud_count": int(val_df[TARGET_COL].sum()),
                "fraud_rate_pct": round(float(val_df[TARGET_COL].mean() * 100), 2),
                "start_time": str(val_df["TX_DATETIME"].min()),
                "end_time": str(val_df["TX_DATETIME"].max()),
            },
            "test": {
                "count": len(test_df),
                "fraud_count": int(test_df[TARGET_COL].sum()),
                "fraud_rate_pct": round(float(test_df[TARGET_COL].mean() * 100), 2),
                "start_time": str(test_df["TX_DATETIME"].min()),
                "end_time": str(test_df["TX_DATETIME"].max()),
            }
        }
    }

    summary_path = os.path.join(metrics_dir, "data_split_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    print("\n----------------- DATA SPLIT SUMMARY -----------------")
    print(f"Train Set      : {summary['splits']['train']['count']:,} rows | Frauds: {summary['splits']['train']['fraud_count']:,} ({summary['splits']['train']['fraud_rate_pct']}%)")
    print(f"  Time Window  : {summary['splits']['train']['start_time']} -> {summary['splits']['train']['end_time']}")
    print(f"Validation Set : {summary['splits']['validation']['count']:,} rows | Frauds: {summary['splits']['validation']['fraud_count']:,} ({summary['splits']['validation']['fraud_rate_pct']}%)")
    print(f"  Time Window  : {summary['splits']['validation']['start_time']} -> {summary['splits']['validation']['end_time']}")
    print(f"Test Set       : {summary['splits']['test']['count']:,} rows | Frauds: {summary['splits']['test']['fraud_count']:,} ({summary['splits']['test']['fraud_rate_pct']}%)")
    print(f"  Time Window  : {summary['splits']['test']['start_time']} -> {summary['splits']['test']['end_time']}")
    print(f"\nSaved processed splits to: {output_dir}/")
    print(f"Saved metadata summary to: {summary_path}")
    print("=====================================================")

    return summary


if __name__ == "__main__":
    run_preprocessing_pipeline()
