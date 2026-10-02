"""Utility functions for logging, reproducibility, and dataset inspection."""

import os
import random
import numpy as np
import pandas as pd


def set_seed(seed: int = 42) -> None:
    """Set global random seed for numpy, random, and python hash."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def inspect_dataset(df: pd.DataFrame, name: str = "Dataset") -> dict:
    """Print and return comprehensive summary statistics of transaction dataset."""
    total_tx = len(df)
    fraud_tx = int(df["TX_FRAUD"].sum()) if "TX_FRAUD" in df.columns else 0
    fraud_rate = (fraud_tx / total_tx * 100) if total_tx > 0 else 0.0

    print("=" * 60)
    print(f"               INSPECTION: {name.upper()}")
    print("=" * 60)
    print(f"Total Rows (Transactions) : {total_tx:,}")
    print(f"Total Columns             : {len(df.columns)}")
    print(f"Date Range                : {df['TX_DATETIME'].min()} -> {df['TX_DATETIME'].max()}")
    print(f"Legitimate Transactions   : {total_tx - fraud_tx:,} ({100 - fraud_rate:.2f}%)")
    print(f"Fraudulent Transactions   : {fraud_tx:,} ({fraud_rate:.2f}%)")
    print(f"Unique Customers          : {df['CUSTOMER_ID'].nunique():,}")
    print(f"Unique Terminals          : {df['TERMINAL_ID'].nunique():,}")
    print("-" * 60)
    print("Column Null Value Counts:")
    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        dtype_str = str(df[col].dtype)
        print(f"  {col:<24} | Type: {dtype_str:<10} | Nulls: {null_count}")
    print("-" * 60)
    print("TX_AMOUNT Distribution:")
    stats = df["TX_AMOUNT"].describe()
    for k, v in stats.items():
        print(f"  {k:<10}: {v:.2f}")
    print("=" * 60)

    return {
        "total_tx": total_tx,
        "fraud_tx": fraud_tx,
        "fraud_rate": fraud_rate,
        "unique_customers": int(df["CUSTOMER_ID"].nunique()),
        "unique_terminals": int(df["TERMINAL_ID"].nunique()),
    }


if __name__ == "__main__":
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "transactions.csv")
    if os.path.exists(csv_path):
        df_raw = pd.read_csv(csv_path)
        inspect_dataset(df_raw, name="Raw Transactions Simulator")
    else:
        print(f"File not found: {csv_path}")

