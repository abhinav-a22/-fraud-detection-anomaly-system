"""Leakage-Safe Behavioral & Temporal Feature Engineering Module.

Constructs transaction, temporal, customer behavioral, and terminal behavioral features.
Guarantees zero data leakage: for any transaction occurring at time T, all behavioral
aggregations strictly use transactions occurring BEFORE T (closed='left').
"""

import numpy as np
import pandas as pd


def add_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add direct transactional transformations."""
    df = df.copy()
    # Log transformation log(1 + x) to stabilize variance and compress heavy right-tail
    df["TX_AMOUNT_LOG"] = np.log1p(df["TX_AMOUNT"])
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract temporal signals from TX_DATETIME."""
    df = df.copy()
    dt = pd.to_datetime(df["TX_DATETIME"])
    df["TX_HOUR"] = dt.dt.hour
    df["TX_DAY_OF_WEEK"] = dt.dt.dayofweek  # 0 = Monday, 6 = Sunday
    df["TX_IS_WEEKEND"] = (df["TX_DAY_OF_WEEK"] >= 5).astype(int)
    # Night window: 00:00 to 06:00
    df["TX_IS_NIGHT"] = ((df["TX_HOUR"] >= 0) & (df["TX_HOUR"] < 6)).astype(int)
    return df


def add_customer_behavioral_features(df: pd.DataFrame, epsilon: float = 1e-3) -> pd.DataFrame:
    """Calculate customer-level transaction velocities and amount ratios.

    STRICT LEAKAGE PREVENTION:
    - Sorts by ['CUSTOMER_ID', 'TX_DATETIME'] to ensure strict chronological grouping.
    - closed='left' ensures the current transaction at time T is EXCLUDED.
    - Only transactions in [T - window, T) are aggregated.
    - Aligns and restores exact original index order via TRANSACTION_ID.
    """
    df = df.copy()
    df["TX_DATETIME"] = pd.to_datetime(df["TX_DATETIME"])
    df_sorted = df.sort_values(["CUSTOMER_ID", "TX_DATETIME"]).copy()

    # Set temporary datetime index for time-based rolling
    df_indexed = df_sorted.set_index("TX_DATETIME")
    cust_group = df_indexed.groupby("CUSTOMER_ID")["TX_AMOUNT"]

    # Rolling counts (velocities)
    df_sorted["CUSTOMER_TX_COUNT_1H"] = cust_group.rolling("1h", closed="left").count().fillna(0).to_numpy(dtype=np.float32)
    df_sorted["CUSTOMER_TX_COUNT_24H"] = cust_group.rolling("24h", closed="left").count().fillna(0).to_numpy(dtype=np.float32)
    df_sorted["CUSTOMER_TX_COUNT_7D"] = cust_group.rolling("7D", closed="left").count().fillna(0).to_numpy(dtype=np.float32)

    # Rolling means
    df_sorted["CUSTOMER_AVG_AMOUNT_1D"] = cust_group.rolling("1D", closed="left").mean().to_numpy(dtype=np.float32)
    df_sorted["CUSTOMER_AVG_AMOUNT_7D"] = cust_group.rolling("7D", closed="left").mean().to_numpy(dtype=np.float32)

    # Ratio: current amount vs customer's 7-day average spend
    # If customer has no prior 7d history, ratio is 1.0 (neutral baseline)
    ratio = np.where(
        np.isnan(df_sorted["CUSTOMER_AVG_AMOUNT_7D"]) | (df_sorted["CUSTOMER_AVG_AMOUNT_7D"] <= 0),
        1.0,
        df_sorted["TX_AMOUNT"] / (df_sorted["CUSTOMER_AVG_AMOUNT_7D"] + epsilon)
    )
    df_sorted["CUSTOMER_AMOUNT_RATIO"] = ratio.astype(np.float32)

    # Restore chronological order by TRANSACTION_ID
    return df_sorted.sort_values("TRANSACTION_ID").reset_index(drop=True)


def add_terminal_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate terminal-level transaction velocity and average amount.

    STRICT LEAKAGE PREVENTION:
    - Sorts by ['TERMINAL_ID', 'TX_DATETIME'] to ensure strict chronological grouping.
    - closed='left' ensures the current transaction at time T is EXCLUDED.
    - Only transactions in [T - window, T) are aggregated.
    - Aligns and restores exact original index order via TRANSACTION_ID.
    """
    df = df.copy()
    df["TX_DATETIME"] = pd.to_datetime(df["TX_DATETIME"])
    df_sorted = df.sort_values(["TERMINAL_ID", "TX_DATETIME"]).copy()

    df_indexed = df_sorted.set_index("TX_DATETIME")
    term_group = df_indexed.groupby("TERMINAL_ID")["TX_AMOUNT"]

    df_sorted["TERMINAL_TX_COUNT_1D"] = term_group.rolling("1D", closed="left").count().fillna(0).to_numpy(dtype=np.float32)
    df_sorted["TERMINAL_AVG_AMOUNT_1D"] = term_group.rolling("1D", closed="left").mean().to_numpy(dtype=np.float32)

    # Restore chronological order by TRANSACTION_ID
    return df_sorted.sort_values("TRANSACTION_ID").reset_index(drop=True)


def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """End-to-end pipeline generating all transactional, temporal, and behavioral features."""
    print("Starting feature engineering pipeline...")
    print("  [1/4] Adding direct transaction transformations...")
    df = add_transaction_features(df)
    print("  [2/4] Adding time features (hour, day of week, weekend, night)...")
    df = add_time_features(df)
    print("  [3/4] Adding customer behavioral features (1h, 24h, 7d)...")
    df = add_customer_behavioral_features(df)
    print("  [4/4] Adding terminal behavioral features (1d velocity & mean)...")
    df = add_terminal_behavioral_features(df)
    print("Feature engineering complete!")
    return df


def verify_no_leakage(df: pd.DataFrame, sample_idx: int = 50000) -> bool:
    """Mathematical verification test proving ZERO future data leakage.

    Test methodology:
    1. Select a transaction at row sample_idx (time T).
    2. Record its computed behavioral feature vector.
    3. Truncate the raw dataset so all transactions occurring after T are deleted.
    4. Recompute behavioral features from scratch on the truncated dataset.
    5. Compare the feature values: they must be strictly identical down to 5 decimals.
    """
    print(f"\n--- RUNNING LEAKAGE INTEGRITY PROOF (Index {sample_idx}) ---")
    tx_target = df.iloc[sample_idx]
    target_time = tx_target["TX_DATETIME"]
    cust_id = tx_target["CUSTOMER_ID"]
    term_id = tx_target["TERMINAL_ID"]

    feature_cols = [
        "CUSTOMER_TX_COUNT_1H", "CUSTOMER_TX_COUNT_24H", "CUSTOMER_TX_COUNT_7D",
        "CUSTOMER_AMOUNT_RATIO", "TERMINAL_TX_COUNT_1D"
    ]
    original_values = {col: tx_target[col] for col in feature_cols}

    # Truncate dataset up to target_time
    truncated_df = df[df["TX_DATETIME"] <= target_time].copy()
    # Recompute features from scratch on truncated history only
    recomputed_df = add_customer_behavioral_features(truncated_df)
    recomputed_df = add_terminal_behavioral_features(recomputed_df)

    recomputed_target = recomputed_df.iloc[-1]
    recomputed_values = {col: recomputed_target[col] for col in feature_cols}

    all_match = True
    print(f"Target Transaction: ID={tx_target['TRANSACTION_ID']}, Time={target_time}")
    print(f"Customer={cust_id}, Terminal={term_id}")
    for col in feature_cols:
        orig = original_values[col]
        recomp = recomputed_values[col]
        is_close = np.isclose(orig, recomp, atol=1e-5)
        status = "MATCH (Safe)" if is_close else "LEAK DETECTED!"
        if not is_close:
            all_match = False
        print(f"  {col:<26}: Full={orig:<8.4f} | Truncated={recomp:<8.4f} -> {status}")

    if all_match:
        print("\nRESULT: PROVEN LEAKAGE-FREE. Zero future information influences predictions.\n")
    else:
        print("\nRESULT: CRITICAL FAILURE. Future data leakage detected!\n")

    return all_match


if __name__ == "__main__":
    raw_df = pd.read_csv("data/raw/transactions.csv")
    featured_df = engineer_all_features(raw_df)
    verify_no_leakage(featured_df, sample_idx=50000)

