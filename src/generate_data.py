"""Fraud Detection Handbook synthetic transaction generator.

Generates synthetic credit card transaction data modeling customer spending profiles,
terminal locations, and real-world fraud scenarios following the benchmark design from
the 'Reproducible Machine Learning for Credit Card Fraud Detection' handbook
(Le Borgne et al.).
"""

import argparse
import os
import time
import numpy as np
import pandas as pd
from scipy.spatial import KDTree


def generate_customer_profiles(n_customers: int, seed: int = 42) -> pd.DataFrame:
    """Generate spatial and spending behavioral profiles for customers."""
    rng = np.random.default_rng(seed)
    x_cust = rng.uniform(0, 100, n_customers)
    y_cust = rng.uniform(0, 100, n_customers)
    mean_amount = rng.uniform(10, 80, n_customers)
    std_amount = mean_amount / 2.0
    mean_tx_per_day = rng.uniform(0.5, 1.8, n_customers)

    return pd.DataFrame({
        "CUSTOMER_ID": np.arange(n_customers, dtype=np.int32),
        "x_customer": x_cust,
        "y_customer": y_cust,
        "mean_amount": mean_amount,
        "std_amount": std_amount,
        "mean_tx_per_day": mean_tx_per_day,
    })


def generate_terminal_profiles(n_terminals: int, seed: int = 42) -> pd.DataFrame:
    """Generate spatial locations for payment terminals."""
    rng = np.random.default_rng(seed + 1)
    x_term = rng.uniform(0, 100, n_terminals)
    y_term = rng.uniform(0, 100, n_terminals)

    return pd.DataFrame({
        "TERMINAL_ID": np.arange(n_terminals, dtype=np.int32),
        "x_terminal": x_term,
        "y_terminal": y_term,
    })


def associate_terminals(
    customer_profiles: pd.DataFrame,
    terminal_profiles: pd.DataFrame,
    radius: float = 25.0
) -> list:
    """Map each customer to terminals within geographic radius r."""
    term_coords = terminal_profiles[["x_terminal", "y_terminal"]].to_numpy()
    cust_coords = customer_profiles[["x_customer", "y_customer"]].to_numpy()

    term_tree = KDTree(term_coords)
    available_terminals = term_tree.query_ball_point(cust_coords, r=radius)

    for i in range(len(available_terminals)):
        if len(available_terminals[i]) == 0:
            _, idx = term_tree.query(cust_coords[i], k=1)
            available_terminals[i] = [int(idx)]
        available_terminals[i] = np.array(available_terminals[i], dtype=np.int32)

    return available_terminals


def generate_transactions(
    customer_profiles: pd.DataFrame,
    available_terminals: list,
    nb_days: int = 60,
    start_date: str = "2026-01-01",
    seed: int = 42
) -> pd.DataFrame:
    """Generate legitimate transactions based on customer profiles and Poisson arrivals."""
    rng = np.random.default_rng(seed + 2)
    n_customers = len(customer_profiles)
    mean_tx_per_day = customer_profiles["mean_tx_per_day"].to_numpy()
    mean_amount = customer_profiles["mean_amount"].to_numpy()
    std_amount = customer_profiles["std_amount"].to_numpy()

    records = []

    for day in range(nb_days):
        daily_counts = rng.poisson(mean_tx_per_day)
        active_customers = np.where(daily_counts > 0)[0]

        for c_id in active_customers:
            k = daily_counts[c_id]
            # Transaction times uniform across the 24 hours of the day
            times = day * 86400 + rng.uniform(0, 86400, k)
            # Transaction amounts normally distributed around customer average
            amounts = rng.normal(mean_amount[c_id], std_amount[c_id], k)
            amounts = np.clip(amounts, 1.0, 500.0)
            terms = rng.choice(available_terminals[c_id], size=k)

            for t, a, term in zip(times, amounts, terms):
                records.append((t, c_id, term, round(float(a), 2)))

    df = pd.DataFrame(records, columns=["TX_TIME_SECONDS", "CUSTOMER_ID", "TERMINAL_ID", "TX_AMOUNT"])
    df = df.sort_values("TX_TIME_SECONDS").reset_index(drop=True)
    df["TRANSACTION_ID"] = np.arange(len(df), dtype=np.int64)
    start_ts = pd.to_datetime(start_date)
    df["TX_DATETIME"] = start_ts + pd.to_timedelta(df["TX_TIME_SECONDS"], unit="s")
    df["TX_TIME_DAYS"] = (df["TX_TIME_SECONDS"] // 86400).astype(np.int32)

    # Reorder initial columns
    df = df[["TRANSACTION_ID", "TX_DATETIME", "CUSTOMER_ID", "TERMINAL_ID", "TX_AMOUNT", "TX_TIME_SECONDS", "TX_TIME_DAYS"]]
    return df


def inject_fraud_scenarios(
    transactions_df: pd.DataFrame,
    n_customers: int,
    n_terminals: int,
    nb_days: int = 60,
    seed: int = 42
) -> pd.DataFrame:
    """Inject three benchmark fraud scenarios from the Fraud Detection Handbook.

    Scenario 1: High amount transactions (> 220) are marked fraudulent.
    Scenario 2: Compromised terminals (2 terminals compromised per day for 28 days).
    Scenario 3: Compromised customers / phishing (2 customers compromised per day,
                1/3 of their transactions amount x 5 for 14 days).
    """
    rng = np.random.default_rng(seed + 3)
    df = transactions_df.copy()
    df["TX_FRAUD"] = 0
    df["TX_FRAUD_SCENARIO"] = 0

    # Scenario 1: Amount threshold
    s1_mask = df["TX_AMOUNT"] > 220
    df.loc[s1_mask, "TX_FRAUD"] = 1
    df.loc[s1_mask, "TX_FRAUD_SCENARIO"] = 1

    # Scenario 2: Compromised terminals
    for d in range(nb_days):
        compromised_terminals = rng.choice(n_terminals, size=2, replace=False)
        mask = (
            df["TERMINAL_ID"].isin(compromised_terminals) &
            (df["TX_TIME_DAYS"] >= d) &
            (df["TX_TIME_DAYS"] < d + 28)
        )
        df.loc[mask, "TX_FRAUD"] = 1
        df.loc[mask, "TX_FRAUD_SCENARIO"] = 2

    # Scenario 3: Compromised customers (account takeover / phishing)
    for d in range(nb_days):
        compromised_customers = rng.choice(n_customers, size=2, replace=False)
        cust_mask = (
            df["CUSTOMER_ID"].isin(compromised_customers) &
            (df["TX_TIME_DAYS"] >= d) &
            (df["TX_TIME_DAYS"] < d + 14)
        )
        candidate_indices = df[cust_mask].index.to_numpy()
        if len(candidate_indices) > 0:
            k_compromised = max(1, len(candidate_indices) // 3)
            selected_indices = rng.choice(candidate_indices, size=k_compromised, replace=False)
            df.loc[selected_indices, "TX_AMOUNT"] = (df.loc[selected_indices, "TX_AMOUNT"] * 5.0).round(2)
            df.loc[selected_indices, "TX_FRAUD"] = 1
            df.loc[selected_indices, "TX_FRAUD_SCENARIO"] = 3

    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic transaction dataset")
    parser.add_argument("--n_customers", type=int, default=3500, help="Number of customer profiles")
    parser.add_argument("--n_terminals", type=int, default=5000, help="Number of terminal profiles")
    parser.add_argument("--nb_days", type=int, default=60, help="Number of days to simulate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default="data/raw/transactions.csv", help="Path to save raw dataset")
    parser.add_argument("--start_date", type=str, default="2026-01-01", help="Start date of simulation")

    args = parser.parse_args()

    print("==================================================")
    print("   FRAUD DETECTION HANDBOOK DATASET GENERATOR     ")
    print("==================================================")
    print(f"Parameters: {args.n_customers} customers, {args.n_terminals} terminals, {args.nb_days} days")
    print(f"Start date: {args.start_date}, Seed: {args.seed}")
    
    t0 = time.time()
    print("\n[1/4] Generating customer and terminal spatial profiles...")
    cust_df = generate_customer_profiles(args.n_customers, seed=args.seed)
    term_df = generate_terminal_profiles(args.n_terminals, seed=args.seed)

    print("[2/4] Mapping customer terminal associations (KDTree radius = 25)...")
    available_terms = associate_terminals(cust_df, term_df, radius=25.0)

    print("[3/4] Generating baseline transactions across timeline...")
    tx_df = generate_transactions(
        cust_df, available_terms, nb_days=args.nb_days, start_date=args.start_date, seed=args.seed
    )

    print("[4/4] Injecting fraud scenarios (1: High amount, 2: Terminal compromise, 3: Account takeover)...")
    final_df = inject_fraud_scenarios(
        tx_df, n_customers=args.n_customers, n_terminals=args.n_terminals, nb_days=args.nb_days, seed=args.seed
    )

    # Save to disk
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    final_df.to_csv(args.output, index=False)
    elapsed = time.time() - t0

    fraud_count = int(final_df["TX_FRAUD"].sum())
    total_tx = len(final_df)
    fraud_pct = (fraud_count / total_tx) * 100.0
    file_size_mb = os.path.getsize(args.output) / (1024 * 1024)

    print("\n---------------- GENERATION SUMMARY ----------------")
    print(f"Total Transactions Generated : {total_tx:,}")
    print(f"Total Frauds                 : {fraud_count:,} ({fraud_pct:.2f}%)")
    print(f"Total Legitimate             : {total_tx - fraud_count:,} ({100 - fraud_pct:.2f}%)")
    print(f"Time Horizon                 : {final_df['TX_DATETIME'].min()} to {final_df['TX_DATETIME'].max()}")
    print("\nFraud Breakdown by Scenario:")
    scenarios = {
        0: "Legitimate (No Fraud)",
        1: "Scenario 1: High Transaction Amount (> 220)",
        2: "Scenario 2: Compromised Terminal Window",
        3: "Scenario 3: Compromised Customer (Account Takeover)"
    }
    counts = final_df["TX_FRAUD_SCENARIO"].value_counts().sort_index()
    for sc_id, count in counts.items():
        print(f"  [{sc_id}] {scenarios.get(sc_id, 'Unknown')}: {count:,} ({count / total_tx * 100:.2f}%)")

    print(f"\nSaved raw dataset to: {args.output} ({file_size_mb:.2f} MB)")
    print(f"Completed in: {elapsed:.2f} seconds")
    print("==================================================")


if __name__ == "__main__":
    main()

