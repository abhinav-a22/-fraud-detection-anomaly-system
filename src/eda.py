"""Exploratory Data Analysis (EDA) module for Fraud Detection System.

Analyzes class imbalance, transaction amounts, temporal distributions,
and customer/terminal behavioral dynamics. Generates diagnostic plots
in outputs/plots/ and summarizes key statistical takeaways.
"""

import os
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def load_raw_data(data_path: str = "data/raw/transactions.csv") -> pd.DataFrame:
    """Load raw transactions CSV and parse datetime."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Raw dataset not found at {data_path}. Run src/generate_data.py first.")
    df = pd.read_csv(data_path)
    df["TX_DATETIME"] = pd.to_datetime(df["TX_DATETIME"])
    return df


def perform_eda(df: pd.DataFrame, output_dir: str = "outputs/plots") -> dict:
    """Run full exploratory data analysis and export diagnostic figures."""
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    
    total_tx = len(df)
    fraud_tx = int(df["TX_FRAUD"].sum())
    legit_tx = total_tx - fraud_tx
    fraud_rate = (fraud_tx / total_tx) * 100.0

    print("=" * 65)
    print("           EXPLORATORY DATA ANALYSIS (EDA) REPORT")
    print("=" * 65)
    print(f"Total Transactions : {total_tx:,}")
    print(f"Legitimate [Class 0]: {legit_tx:,} ({100 - fraud_rate:.2f}%)")
    print(f"Fraudulent [Class 1]: {fraud_tx:,} ({fraud_rate:.2f}%)")
    print(f"Class Imbalance Ratio: 1 fraud per {legit_tx // fraud_tx:.0f} legitimate transactions (~1 : {legit_tx / fraud_tx:.1f})")
    print("-" * 65)

    # 1. Amount Statistics by Class
    legit_amounts = df[df["TX_FRAUD"] == 0]["TX_AMOUNT"]
    fraud_amounts = df[df["TX_FRAUD"] == 1]["TX_AMOUNT"]

    print("Transaction Amount Comparison ($):")
    print(f"  Legitimate Mean : ${legit_amounts.mean():.2f} (Median: ${legit_amounts.median():.2f}, Std: ${legit_amounts.std():.2f})")
    print(f"  Fraudulent Mean : ${fraud_amounts.mean():.2f} (Median: ${fraud_amounts.median():.2f}, Std: ${fraud_amounts.std():.2f})")
    print(f"  Legitimate Max  : ${legit_amounts.max():.2f} | Fraudulent Max : ${fraud_amounts.max():.2f}")
    print("-" * 65)

    # ---------------- PLOT 1: Class Imbalance ----------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # Bar plot of counts
    counts = [legit_tx, fraud_tx]
    labels = ["Legitimate (0)", "Fraud (1)"]
    colors = ["#2b5c8f", "#d95f02"]
    axes[0].bar(labels, counts, color=colors, edgecolor="black", alpha=0.85)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Count (Log Scale)")
    axes[0].set_title("Class Distribution (Log Scale)", fontsize=12, fontweight="bold")
    for i, v in enumerate(counts):
        axes[0].text(i, v * 1.3, f"{v:,}\n({v/total_tx*100:.2f}%)", ha="center", fontweight="bold")

    # Donut chart
    axes[1].pie([legit_tx, fraud_tx], labels=labels, colors=colors, autopct="%1.2f%%",
                startangle=140, explode=(0, 0.15), wedgeprops=dict(width=0.4, edgecolor="white"))
    axes[1].set_title("Proportion of Target Classes", fontsize=12, fontweight="bold")
    
    plt.tight_layout()
    p1 = os.path.join(output_dir, "eda_class_imbalance.png")
    fig.savefig(p1, dpi=200)
    plt.close(fig)
    print(f"[Plot Saved] Class imbalance -> {p1}")

    # ---------------- PLOT 2: Amount Distribution (Raw vs Log) ----------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    
    # Raw amounts
    sns.kdeplot(data=df, x="TX_AMOUNT", hue="TX_FRAUD", common_norm=False,
                palette={0: "#2b5c8f", 1: "#d95f02"}, fill=True, alpha=0.3, ax=axes[0], log_scale=False)
    axes[0].set_title("Raw Transaction Amount Distribution", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("TX_AMOUNT ($)")
    axes[0].set_ylabel("Density")
    axes[0].legend(title="TX_FRAUD", labels=["Fraud (1)", "Legitimate (0)"])

    # Log amounts
    df["TX_AMOUNT_LOG"] = np.log1p(df["TX_AMOUNT"])
    sns.kdeplot(data=df, x="TX_AMOUNT_LOG", hue="TX_FRAUD", common_norm=False,
                palette={0: "#2b5c8f", 1: "#d95f02"}, fill=True, alpha=0.3, ax=axes[1])
    axes[1].set_title("Log-Transformed Amount Distribution: log(1 + TX_AMOUNT)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("TX_AMOUNT_LOG")
    axes[1].set_ylabel("Density")
    axes[1].legend(title="TX_FRAUD", labels=["Fraud (1)", "Legitimate (0)"])

    plt.tight_layout()
    p2 = os.path.join(output_dir, "eda_amount_distribution.png")
    fig.savefig(p2, dpi=200)
    plt.close(fig)
    print(f"[Plot Saved] Amount distributions -> {p2}")

    # ---------------- PLOT 3: Daily Timeline (Volume & Fraud Rate) ----------------
    daily_stats = df.groupby("TX_TIME_DAYS").agg(
        total_tx=("TX_FRAUD", "count"),
        fraud_tx=("TX_FRAUD", "sum"),
    ).reset_index()
    daily_stats["fraud_rate_pct"] = (daily_stats["fraud_tx"] / daily_stats["total_tx"]) * 100.0

    fig, ax1 = plt.subplots(figsize=(13, 5))
    ax2 = ax1.twinx()

    line1 = ax1.plot(daily_stats["TX_TIME_DAYS"], daily_stats["total_tx"], color="#2b5c8f",
                     linewidth=1.8, label="Daily Total Volume")
    line2 = ax2.plot(daily_stats["TX_TIME_DAYS"], daily_stats["fraud_rate_pct"], color="#d95f02",
                     linewidth=2.2, linestyle="--", label="Daily Fraud Rate (%)")

    ax1.set_xlabel("Simulation Day (0 to 59)", fontsize=11)
    ax1.set_ylabel("Total Transactions / Day", color="#2b5c8f", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Fraud Rate (%)", color="#d95f02", fontsize=11, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax2.grid(False)

    # Shading Chronological Split zones
    # 60% Train (Days 0-35), 20% Val (Days 36-47), 20% Test (Days 48-59)
    ax1.axvspan(0, 35.9, color="#2b5c8f", alpha=0.08, label="Train Period (60%)")
    ax1.axvspan(36, 47.9, color="#7570b3", alpha=0.10, label="Validation Period (20%)")
    ax1.axvspan(48, 59, color="#e7298a", alpha=0.10, label="Test Period (20%)")

    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines] + ["Train (60%)", "Val (20%)", "Test (20%)"]
    ax1.legend(loc="upper left")

    plt.title("Transaction Volume & Fraud Rate Across Time (with 60/20/20 Chronological Split)",
              fontsize=12, fontweight="bold")
    plt.tight_layout()
    p3 = os.path.join(output_dir, "eda_fraud_over_time.png")
    fig.savefig(p3, dpi=200)
    plt.close(fig)
    print(f"[Plot Saved] Daily timeline & splits -> {p3}")

    # ---------------- PLOT 4: Hourly Distribution & Night Activity ----------------
    df["TX_HOUR"] = df["TX_DATETIME"].dt.hour
    hourly_stats = df.groupby("TX_HOUR").agg(
        total_tx=("TX_FRAUD", "count"),
        fraud_tx=("TX_FRAUD", "sum")
    ).reset_index()
    hourly_stats["fraud_rate_pct"] = (hourly_stats["fraud_tx"] / hourly_stats["total_tx"]) * 100.0

    fig, ax = plt.subplots(figsize=(12, 4.5))
    sns.barplot(data=hourly_stats, x="TX_HOUR", y="fraud_rate_pct", ax=ax, color="#1b9e77", edgecolor="black")
    ax.axvspan(-0.5, 5.5, color="red", alpha=0.15, label="Night Window (00:00 - 06:00)")
    ax.set_title("Fraud Rate (%) by Hour of Day", fontsize=12, fontweight="bold")
    ax.set_xlabel("Hour of Day (0 to 23)")
    ax.set_ylabel("Fraud Rate (%)")
    ax.legend()
    plt.tight_layout()
    p4 = os.path.join(output_dir, "eda_hourly_distribution.png")
    fig.savefig(p4, dpi=200)
    plt.close(fig)
    print(f"[Plot Saved] Hourly fraud distribution -> {p4}")

    print("=" * 65)
    print("EDA completed successfully. All diagnostic figures exported.")
    print("=" * 65)

    return {
        "total_tx": total_tx,
        "fraud_rate": fraud_rate,
        "legit_mean_amount": float(legit_amounts.mean()),
        "fraud_mean_amount": float(fraud_amounts.mean()),
        "plots": [p1, p2, p3, p4]
    }


if __name__ == "__main__":
    raw_df = load_raw_data()
    perform_eda(raw_df)

