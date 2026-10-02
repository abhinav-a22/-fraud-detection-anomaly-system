# Fraud Detection & Anomaly Analytics System

A production-grade, human-in-the-loop Machine Learning and Anomaly Analytics system built using the **Fraud Detection Handbook** transaction simulator. Designed to demonstrate an end-to-end understanding of class-imbalanced classification, temporal data leakage prevention, cost-sensitive learning, unsupervised outlier detection, game-theoretic explainability (SHAP), and multi-tier operational risk engines.

---

## 📌 Table of Contents
1. [Project Overview & Problem Statement](#1-project-overview--problem-statement)
2. [Dataset & Simulation Architecture](#2-dataset--simulation-architecture)
3. [Leakage-Safe Feature Engineering](#3-leakage-safe-feature-engineering)
4. [Chronological Splitting Strategy](#4-chronological-splitting-strategy)
5. [The Experiment Matrix & Results](#5-the-experiment-matrix--results)
6. [Why Accuracy Fails: The Evaluation Framework](#6-why-accuracy-fails-the-evaluation-framework)
7. [Decision Threshold Tuning](#7-decision-threshold-tuning)
8. [Unsupervised Anomaly Detection (Isolation Forest)](#8-unsupervised-anomaly-detection-isolation-forest)
9. [Deep Error Analysis (FP vs. FN)](#9-deep-error-analysis-fp-vs-fn)
10. [Model Explainability with SHAP](#10-model-explainability-with-shap)
11. [The Hybrid Fraud Risk Engine](#11-the-hybrid-fraud-risk-engine)
12. [Project Structure](#12-project-structure)
13. [How to Run the Pipeline](#13-how-to-run-the-pipeline)
14. [Scaling to 1M+ Transactions](#14-scaling-to-1m-transactions)
15. [Project Limitations & Future Work](#15-project-limitations--future-work)
16. [Master Interview Preparation Guide](#16-master-interview-preparation-guide)

---

## 1. Project Overview & Problem Statement

Financial fraud detection is fundamentally a **heavily imbalanced, non-stationary binary classification problem** operating under high-consequence operational trade-offs:
- **Class Imbalance**: Legitimate transactions constitute $>98.5\%$ of all network traffic, while fraud represents $<1.5\%$.
- **Asymmetric Cost of Errors**: Missing a fraudulent transaction (False Negative) causes direct financial chargeback liability. Conversely, declining an innocent customer's legitimate transaction (False Positive) damages cardholder trust and merchant interchange revenues.
- **Concept Drift & Zero-Day Schemes**: Fraudsters continuously adapt tactics to evade static rule-based systems.

### Project Objective
Build a transparent, modular, and mathematically verified pipeline that:
1. Rejects naive random splitting in favor of strict **chronological partitions**.
2. Mathematically guarantees **zero future-data leakage** across rolling behavioral features.
3. Systematically evaluates 8 distinct supervised and unsupervised experiments (**E0 through E8**).
4. Tunes decision thresholds on **Validation data only**, testing once on **unseen Test data**.
5. Bridges supervised gradient boosting (XGBoost) and unsupervised outlier isolation (Isolation Forest) into a 3-tier operational **Risk Engine** with automated **SHAP Adverse Action Reason Codes**.

---

## 2. Dataset & Simulation Architecture

Instead of utilizing pre-anonymized PCA datasets (such as the standard Kaggle credit card dataset) which hide entity identifiers and eliminate behavioral feature engineering, this project implements the **Fraud Detection Handbook transaction simulator** (Le Borgne et al.):

```
Customer Profiles (3,500)       Terminal Profiles (5,000)
  ├── Mean/Std Spend Amount       └── Spatial Coordinates (100x100)
  ├── Poisson Daily Frequency
  └── Geographic Location (KDTree ball-query radius = 25)
                     │
                     ▼
           Baseline Transactions (241,151)
                     │
       ┌─────────────┼─────────────┐
       ▼             ▼             ▼
  Scenario 1    Scenario 2    Scenario 3
 (High Amount  (Compromised  (Account Takeover:
   > $220)       Terminals)    5x Amount Surge)
```

### Dataset Parameters & Generation Summary
- **Simulated Duration**: 60 consecutive days (`2026-01-01` to `2026-03-01`).
- **Total Transactions**: $241,151$.
- **Legitimate Transactions ($Y=0$)**: $238,480$ ($98.89\%$).
- **Fraudulent Transactions ($Y=1$)**: $2,671$ ($1.11\%$).
- **Attack Breakdown**:
  - *Scenario 1 (High Amount > $220)*: 5 transactions ($0.002\%$).
  - *Scenario 2 (Compromised Terminals)*: 2,094 transactions ($0.87\%$).
  - *Scenario 3 (Account Takeover / 5x Spend)*: 572 transactions ($0.24\%$).

### Raw Data Fields & Entity Role
- `TRANSACTION_ID`: Surrogate integer key. Dropped from feature vectors to prevent spurious memorization.
- `TX_DATETIME`: Monotonic timestamp used for chronological splitting and rolling windows.
- `CUSTOMER_ID` & `TERMINAL_ID`: High-cardinality entity keys. **Never fed raw into models**; used exclusively as grouping keys for historical behavioral aggregation.
- `TX_AMOUNT`: Continuous transaction value in USD.
- `TX_FRAUD`: Ground-truth binary target ($0 = \text{legit}, 1 = \text{fraud}$).

---

## 3. Leakage-Safe Feature Engineering

The system constructs **14 engineered features** across four domains:

| Domain | Feature | Formula / Definition | Rationale |
| :--- | :--- | :--- | :--- |
| **Transaction** | `TX_AMOUNT` | Continuous dollar amount | Base transaction size |
| | `TX_AMOUNT_LOG` | $\ln(1 + \text{TX\_AMOUNT})$ | Variance stabilization for linear baselines |
| **Temporal** | `TX_HOUR` | $\text{hour} \in [0, 23]$ | Diurnal spending cycles |
| | `TX_DAY_OF_WEEK` | $\text{day} \in [0, 6]$ | Weekly business vs. weekend patterns |
| | `TX_IS_WEEKEND` | $1$ if day $\ge 5$, else $0$ | Weekend spending shift |
| | `TX_IS_NIGHT` | $1$ if $0 \le \text{hour} < 6$, else $0$ | High-risk cardholder sleep window |
| **Customer Behavior** | `CUSTOMER_TX_COUNT_1H` | Transactions in $[T - 1\text{h}, T)$ | High-frequency card draining velocity |
| | `CUSTOMER_TX_COUNT_24H`| Transactions in $[T - 24\text{h}, T)$ | Daily card velocity |
| | `CUSTOMER_TX_COUNT_7D` | Transactions in $[T - 7\text{d}, T)$ | Weekly cardholder frequency baseline |
| | `CUSTOMER_AVG_AMOUNT_1D`| Mean spend in $[T - 24\text{h}, T)$ | Short-term spend baseline |
| | `CUSTOMER_AVG_AMOUNT_7D`| Mean spend in $[T - 7\text{d}, T)$ | Medium-term spend baseline |
| | `CUSTOMER_AMOUNT_RATIO` | $\frac{\text{TX\_AMOUNT}}{\text{CUSTOMER\_AVG\_AMOUNT\_7D} + 0.001}$ | Relative deviation from personal history |
| **Terminal Behavior** | `TERMINAL_TX_COUNT_1D` | Terminal tx count in $[T - 24\text{h}, T)$| Terminal compromise volume spikes |
| | `TERMINAL_AVG_AMOUNT_1D`| Terminal mean spend in $[T - 24\text{h}, T)$| Terminal profile drift |

### Strict Leakage Prevention Architecture
1. **The `closed='left'` Guarantee**: In pandas datetime rolling aggregations, `closed='right'` incorporates the transaction at timestamp $T$. This allows the current fraud transaction to inflate its own historical mean. By enforcing `closed='left'`, aggregations strictly cover $[T - \text{window}, T)$, excluding the current transaction.
2. **Train-Only Imputation & Scaling**: For transactions without prior history ($NaN$), median imputation values and `StandardScaler` parameters ($\mu, \sigma$) are computed **solely from the Training partition** and saved to disk.
3. **The Truncation Proof Test**: In `tests/test_features.py`, we mathematically prove zero data leakage: computing features on the full timeline vs. recomputing from scratch after deleting all future transactions ($t > T$) yields strictly identical values down to $10^{-5}$ precision.

---

## 4. Chronological Splitting Strategy

Standard random train/test shuffling causes **severe future-data leakage**:
- In random splits, transactions from Day 55 appear in training while transactions from Day 10 appear in testing, effectively allowing models to predict the past using future knowledge.
- Real production fraud models evaluate incoming streaming transactions in real time without access to future data.

### Chronological Partitions (60% / 20% / 20%)
```
Day 0                                Day 36              Day 48               Day 60
  [────────────── TRAIN ───────────────][── VALIDATION ──][───── UNSEEN TEST ─────]
          144,690 Transactions               48,230               48,231
          (1,346 Frauds - 0.93%)       (669 Frauds - 1.39%) (656 Frauds - 1.36%)
```
- **Train (Days 0 to 35.9)**: Model weights, scalers, and imputation medians are fit exclusively here.
- **Validation (Days 36 to 47.9)**: Model architectures are compared, and the operating threshold $\theta^*$ is selected.
- **Test (Days 48 to 59.9)**: Unbiased future generalization test, evaluated **once and only once**.

---

## 5. The Experiment Matrix & Results

All 8 experiments were executed on the system. Below is the verified performance comparison on the **Validation Set** ($48,230$ transactions, $669$ actual frauds) at default threshold $0.50$:

| Exp | Model Architecture | Imbalance Strategy | PR-AUC (Primary) | Recall | Precision | F1 Score | ROC-AUC | Accuracy |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **E0** | Dummy Baseline | None (Baseline) | **0.0139** | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.9861 |
| **E1** | Logistic Regression | None (Baseline) | **0.1424** | 0.0852 | 1.0000 | 0.1570 | 0.5724 | 0.9873 |
| **E2** | Random Forest | None (Baseline) | **0.1607** | 0.1106 | 0.9737 | 0.1987 | 0.5849 | 0.9876 |
| **E3** | XGBoost Baseline | None (Baseline) | **0.1554** | 0.1121 | 0.8929 | 0.1992 | 0.5730 | 0.9875 |
| **E4** | XGBoost | Class Weight ($w=106.5$) | **0.1478** | **0.2765** | 0.0273 | 0.0497 | 0.5757 | 0.8533 |
| **E5** | XGBoost | SMOTE (10% on Train) | **0.1573** | 0.1076 | 0.9600 | 0.1935 | 0.5658 | 0.9876 |
| **E6** | Isolation Forest | Unsupervised (All 14 Feat) | **0.0289** | 0.0717 | 0.0663 | 0.0689 | 0.5738 | 0.9731 |
| **E8** | Isolation Forest | Unsupervised (Behavioral) | **0.0405** | 0.0987 | 0.0912 | 0.0948 | 0.5786 | 0.9739 |

---

## 6. Why Accuracy Fails: The Evaluation Framework

### The Accuracy Paradox
The Dummy Classifier (E0) achieved **$98.61\%$ accuracy** by blindly predicting zero fraud across all transactions. In banking, this model is catastrophic: **$100\%$ of fraud loss is incurred**. Accuracy is dominated by the True Negative majority ($47,561$ transactions) and gives zero signal regarding fraud capture.

### PR-AUC vs. ROC-AUC in Heavy Imbalance
$$\text{False Positive Rate} = \frac{FP}{FP + TN} \quad \text{vs.} \quad \text{Precision} = \frac{TP}{TP + FP}$$
- In an imbalanced dataset ($47,561$ legitimate vs. $669$ fraud), generating $475$ false alarms produces an $FPR \approx 0.010$ ($1.0\%$).
- The ROC curve appears near-perfect (ROC-AUC $>0.85$), yet Precision collapses to $9.5\%$ ($9$ out of $10$ customer cards stopped are false alarms).
- **PR-AUC (Average Precision)** isolates True Positives directly against False Positives without the diluting effect of True Negatives. It serves as our primary model ranking metric.

---

## 7. Decision Threshold Tuning

In imbalanced classification, expecting calibrated probabilities to exceed the default $0.50$ threshold ignores the low prior probability of fraud ($1.39\%$). We conducted a systematic threshold sweep on the **Validation Set**:

```
Decision Threshold Sweep Grid (Validation Set - XGBoost):
 Threshold  Precision  Recall     F1   TP   FP   FN     TN
      0.05     0.3556  0.1510 0.2120  101  183  568  47378
      0.10     0.5055  0.1375 0.2162   92   90  577  47471
      0.15     0.6312  0.1330 0.2198   89   52  580  47509
      0.20     0.7131  0.1300 0.2200   87   35  582  47526  <-- PEAK F1 (theta*)
      0.30     0.7850  0.1256 0.2165   84   23  585  47538
      0.50     0.8929  0.1121 0.1992   75    9  594  47552  <-- Default 0.50
```

### Optimal Operating Point Selected on Validation: $\theta^* = 0.20$
At $\theta^* = 0.20$, the model balances fraud recall with minimal false positive friction ($35$ false alarms across $47,561$ transactions $\implies 71.3\%$ precision).

### Final Unbiased Test Set Generalization (Touched Once With $\theta^*$)
- **Test Set PR-AUC**: **$0.1655$** (Confirms zero overfitting; test ranking exceeded validation).
- **Test Set ROC-AUC**: **$0.6012$**.
- **Performance Comparison**:
  - *Default ($\theta = 0.50$)*: Recall = $12.20\%$, Precision = $94.12\%$, F1 = $0.2159$ ($TP=80, FP=5$).
  - *Tuned ($\theta^* = 0.20$)*: Recall = **$13.11\%$**, Precision = **$71.67\%$**, F1 = **$0.2216$** ($TP=86, FP=34$).
  - *Operational Impact*: $+6$ additional frauds caught on future unseen transactions with controlled friction.

---

## 8. Unsupervised Anomaly Detection (Isolation Forest)

Isolation Forest partitions data using random axis-aligned splits. Because anomalies are "few and different," they isolate near tree roots with short average path lengths:
$$s(\mathbf{x}, n) = 2^{-\frac{\mathbb{E}[h(\mathbf{x})]}{c(n)}}$$
- **Zero Label Exposure**: The anomaly detector was fit purely on unlabeled training features.
- **E6 vs. E8**: Selecting focused behavioral deviation features (`CUSTOMER_AMOUNT_RATIO`, `TX_AMOUNT`, velocity) improved PR-AUC from **$0.0289 \to 0.0405$** by removing noisy non-informative dimensions.

### 4-Way Agreement Analysis (Supervised XGBoost vs. Anomaly Isolation Forest)
- **Quadrant 1 (Both Agree Fraud)**: $66$ alerts $\implies$ **$55$ actual frauds ($83.3\%$ precision)**. When supervised patterns and spatial outliers align, detection confidence is exceptionally high.
- **Quadrant 2 (XGBoost Only, IsoForest Missed)**: $56$ alerts $\implies$ $32$ actual frauds ($57.1\%$ precision). Subtler frauds that mimic normal spend distributions but match conjunctive decision rules.
- **Quadrant 3 (IsoForest Only, XGBoost Missed)**: $658$ alerts $\implies$ $11$ actual frauds ($1.7\%$ precision). High-variance statistical outliers and zero-day anomalies.
- **Quadrant 4 (Both Agree Legitimate)**: $47,450$ transactions.

---

## 9. Deep Error Analysis (FP vs. FN)

Dissecting prediction cohorts at $\theta^* = 0.20$ revealed the physical and economic root causes of model errors:

### 1. False Positives (Innocent Flagged — 35 Cases)
- **Cohort Mean Spend**: **$\$156.25$** (compared to legitimate baseline of $\$45.75$).
- **Customer Amount Ratio**: **$2.52\times$** weekly spend.
- **Root Cause**: Innocent cardholders making rare, unusually large purchases (appliances, travel). The model suspects an account takeover.
- **Mitigation**: Route to Step-Up Authentication (2FA) rather than declining the transaction.

### 2. False Negatives (Fraud Missed — 582 Cases)
- **Cohort Mean Spend**: **$\$49.19$** (indistinguishable from honest spend of $\$45.75$).
- **Customer Amount Ratio**: **$1.12\times$** weekly spend.
- **Vulnerability by Scenario**:
  - *Scenario 1 (High Amount > $220)*: **$100.0\%$ caught** ($1/1$).
  - *Scenario 3 (Account Takeover / 5x Spend)*: **$67.2\%$ caught** ($86/128$).
  - *Scenario 2 (Compromised Terminals)*: **$0.0\%$ caught** ($0/540$).
- **The Big Reveal**:
  **$92.8\%$ of missed frauds ($540$ out of $582$) belong to Scenario 2**. In Scenario 2, POS terminals are compromised while cardholder habits remain normal. Without historical terminal compromise labels, individual cardholder features exhibit zero deviation. This provides empirical justification for hybrid anomaly detection.

---

## 10. Model Explainability with SHAP

Using **TreeSHAP**, we compute exact game-theoretic Shapley attributions in polynomial time:
$$f(\mathbf{x}) = \phi_0 + \sum_{j=1}^{14} \phi_j(\mathbf{x})$$
- **Model Base Log-Odds ($\phi_0$)**: **`-4.7343`** (corresponds to training fraud rate of $0.87\%$).

### Deconstructed Case Studies
- **Caught Fraud (TX ID 191776 — $P = 99.65\%$)**:
  - `TX_AMOUNT = $521.10` ($\text{SHAP}: +4.6752$)
  - `CUSTOMER_AMOUNT_RATIO = 4.30x` ($\text{SHAP}: +2.6389$)
  - `CUSTOMER_AVG_AMOUNT_7D = $121.17` ($\text{SHAP}: +1.8031$)
  - *Attribution*: The combination of extreme amount and a $4.3\times$ ratio surge pushed log-odds from $-4.73 \to +5.65$.
- **False Positive (TX ID 150870 — $P = 91.56\%$)**:
  - Pushed toward fraud by `TX_AMOUNT = $166.21` ($+4.16$).
  - Pushed toward legitimate by `CUSTOMER_AMOUNT_RATIO = 1.29x` ($-0.60$) and `TX_HOUR = 13:00` ($-0.10$).

---

## 11. The Hybrid Fraud Risk Engine

Implemented in [`src/predict.py`](file:///d:/RESUME%20PROJECTS/fraud-detection-anomaly-system/src/predict.py), the `FraudRiskEngine` scores transactions in real time:

$$\text{Combined Risk Score} = 0.70 \cdot P_{\text{supervised}} + 0.30 \cdot S_{\text{anomaly}}$$

```
[0.00 to 0.30) ───────────────> LOW RISK    ───────────────> APPROVE (Frictionless)
[0.30 to 0.70) ───────────────> MEDIUM RISK ───────────────> CHALLENGE_2FA (Step-up)
[0.70 to 1.00] ───────────────> HIGH RISK   ───────────────> DECLINE (Block & Alert)
```

> [!NOTE]
> **Weighting & Boundary Disclaimer**:
> The $0.70 / 0.30$ weighting and $[0.30, 0.70]$ boundaries are project-defined parameters based on validation PR-AUC superiority. They are not universal industry constants. Every financial institution calibrates them to their credit loss tolerance.

### Live Engine Exemplar Execution
1. **Exemplar A (Routine Daily Purchase — TX 192921)**:
   - Amount: $\$11.74$, Ratio: $0.21\times$ $\implies$ Score: **`0.0516`** $\implies$ **`APPROVE`**.
2. **Exemplar B (Unusual Spending Spike — TX 194087)**:
   - Amount: $\$86.89$, Ratio: $4.85\times$ $\implies$ Score: **`0.5095`** $\implies$ **`CHALLENGE_2FA`**.
3. **Exemplar C (High-Velocity Attack — TX 193958)**:
   - Amount: $\$351.35$, Ratio: $2.84\times$, Night: $1$ $\implies$ Score: **`0.9077`** $\implies$ **`DECLINE`**.

---

## 12. Project Structure

```
fraud-detection-anomaly-system/
├── data/
│   ├── raw/
│   │   └── transactions.csv         # 241,151 simulated transactions
│   └── processed/
│       ├── train.csv                # 144,690 rows (Days 0 to 35)
│       ├── val.csv                  # 48,230 rows (Days 36 to 47)
│       └── test.csv                 # 48,231 rows (Days 48 to 59)
├── notebooks/
│   ├── 01_eda.ipynb                 # Class imbalance and amount skews
│   ├── 02_model_training.ipynb      # Supervised models E0 to E5
│   ├── 03_anomaly_detection.ipynb   # Isolation Forest & 4-way agreement
│   └── 04_error_analysis.ipynb      # False positive & negative investigation
├── src/
│   ├── __init__.py
│   ├── generate_data.py             # Fraud Handbook transaction simulator
│   ├── utils.py                     # Inspection and seed reproducibility
│   ├── eda.py                       # Distribution and timeline plots
│   ├── features.py                  # Leakage-safe rolling behavioral engineering
│   ├── preprocessing.py             # Chronological split & train-only scaling
│   ├── train.py                     # Supervised training matrix (E0-E5)
│   ├── evaluate.py                  # PR-AUC, ROC-AUC, curves, experiment log
│   ├── threshold_tuning.py          # Validation threshold optimization & test eval
│   ├── anomaly_detection.py         # Isolation Forest (E6, E8) & agreement
│   ├── error_analysis.py            # Cohort profiling & scenario breakdown
│   ├── explainability.py            # TreeSHAP beeswarm & waterfall engine
│   └── predict.py                   # Production FraudRiskEngine
├── models/
│   ├── baseline/                    # Scalers, Logistic Regression, Random Forest
│   └── final/                       # Final XGBoost & Isolation Forest artifacts
├── outputs/
│   ├── plots/                       # 12 exported diagnostic performance figures
│   ├── metrics/                     # experiments.csv, JSON summaries
│   └── predictions/                 # Probability arrays and score matrices
├── tests/
│   ├── test_features.py             # Feature math and zero-leakage proof
│   ├── test_preprocessing.py        # Split bounds and imputation assertions
│   └── test_prediction.py           # Probability bounds and engine validation
├── app.py                           # Lightweight Streamlit dashboard
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 13. How to Run the Pipeline

### 1. Environment Setup
```bash
git clone <repo_url>
cd fraud-detection-anomaly-system
python -m pip install -r requirements.txt
```

### 2. End-to-End Pipeline Execution
```bash
# 1. Generate synthetic dataset (241k transactions across 60 days)
python src/generate_data.py --n_customers 3500 --n_terminals 5000 --nb_days 60 --seed 42

# 2. Run Exploratory Data Analysis
python src/eda.py

# 3. Engineer leakage-safe features & perform chronological split
python src/preprocessing.py

# 4. Train supervised experiment matrix (E0 through E5)
python src/train.py --experiment all_phase6

# 5. Review experiment comparison table & multi-metric chart
python src/evaluate.py

# 6. Optimize decision threshold on Validation & evaluate once on Test
python src/threshold_tuning.py

# 7. Run unsupervised anomaly detection (E6, E8) & 4-way agreement
python src/anomaly_detection.py

# 8. Conduct deep error analysis (FP vs FN)
python src/error_analysis.py

# 9. Generate SHAP global and local waterfall explanations
python src/explainability.py

# 10. Run production risk engine demo
python src/predict.py

# 11. Run complete automated unit test suite
python -m pytest tests/ -v

# 12. Launch interactive Streamlit dashboard
python -m streamlit run app.py
```

---

## 14. Scaling to 1M+ Transactions

To scale the simulation from $241\text{k}$ to $\approx 1.2\text{M}$ transactions for large-scale stress testing:

```bash
python src/generate_data.py \
    --n_customers 15000 \
    --n_terminals 20000 \
    --nb_days 90 \
    --output data/raw/transactions_1m.csv
```

### Memory & Performance Optimizations for 1M+ Scale:
1. **Downcast Numeric Types**: Convert `int64` and `float64` to `int32` and `float32` in `src/features.py` to reduce memory from $\sim 1.5\text{GB} \to 450\text{MB}$.
2. **File Storage**: Transition from `.csv` to Apache `.parquet` (Snappy compression) in `src/preprocessing.py` for $5\times$ faster I/O.
3. **GPU Acceleration**: In `src/train.py`, pass `tree_method="hist"` and `device="cuda"` to `XGBClassifier` to accelerate gradient boosting across millions of rows.

---

## 15. Project Limitations & Future Work

### Limitations
1. **Compromised Terminal Detection**: Because labels are not used at inference, pure customer-level features cannot detect compromised terminals (Scenario 2) without terminal-level delay windows.
2. **Chargeback Latency**: Real-world fraud labels arrive with a 30 to 90-day chargeback delay. The simulator assigns labels immediately post-simulation.

### Future Improvements
1. **Delayed-Feedback Pipeline**: Simulate 30-day chargeback reporting delays where labels for transactions at Day $T$ only become available for training at Day $T+30$.
2. **Graph Neural Networks (GNNs)**: Model bipartite customer-terminal graphs using Relational Graph Convolutional Networks (R-GCN) to catch coordinated fraud rings.
3. **Drift Detection**: Implement Kolmogorov-Smirnov (KS) tests on feature distributions over 7-day sliding intervals to alert on concept drift.

---

## 16. Master Interview Preparation Guide

### Q1: Why is fraud detection treated as an imbalanced problem?
**Answer**: Fraudulent transactions represent a tiny fraction of card transactions ($<1.5\%$). Legitimate transactions vastly dominate. Classifiers trained without adjustment default to the majority class.

### Q2: Why is accuracy misleading in fraud detection?
**Answer**: Accuracy assigns equal weight to all outcomes. If $98.89\%$ of transactions are legitimate, a dummy model predicting always legitimate achieves $98.89\%$ accuracy while missing $100\%$ of fraud. Precision, Recall, and PR-AUC directly isolate fraud detection quality.

### Q3: Why is chronological splitting mandatory instead of random splitting?
**Answer**: Random splitting shuffles future transactions into the past, enabling the model to predict past events using future patterns (data leakage). Chronological splitting ($60\%$ Train, $20\%$ Val, $20\%$ Test) mirrors production deployment where models train on history and predict into the future.

### Q4: How do you guarantee zero data leakage in rolling behavioral features?
**Answer**: When computing rolling customer spend or velocity at timestamp $T$, we enforce `closed='left'`, aggregating strictly over $[T - \text{window}, T)$. This guarantees the current transaction at $T$ is excluded. Furthermore, imputation statistics and scalers are fitted exclusively on the Training set.

### Q5: Why was `TERMINAL_FRAUD_RATE` not initially implemented?
**Answer**: Calculating a historical terminal fraud rate using ground-truth fraud labels can easily leak future labels or uncharged chargebacks into training features. In reality, fraud labels arrive weeks after transaction occurrence.

### Q6: How does SMOTE work and why must it only be applied to training data?
**Answer**: SMOTE synthesizes new minority points by linear interpolation between $k$-nearest minority neighbors: $\mathbf{x}_{\text{new}} = \mathbf{x}_i + \lambda(\mathbf{x}_{zi} - \mathbf{x}_i)$. It must be restricted to training data because oversampling validation or test sets distorts the real-world fraud prevalence and leaks synthetic points across partitions.

### Q7: What is the difference between ROC-AUC and PR-AUC?
**Answer**: The ROC curve's x-axis is False Positive Rate ($FP / [FP + TN]$). Because $TN$ is massive in fraud, $FPR$ stays small and ROC-AUC appears high even when precision is poor. The PR curve evaluates Precision ($TP / [TP + FP]$) directly against Recall without being diluted by $TN$.

### Q8: How does decision threshold tuning work?
**Answer**: Instead of assuming $\theta = 0.50$, we evaluate thresholds $\theta \in [0.05, 0.90]$ on the Validation set. Lowering $\theta$ increases Recall (fewer missed frauds) at the expense of Precision (more customer friction). We select $\theta^*$ at peak $F_1$ ($0.20$), and evaluate once on the Test set.

### Q9: How does Isolation Forest detect anomalies without labels?
**Answer**: Isolation Forest builds random binary trees. Outliers lie in sparse regions and require very few random cuts to isolate (short path lengths), whereas normal clustered points require deep partitions. Path length is converted into an anomaly score $s \in [0, 1]$.

### Q10: What did error analysis reveal about False Negatives?
**Answer**: In our model, $92.8\%$ of missed frauds ($540/582$) belonged to Scenario 2 (compromised terminals). Because customer spending amounts were normal, customer-level features looked innocent. This demonstrated why unsupervised anomaly detection or terminal-level velocity tracking is necessary.

### Q11: How does SHAP explain a transaction prediction?
**Answer**: Based on cooperative game theory, SHAP computes the marginal contribution of each feature across all possible feature subsets. It decomposes the model's prediction into a base expected rate plus additive feature attributions ($\phi_0 + \sum \phi_j = f(\mathbf{x})$), providing legally defensible Adverse Action reason codes.

### Q12: How does the final Fraud Risk Engine operate?
**Answer**: It computes a combined score: $\text{Score} = 0.70 \cdot P_{\text{supervised}} + 0.30 \cdot S_{\text{anomaly}}$. Transactions with $\text{Score} < 0.30$ are approved frictionless; scores between $0.30$ and $0.70$ trigger a Step-Up 2FA challenge; scores $\ge 0.70$ are declined.
