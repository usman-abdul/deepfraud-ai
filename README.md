# DeepFraud AI — Real-Time Fraud Detection for Financial Transactions

A machine learning system that detects suspicious financial transactions in real time
using anomaly detection. Upload a dataset, get instant fraud scores, severity ratings,
and plain-language explanations — all through an interactive dashboard.

---

## Why This Matters

Digital payment fraud is accelerating across Africa. In Q2 2024, Nigeria alone recorded
over **₦42.6 billion** in fraud losses, with mobile transactions accounting for **33.4%**
of all reported cases.

Traditional rule-based systems fail to catch novel fraud patterns. DeepFraud AI uses
unsupervised machine learning — meaning it learns what "normal" looks like and flags
anything that deviates, without needing labeled fraud examples to train on.

---

## Problem

Fraud detection in financial systems faces two core challenges:

1. **Imbalance** — fraudulent transactions are rare, making supervised models hard to train
2. **Novelty** — fraud patterns evolve constantly, making fixed rules obsolete quickly

DeepFraud AI addresses both by modeling normal transaction behavior and treating
significant deviations as potential fraud.

---

## Approach

The system uses **Isolation Forest**, an unsupervised anomaly detection algorithm
well-suited for fraud detection in high-dimensional transaction data.

Rather than learning what fraud looks like, Isolation Forest learns what *normal* looks like.
It builds an ensemble of random decision trees and measures how quickly each transaction
can be isolated from the rest. Fraudulent transactions — being rare and statistically
unusual — are isolated faster, resulting in a higher anomaly score.

Each transaction is evaluated across **28 PCA-transformed behavioral features** (V1–V28)
plus transaction amount, producing:

- An **anomaly score** (0–1) — higher means more suspicious
- A **flag** — Normal or Suspicious
- A **severity tier** — Low, Moderate, or Critical
- A **reason** — plain-language explanation of why it was flagged

---

## How It Works

```
1. Input       →  Upload a CSV of transactions (or use the included demo dataset)
2. Features    →  System extracts V1–V28 PCA features + log-scaled Amount
3. Model       →  Isolation Forest scores each transaction for anomaly likelihood
4. Scoring     →  Anomaly scores normalized to 0–1; threshold applied at 0.7
5. Severity    →  Transactions tiered as Low / Moderate / Critical
6. Dashboard   →  Results displayed with summary chart and per-transaction reasons
7. Export      →  Flagged transactions downloadable as CSV
```

The entire pipeline runs in seconds on a standard laptop.

---

## Features

- Upload any transaction CSV for instant analysis
- Real-time anomaly scoring across all transactions
- Severity classification: Low / Moderate / Critical
- Plain-language alert reason per flagged transaction
- Summary bar chart of Normal vs Suspicious breakdown
- One-click export of suspicious transactions to CSV
- Demo dataset included — works out of the box, no setup needed

---

## Tech Stack

| Layer | Tool |
|---|---|
| Dashboard | Streamlit |
| ML Model | Scikit-learn (Isolation Forest) |
| Data processing | Pandas, NumPy |
| Model persistence | Joblib |
| Dataset | Kaggle IEEE-CIS Credit Card Fraud Detection |

---

## Project Structure

```
DeepFraud-AI/
├── app/
│   └── app.py               # Streamlit dashboard (main entry point)
├── data/
│   └── demo/
│       ├── transactions_for_demo.csv
│       └── fraud_test_cases.csv
├── models/
│   └── isof_model.joblib    # Trained Isolation Forest model
├── notebooks/               # EDA and training notebooks (coming soon)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## How to Run

**1. Clone the repo**
```bash
git clone https://github.com/your-username/deepfraud-ai.git
cd deepfraud-ai
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Launch the dashboard**
```bash
streamlit run app/app.py
```

Open your browser, click **Run Fraud Detection**, and the demo dataset runs instantly.
To test your own data, upload a CSV with columns `V1–V28` and `Amount`.

---

## Dataset

Trained on the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
from Kaggle (IEEE-CIS). Features V1–V28 are PCA-transformed for privacy protection.
`Amount` is the raw transaction value in the original currency.

---

## Future Improvements

For a detailed roadmap toward a more advanced fraud detection system using Autoencoder and LSTM models, see [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md).

- [ ] **Autoencoder model** — reconstruction-error based detection for deeper anomaly signals
- [ ] **LSTM sequence model** — detect fraud across transaction sequences per user over time
- [ ] SHAP values for feature-level explainability on each alert
- [ ] Device and location signals as additional features
- [ ] REST API for live fintech integration
- [ ] User behavioral profiling and session-level scoring

---

## Team

Built by students at **Nile University of Nigeria** as part of an AI for Financial Inclusion initiative.

Roles: AI Engineering · Feature Engineering · Frontend Development · Data Engineering

---

## SDG Alignment

- **SDG 8** — Decent Work and Economic Growth (protecting digital financial ecosystems)
- **SDG 16** — Peace, Justice and Strong Institutions (reducing financial crime and fraud)
