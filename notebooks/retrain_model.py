"""
Retrains the Isolation Forest model with the correct contamination rate.
Run from the project root: python retrain_model.py
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

# ── Load data ────────────────────────────────────────────────────────────────
df = pd.read_csv("data/demo/transactions_for_demo.csv")
df.columns = df.columns.str.strip()

for col in [f"V{i}" for i in range(1, 29)] + ["Amount"]:
    if col not in df.columns:
        df[col] = 0.0

df["amount_log"] = np.log1p(df["Amount"])
features = [f"V{i}" for i in range(1, 29)] + ["amount_log"]
X = df[features].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── Retrain with correct contamination ───────────────────────────────────────
fraud_rate = round(df["Class"].sum() / len(df), 4)
print(f"Detected fraud rate: {fraud_rate} — using as contamination")

model = IsolationForest(
    n_estimators=200,
    contamination=fraud_rate,
    max_samples="auto",
    random_state=42
)
model.fit(X_scaled)

# ── Evaluate ─────────────────────────────────────────────────────────────────
scores = -model.decision_function(X_scaled)
preds = model.predict(X_scaled)
anomaly_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)
flags = np.where(preds == -1, 1, 0)
flags[anomaly_scores > 0.7] = 1

y_true = df["Class"].values

p = precision_score(y_true, flags)
r = recall_score(y_true, flags)
f1 = f1_score(y_true, flags)

print(f"Precision : {round(p, 3)}")
print(f"Recall    : {round(r, 3)}")
print(f"F1 Score  : {round(f1, 3)}")
print(f"Suspicious flagged: {flags.sum()} / {len(flags)}")

# ── Save ─────────────────────────────────────────────────────────────────────
joblib.dump(model, "models/isof_model.joblib")
print("Model saved to models/isof_model.joblib")
