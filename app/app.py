# app/app.py
# DeepFraud AI — Real-Time Fraud Detection Dashboard
# Built with Streamlit + Isolation Forest

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_PATH = "../models/isof_model.joblib"
DEMO_DATA_PATH = "../data/demo/transactions_for_demo.csv"

# ── Load model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# ── Page setup ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="DeepFraud AI", layout="wide")
st.title("💳 DeepFraud AI — Real-Time Fraud Detection")
st.markdown(
    "Upload a transaction dataset to analyze for potential fraud. "
    "The model flags **suspicious transactions** and assigns each an **anomaly score** (0–1)."
)

# ── Data upload ───────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload transactions CSV", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.success("✅ File uploaded successfully!")
else:
    st.info("No file uploaded — using demo dataset.")
    df = pd.read_csv(DEMO_DATA_PATH)

st.subheader("📋 Transaction Data (first 10 rows)")
st.dataframe(df.head(10))


# ── Feature engineering ───────────────────────────────────────────────────────
def featurize(df_in):
    df = df_in.copy()
    df.columns = df.columns.str.strip()

    # Ensure all V1–V28 PCA features and Amount exist
    for col in [f"V{i}" for i in range(1, 29)] + ["Amount"]:
        if col not in df.columns:
            df[col] = 0.0

    df["amount_log"] = np.log1p(df["Amount"])

    features = [f"V{i}" for i in range(1, 29)] + ["amount_log"]
    X = df[features].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return df, X_scaled


# ── Detection ─────────────────────────────────────────────────────────────────
def build_reason(row, mean_amount):
    parts = []
    if row["Amount"] > mean_amount * 5:
        parts.append("Unusually high amount")
    if row["anomaly_score"] > 0.7:
        parts.append("Anomalous pattern detected")
    return "; ".join(parts) if parts else "No unusual behavior"


if st.button("🚀 Run Fraud Detection"):
    df_processed, X = featurize(df)

    scores = -model.decision_function(X)
    preds = model.predict(X)

    df_processed["anomaly_score"] = np.round(
        (scores - scores.min()) / (scores.max() - scores.min() + 1e-6), 3
    )
    df_processed["flag"] = np.where(preds == -1, "Suspicious", "Normal")

    # Override: high anomaly score always flagged
    df_processed.loc[df_processed["anomaly_score"] > 0.7, "flag"] = "Suspicious"

    # Severity tier
    df_processed["severity"] = np.where(
        df_processed["anomaly_score"] > 0.9, "Critical",
        np.where(df_processed["flag"] == "Suspicious", "Moderate", "Low")
    )

    mean_amount = df_processed["Amount"].mean()
    df_processed["reason"] = df_processed.apply(
        lambda row: build_reason(row, mean_amount), axis=1
    )

    # ── Results ──────────────────────────────────────────────────────────────
    st.subheader("🧾 Detection Results")
    st.dataframe(df_processed[["Amount", "anomaly_score", "flag", "severity", "reason"]].head(20))

    st.write("### 📊 Summary")
    st.bar_chart(df_processed["flag"].value_counts())

    suspicious = df_processed[df_processed["flag"] == "Suspicious"]
    st.write(f"🔴 Suspicious Transactions Detected: **{len(suspicious)}**")
    if not suspicious.empty:
        st.dataframe(suspicious[["Amount", "anomaly_score", "severity", "reason"]].head(10))

    st.download_button(
        "📥 Download Suspicious Transactions (CSV)",
        suspicious.to_csv(index=False).encode("utf-8"),
        file_name="DeepFraudAI_Suspicious.csv"
    )
