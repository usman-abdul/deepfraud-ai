# DeepFraud AI — Original System Implementation Plan

## What We're Building

A two-model pipeline:
- **Autoencoder** — learns normal transaction patterns, flags high reconstruction error as fraud
- **LSTM** — analyzes sequences of transactions per user over time to catch behavioral anomalies

---

## Phase 1 — Data Preparation

- Use the IEEE-CIS / Kaggle Credit Card Fraud dataset already referenced
- Split into train (normal transactions only) and test sets
- Engineer sequence windows per user/time for LSTM input
- Normalize features (already partially done in current `featurize()`)

Files to create:
```
notebooks/01_eda.ipynb
notebooks/02_preprocessing.ipynb
data/processed/   ← scaled/sequenced data
```

---

## Phase 2 — Autoencoder Model

- Input: V1–V28 + amount_log (29 features)
- Architecture: Encoder compresses → Decoder reconstructs
- Train on normal transactions only
- Fraud = high reconstruction error (above learned threshold)
- Save model as `models/autoencoder_model.keras`

Files to create:
```
notebooks/03_autoencoder_training.ipynb
models/autoencoder_model.keras
```

---

## Phase 3 — LSTM Model

- Input: sliding window of N transactions per user (sequence)
- Architecture: LSTM layers → Dense → anomaly score
- Detects unusual behavioral sequences (not just single transactions)
- Save as `models/lstm_model.keras`

Files to create:
```
notebooks/04_lstm_training.ipynb
models/lstm_model.keras
```

---

## Phase 4 — Hybrid Scoring

- Combine Autoencoder score + LSTM score into a single fraud probability
- Simple weighted average or threshold voting
- Replaces the current `model.decision_function()` logic in `app.py`

Files to create:
```
app/scoring.py   ← hybrid score logic module
```

---

## Phase 5 — Dashboard Update

- Swap model loading in `app.py` to use new models
- Add reconstruction error visualization
- Show which model triggered the flag (Autoencoder / LSTM / Both)
- Keep all existing UI — just update the backend

---

## Phase 6 — Evaluation

- Target: F1 score > 0.7 (as stated in pitch)
- Report precision, recall, confusion matrix
- Compare against current Isolation Forest baseline

Files to create:
```
notebooks/05_evaluation.ipynb
```

---

## Estimated Timeline

| Phase | Task | Time |
|---|---|---|
| 1 | Data prep + sequencing | 1 day |
| 2 | Autoencoder build + train | 1–2 days |
| 3 | LSTM build + train | 1–2 days |
| 4 | Hybrid scoring logic | half a day |
| 5 | Dashboard wiring | half a day |
| 6 | Evaluation + tuning | 1 day |
| | **Total** | **~5–6 days** |

---

## New Dependencies to Add

```
tensorflow>=2.13.0   # or torch if you prefer PyTorch
```
