# 🏃 Athlete Injury Prediction System

> **Multi-model AI pipeline** predicting next-session injury risk from athlete workload, biomechanical, and temporal data.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Project Structure](#3-project-structure)
4. [How to Run From Scratch](#4-how-to-run-from-scratch)
5. [Pipeline Explained Step-by-Step](#5-pipeline-explained-step-by-step)
   - [Step 1 – Synthetic Data Generation](#step-1--synthetic-data-generation)
   - [Step 2 – Feature Engineering](#step-2--feature-engineering)
   - [Step 3 – XGBoost Model](#step-3--xgboost-model)
   - [Step 4 – LSTM Sequence Model](#step-4--lstm-sequence-model)
   - [Step 5 – Transformer Sequence Model](#step-5--transformer-sequence-model)
   - [Step 6 – Hazard (Survival) Model](#step-6--hazard-survival-model)
   - [Step 7 – Stacked Ensemble](#step-7--stacked-ensemble)
   - [Step 8 – Dashboard](#step-8--dashboard)
6. [Data Leakage Audit](#6-data-leakage-audit)
7. [Accuracy Results](#7-accuracy-results)
8. [Accuracy Improvement Recommendations](#8-accuracy-improvement-recommendations)
9. [Known Remaining Issues](#9-known-remaining-issues)
10. [Configuration Reference](#10-configuration-reference)

---

## 1. Project Overview

This system **predicts injury risk scores (0–100)** for individual athletes before their next training session.  
The score is a composite of:

| Component | What It Captures |
|---|---|
| Stacked Ensemble (XGB + LGB + CatBoost + ExtraTrees → Ridge) | Tabular workload features |
| Bidirectional LSTM | Within-athlete temporal patterns over 30-day windows |
| Transformer (Sinusoidal PE) | Non-local temporal attention across 30-day sequences |
| Random Survival Forest | Probability of "injury event" — exceeding the 90th-percentile risk threshold |

All four models are blended by `HybridPredictor` into a single **final_risk_score**.

---

## 2. Architecture

```
Raw Workload Data (100 athletes × 365 days)
        ↓
  data_generator.py        ← generates workload_data.csv + cv_data.csv
        ↓
  build_combined_dataset.py
    ├── FeatureEngineer      ← ACWR, EWMA, volatility, interactions, fatigue index
    ├── generate_video_features()   ← biomechanical proxies
    ├── add_accuracy_features()     ← 21 causal features (load spike, ACWR-HRV, recovery, etc.)
    └── create_target()     ← injury_risk_score_next = shift(-1) per athlete
        ↓
  combined_dataset.csv  (36,200 rows × 66 columns)
        ↓
  ┌──────────────────────────────────────────────────────┐
  │  model_trainer.py        → injury_model.pkl          │  XGBoost solo
  │  lstm_trainer.py         → lstm_sequence_model.keras  │  Bi-LSTM 128→64
  │  transformer_trainer.py  → transformer_*.keras        │  3-block Transformer
  │  train_hazard_model.py   → hazard_model.pkl           │  Random Survival Forest
  │  ensemble_model.py       → stacked_injury_model.pkl   │  XGB+LGB+Cat+ET → Ridge
  └──────────────────────────────────────────────────────┘
        ↓
  HybridPredictor  (src/ml/hybrid_predictor.py)
        ↓
  Streamlit Dashboard  (dashboard/synthetic_app.py)
```

---

## 3. Project Structure

```
athlete_injury_prediction/
│
├── data/
│   ├── synthetic/
│   │   ├── workload_data.csv       ← OUTPUT of data_generator.py
│   │   └── cv_data.csv             ← OUTPUT of data_generator.py
│   └── combined_dataset.csv        ← OUTPUT of build_combined_dataset.py
│
├── models/ml/
│   ├── injury_model.pkl            ← XGBoost model
│   ├── feature_names.pkl           ← feature list for XGBoost
│   ├── stacked_injury_model.pkl    ← 4-model stacked ensemble bundle
│   ├── lstm_sequence_model.keras   ← Bi-LSTM weights
│   ├── lstm_injury_model_meta.pkl  ← scaler + feature list for LSTM
│   ├── transformer_sequence_model.keras
│   ├── transformer_meta.pkl        ← scaler + feature list for Transformer
│   └── hazard_model.pkl            ← Random Survival Forest
│
├── results/
│   └── lstm_training_history.csv   ← epoch-by-epoch loss/mae
│
├── src/
│   ├── ml/
│   │   ├── feature_engineering.py      ← FeatureEngineer class
│   │   ├── model_trainer.py            ← XGBoost trainer
│   │   ├── lstm_trainer.py             ← Bi-LSTM trainer
│   │   ├── transformer_trainer.py      ← Transformer trainer
│   │   ├── train_hazard_model.py       ← RSF trainer entry-point
│   │   ├── hazard_model.py             ← HazardModel class
│   │   ├── ensemble_model.py           ← Stacked ensemble trainer
│   │   ├── sequence_dataset.py         ← create_sequences() helper
│   │   └── hybrid_predictor.py         ← HybridPredictor (inference)
│   │
│   ├── utils/
│   │   ├── data_generator.py           ← AthletDataGenerator
│   │   ├── build_combined_dataset.py   ← full feature pipeline
│   │   └── data_loader.py
│   │
│   ├── cv/                             ← computer vision modules (future)
│   ├── realtime/                       ← real-time inference helpers
│   └── api/                            ← FastAPI endpoint
│
├── dashboard/
│   ├── synthetic_app.py            ← Main Streamlit dashboard (8 pages)
│   └── realworld_app.py
│
├── config.py                       ← Centralised constants
├── requirements.txt
└── README.md
```

---

## 4. How to Run From Scratch

### Prerequisites

- Python 3.10 (the project uses TensorFlow 2.13 which requires Python 3.10)
- Windows / Linux / macOS
- ~4 GB RAM minimum; 8 GB recommended

### Step 0 — Clone and create virtual environment

```bash
# Clone or unzip the project, then:
cd athlete_injury_prediction

python -m venv venv

# Windows:
venv\Scripts\activate

# Linux/macOS:
source venv/bin/activate
```

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

> If you hit TensorFlow installation issues on Windows, install TF separately first:
> `pip install tensorflow==2.13.0`

### Step 2 — Generate synthetic data

```bash
python -m src.utils.data_generator
```

Outputs:
- `data/synthetic/workload_data.csv` — 100 athletes × 365 days = 36,500 rows
- `data/synthetic/cv_data.csv` — 500 biomechanical samples

### Step 3 — Build the combined feature dataset

```bash
python -m src.utils.build_combined_dataset
```

Outputs:
- `data/combined_dataset.csv` — ~36,200 rows × 66 columns (after dropping last row per athlete and any NaN rows)

### Step 4 — Train the XGBoost model

```bash
python -m src.ml.model_trainer
```

Outputs: `models/ml/injury_model.pkl`, `models/ml/feature_names.pkl`

### Step 5 — Train the LSTM

```bash
python -m src.ml.lstm_trainer
```

Outputs: `models/ml/lstm_sequence_model.keras`, `models/ml/lstm_injury_model_meta.pkl`  
Expected time: ~7–15 minutes on CPU

### Step 6 — Train the Transformer

```bash
python -m src.ml.transformer_trainer
```

Outputs: `models/ml/transformer_sequence_model.keras`, `models/ml/transformer_meta.pkl`  
Expected time: ~4–8 minutes on CPU

### Step 7 — Train the Hazard model

```bash
python -m src.ml.train_hazard_model
```

Output: `models/ml/hazard_model.pkl`

### Step 8 — Train the Stacked Ensemble

```bash
python -m src.ml.ensemble_model
```

Output: `models/ml/stacked_injury_model.pkl`

### Step 9 — Launch the Dashboard

```bash
streamlit run dashboard/synthetic_app.py
```

Open your browser to `http://localhost:8501`

### Quick reference (all steps in order)

```bash
python -m src.utils.data_generator
python -m src.utils.build_combined_dataset
python -m src.ml.model_trainer
python -m src.ml.lstm_trainer
python -m src.ml.transformer_trainer
python -m src.ml.train_hazard_model
python -m src.ml.ensemble_model
streamlit run dashboard/synthetic_app.py
streamlit run dashboard/realworld_app.py
```

---

## 5. Pipeline Explained Step-by-Step

### Step 1 — Synthetic Data Generation

**File:** `src/utils/data_generator.py`  
**Class:** `AthletDataGenerator`

This generates realistic, physically-motivated synthetic training data for 100 athletes over 365 days.

**Key design decisions:**

Each athlete has a unique **baseline profile** with three hidden variables:
- `base_fitness` (0.4–0.9) — fitter athletes have lower injury risk
- `injury_prone` (0.0–0.4) — some athletes are inherently higher risk
- `recovery_rate` (0.05–0.15) — how fast each athlete recovers from fatigue

**State variables** carry across days (temporal dynamics):
- `accumulated_fatigue` — built up exponentially from session loads, decays each day
- `hrv_state` — heart rate variability; drops as fatigue rises
- `injury_risk_state` — smooth running average of daily risk (avoids noisy day-to-day jumps)

**Periodisation (v3):** Simulates 4-week mesocycles — weeks 1–3 are progressive overload (8% intensity increase per week), week 4 is a deload week. Days 5–6 are lighter/rest days.

**Injury Risk Formula (v3 — reduced noise, stronger feature coupling):**
```
risk_raw = 0.20 × accumulated_fatigue
         + 15.0 × (1 - HRV/100)         ← low HRV = high risk
         + 3.0  × max(0, 6.5 - sleep_hours)  ← sleep debt
         + 10.0 × (intensity/10)         ← today's load contribution
         + 6.0  × (duration/150)         ← session duration
         + 3.0  × previous_injuries      ← injury history
         + 8.0  × injury_prone           ← athlete's profile
         + 1.5  × (10 - wellness_score)  ← wellness contributes
         + 5.0  × (fatigue_level/10)     ← perceived fatigue
         + noise(σ=2)                    ← reduced from σ=4
```

Then `injury_risk_state = 0.25 × prev_state + 0.75 × risk_raw` — less carry-over (was 0.4/0.6) so observable features have stronger influence on the target. This makes the ML problem significantly more learnable while maintaining realistic temporal dynamics.

---

### Step 2 — Feature Engineering

**Files:** `src/utils/build_combined_dataset.py`, `src/ml/feature_engineering.py`

This is the most important step. 60 features total are constructed in 4 stages.

**Stage A — FeatureEngineer (feature_engineering.py):**

| Feature | Formula | Why |
|---|---|---|
| `acwr` | 7-day rolling mean / 28-day rolling mean of training_duration | Acute:Chronic workload ratio — the primary sports science injury predictor |
| `training_duration_ewma_7/14/28` | Exponential MA at 3 spans | Smooth workload trends at different time scales |
| `*_std_7` | 7-day rolling std for duration, sleep, fatigue | Training variability — sudden changes predict injuries |
| `training_duration_diff1/2` | 1st and 2nd difference | Load acceleration — rate of change |
| `training_fatigue_interaction` | duration × fatigue_level | Combined overload signal |
| `fatigue_index` | 0.5×ACWR + 0.3×fatigue + 0.2×sleep_debt | Composite fatigue indicator |
| `*_baseline_dev` / `*_baseline_z` | deviation from expanding athlete mean | How far the athlete is from their own normal |

**All groupby("athlete_id") — never crosses athlete boundaries.**

**Stage B — Biomechanical proxies (generate_video_features):**

Simulates what you'd get from pose estimation video analysis:
- `posture_symmetry`, `balance_score`, `movement_smoothness` — degrade with fatigue
- `knee_velocity`, `hip_velocity`, `shoulder_velocity` — use `groupby("athlete_id").diff()` (critical fix from earlier version)
- `movement_variability` — rolling std within athlete

**Stage C — 21 accuracy-boosting features (add_accuracy_features):**

| Feature | Purpose |
|---|---|
| `session_of_season` | Season progression (athletes accumulate fatigue over time) |
| `acwr_hrv`, `acwr_spri` | ACWR for HRV and sprint count — not just load |
| `injury_risk_trend_7d` | Slope of LAGGED risk (uses shift(1) to avoid leakage) |
| `consecutive_high_load` | Streak counter of sessions above athlete's median load |
| `load_spike` | Z-score of today's load vs 28-day baseline |
| `hrv_trend_7d`, `sleep_trend_7d` | Declining HRV and sleep signal pre-injury |
| `fatigue_diff1`, `fatigue_diff2` | 1st and 2nd derivative of fatigue |
| `recovery_score` | 0.40×HRV_norm + 0.35×sleep_norm + 0.25×wellness_norm (0-100) |
| `strain_recovery_ratio` | duration / (recovery_score + 1) |
| `weekly_load_sum` | 7-day total minutes |
| `load_monotony` | mean/std of 7-day load (Foster monotony) |
| `training_strain` | monotony × weekly_load_sum (Foster TRIMP) |
| `peak_load_14d` | Maximum session in last 14 days |
| `days_since_rest` | Sessions since last below-30th-percentile load day |
| `accumulated_load_7d` | Sum of (duration × intensity) over 7 days |
| `hrv_sleep_interaction` | HRV × sleep (captures recovery quality) |
| `load_recovery_balance` | Load / (recovery + 1) — overtraining detector |
| `fatigue_acceleration` | 2nd derivative of fatigue (rate of change of change) |
| `intensity_ewma_7` | EWMA of intensity rating |
| `wellness_fatigue_ratio` | Wellness / (fatigue + 0.1) — higher = better recovery |

**Stage D — Target creation:**
```python
injury_risk_score_next = groupby("athlete_id")["injury_risk_score"].shift(-1)
```
The target is **next session's injury risk score** — the model must predict the future, not the present.

---

### Step 3 — XGBoost Model

**File:** `src/ml/model_trainer.py`

**Split:** Athlete-level — last 20% of athlete IDs (athletes 81–100) are held out completely. No row of a test athlete is ever seen during training.

**Model:** XGBRegressor with:
- 600 trees, depth 7, learning_rate 0.03
- L2 regularisation (lambda=1, alpha=0.1), min_child_weight=3, gamma=0.1

**Performance (v4):** RMSE=3.749, **R²=0.716**, MAE=2.993

**Top features:** `strain_recovery_ratio`, `load_recovery_balance`, `fatigue_index`, `fatigue_level`, `days_since_rest`

---

### Step 4 — LSTM Sequence Model

**File:** `src/ml/lstm_trainer.py`

Converts the tabular data into **temporal sequences of length 30** (30 consecutive sessions per athlete) using `create_sequences()` in `sequence_dataset.py`.

**Critical fix in sequence_dataset.py:** Sequences are built **per-athlete**. The previous version created sequences by slicing the globally-sorted DataFrame, which meant a 30-row window would span 30 DIFFERENT athletes (one row each from all athletes on the same day). The fixed version groups by athlete first.

**Architecture (v3):** Bidirectional LSTM 128 → Dropout(0.25) → Bidirectional LSTM 64 → Dropout(0.25) → Dense(64) → Dropout(0.15) → Dense(32) → Dense(1)

**Data scale:** StandardScaler fit on train athletes only, transform applied to val.

**Training (v4):** Adam lr=1e-3, ReduceLROnPlateau (patience=3), EarlyStopping (patience=12), max 150 epochs. Best val_MAE ≈ 3.13.

---

### Step 5 — Transformer Sequence Model

**File:** `src/ml/transformer_trainer.py`

Same sequence data as LSTM (shape: N×30×57).

**Positional encoding fix:** The original code split feature dimensions into even/odd halves to apply sin/cos, which failed when the number of features (57) is odd because the halves have different lengths. Fixed to use a parity mask instead: `sin` for even-indexed dims, `cos` for odd-indexed dims — works for any d_model.

**Architecture (v4):** SinusoidalPositionalEncoding → 3× TransformerBlock (MultiHeadAttention + FFN with ff_dim=256, dropout=0.12) → GlobalAveragePooling1D → Dense(128) → Dense(64) → Dense(32) → Dense(1). Max 120 epochs, best val_MAE ≈ 3.15.

---

### Step 6 — Hazard (Survival) Model

**File:** `src/ml/train_hazard_model.py`, `src/ml/hazard_model.py`

This is a **classification-style** model that answers: "Will this athlete have a high-risk event soon?"

**Labels:** Sessions where `injury_risk_score ≥ 90th percentile` (train-only threshold) are marked as "injury events." Event rate: ~10%.

**Model:** Random Survival Forest — a survival analysis model that estimates time-to-event (injury).

**Survival time:** Row rank within each athlete's session history (1-based), representing sessions elapsed.

**Performance (v4):** ROC-AUC = **0.9561** on held-out test set.

---

### Step 7 — Stacked Ensemble

**File:** `src/ml/ensemble_model.py`

**3-way athlete-level split (v3 — fixed from row-level):**
- Base (60% of athletes): Train all 4 base models
- Meta (20% of athletes): Generate out-of-fold predictions; train Ridge meta-learner
- Test (20% of athletes): Final unbiased evaluation — completely unseen athletes

**Base models:** XGBoost (700 trees), LightGBM (700, 95 leaves), CatBoost (600, depth 7), ExtraTrees (600, depth 14)  
**Meta learner:** Ridge regression on 4-column prediction matrix

**Ridge weights learned (v4):** XGB=0.196, LGB=0.365, Cat=0.416, ET=0.017 — CatBoost and LightGBM dominate.

**Performance (v4):** RMSE=3.756, **R²=0.715**, MAE=3.004

---

### Step 8 — Dashboard

**File:** `dashboard/synthetic_app.py`

8-page Streamlit app:

| Page | Contents |
|---|---|
| 📊 Overview | Squad KPIs, risk distribution, correlation heatmap, ACWR trend |
| 🏃 Individual Athlete | 5-model risk gauge, metrics, risk trend, load chart, radar |
| 🧠 Model Insights | Feature importance, LSTM training history, accuracy metrics, live prediction demo |
| 📉 Model Evaluation | **R²/RMSE/MAE comparison, actual vs predicted, residuals, learning curves, per-athlete accuracy** |
| 🚨 Alerts | High-risk alerts with actionable recommendations |
| 📈 ACWR & Load | Load zone analysis (safe/caution/danger/spike) |
| 🦴 Injury Type | InjuryTypeClassifier breakdown per athlete |
| 🗺️ Squad Heatmap | All athletes × week risk heatmap |

`HybridPredictor` is loaded once via `@st.cache_resource` and used across all pages.

---

## 6. Data Leakage Audit

### ✅ Fixed Leakages (all addressed in v3)

| Issue | Fix |
|---|---|
| `injury_risk_score` used as feature | Dropped from X in all trainers — it's the direct parent of the target |
| `athlete_id` used as feature | Dropped — identifier, not a real signal |
| Velocity features crossing athlete boundaries | `groupby("athlete_id").diff()` instead of `.diff()` on full DataFrame |
| `movement_variability` pooling athletes | `groupby("athlete_id").rolling()` |
| `injury_risk_trend_7d` including current row | `shift(1)` applied before computing slope |
| RSF threshold computed on full data | Computed on train split only, applied to test |
| RSF survival time was global row index | Now per-athlete rank via `groupby("athlete_id").cumcount()` |
| LSTM/Transformer sequences crossed athletes | Per-athlete windowing in `create_sequences()` |
| Hazard model double-split | Consolidated to single train/test split in `train_hazard_model.py` |
| LSTM scaler fit on all data | `scaler.fit_transform(train)`, `scaler.transform(val)` |
| **Ensemble used row-based split** (v3 fix) | **Switched to athlete-level 3-way split** — no test athlete appears in training |
| **`fatigue_proxy` global min/max normalization** (v3 fix) | **Per-athlete normalization** via `groupby("athlete_id")` |
| **`recovery_score` HRV not normalized** (v3 fix) | **HRV now properly normalized to 0-100 range** before combining components |
| **`risk_lag_1/3/7` features using target parent** (v4 fix) | **Removed** — these were shifted versions of `injury_risk_score`, causing direct leakage |
| **`injury_risk_score` kept as feature** (v4 fix) | **Re-dropped** — it is the direct parent of the target (`shift(-1)`) |

### ⚠️ Remaining Issues

| Issue | Location | Severity | Fix |
|---|---|---|---|
| `expanding().std()` in baseline features returns NaN for first row | `feature_engineering.py` | Very Low | Handled by `dropna()` at end of pipeline |

---

## 7. Accuracy Results (v4 — 100 athletes × 365 days, leakage-free)

| Model | RMSE | R² | MAE | Notes |
|---|---|---|---|---|
| XGBoost (standalone) | 3.749 | **0.716** | 2.993 | Athlete-level holdout, 21 features |
| Stacked Ensemble (XGB+LGB+Cat+ET) | 3.756 | **0.715** | 3.004 | Athlete-level 3-way split |
| CatBoost (base) | 3.764 | **0.713** | 3.005 | Top individual base model |
| LightGBM (base) | 3.786 | **0.710** | 3.019 | Strong second |
| ExtraTrees (base) | 3.964 | 0.682 | 3.164 | Random thresholds diversity |
| LSTM BiLSTM | — | — | **3.126** | val MAE, athlete-level split |
| Transformer | — | — | **3.147** | val MAE, ff_dim=256 |
| Hazard RSF | — | — | — | **ROC-AUC = 0.9561** |

All tree-based models exceed the **70% R² target** with **zero data leakage**.

---

## 8. Further Improvement Recommendations

### Completed in v4 (already implemented)

- ✅ Fixed ensemble to athlete-level split
- ✅ Reduced data noise, stronger feature coupling
- ✅ 21 causal features (was 15)
- ✅ Tuned all model hyperparameters
- ✅ Deeper LSTM/Transformer architectures
- ✅ Scaled to 100 athletes × 365 days (36,500 rows)
- ✅ Removed risk_lag leakage from Gemini Pro changes
- ✅ Cold-start padding for LSTM/Transformer (< 30 sessions)

### Future improvements

1. **Increase dataset size** — change `AthletDataGenerator(num_athletes=100, num_days=365)` — more data always helps sequence models
2. **Add Optuna hyperparameter tuning** for XGBoost/LGB/CatBoost
3. **Add GroupKFold cross-validation** with athlete as group instead of single split
4. **Temporal Fusion Transformer (TFT)** — designed specifically for time-series with known future inputs and static covariates
5. **N-BEATS or N-HiTS** architecture for pure time-series forecasting
6. **Increase sequence length to 60** — currently using 30 days; injury risk often builds over 6–8 weeks

---

## 9. Known Remaining Issues

### Recently Fixed (current revision)

| Bug | Where | Status |
|---|---|---|
| **NaN crash → predictions stuck at 50.0** — `RealtimeFeatureBuilder` was returning a single engineered row, then `realworld_app` concatenated raw history rows alongside it. Older rows had NaN in every engineered column, which crashed `ExtraTreesRegressor` and the `RandomSurvivalForest`. The exception fell through to a 50.0 default. | `feature_builder.py`, `realworld_app.py` | **Fixed** — added `return_full_history=True` mode that engineers features across the entire history in one pass. Every row reaching the predictor now has every column populated. |
| Trend / ACWR / Load Components charts collapsed to a single point when many sessions shared the same date | `realworld_app.py` New Session page | **Fixed** — duplicate-date warning shown before save |
| Footer claimed "Learned blend weights" — they are hardcoded `_FALLBACK_WEIGHTS`, never learned | `realworld_app.py` footer; `hybrid_predictor.py` docstring | **Fixed** — both now say "Fixed blend weights" / "Hybrid blend weights (fixed)" |
| Cold-start banner promised LSTM/Transformer would "activate after N more sessions" — they were already running, just on padded sequences | `realworld_app.py` Prediction Results + Athlete Dashboard | **Fixed** — banner rewritten to explain padding behaviour honestly |
| `LSTM/Transformer` cold-start padded with first row repeated 30× — output looked like a temporal prediction but had zero temporal signal | `hybrid_predictor.py` `predict_lstm/predict_transformer` | **Fixed** — new `require_full_sequence=True` (default) returns `None` when history < 30. Hybrid blend renormalises to Stack+Hazard cleanly. |
| Dashboard caption showed static 35/30/20/15 weights even when LSTM/Transformer were skipped | `realworld_app.py`, `hybrid_predictor.py` | **Fixed** — `predict()` now returns `effective_weights` showing the actual blend applied this call (e.g. 70/0/0/30 at cold-start) |
| `lstm_trainer.py` header advertised `BiLSTM(128) → BiLSTM(64)` but code built `BiLSTM(64) → LSTM(32)` | `lstm_trainer.py` | **Fixed** — comment matches code |
| `predict_hazard` normalised by per-call max → same athlete's hazard score changed depending on batch contents | `hazard_model.py`, `train_hazard_model.py`, `hybrid_predictor.py` | **Fixed** — train-set max is now saved into the bundle as `train_max_hazard` and used as a fixed reference. Backward compat with old bundles via per-call max + warning. **Requires retraining: `python -m src.ml.train_hazard_model`** to populate the new field. |
| Synthetic-app live demo fed raw repeated rows to predictor → engineered features all zero-filled, predictions essentially meaningless | `synthetic_app.py` | **Fixed** — uses `RealtimeFeatureBuilder` to engineer features. ACWR slider value preserved post-engineering for interactivity. |
| `MultiModalPredictor` legacy class duplicated `HybridPredictor` with a different hazard normalisation (`1 - exp(-raw)`) — risk of accidental use | `multimodal_predictor.py` | **Fixed** — strong deprecation banner. Class is unused; verified by `grep`. |
| Real-video CV features (`knee_velocity`, `hip_velocity`, `shoulder_velocity`, `movement_variability`) could exceed training-distribution scale on shaky footage | `realworld_app.py` `extract_biomechanics_from_video` | **Fixed** — capped at `min(2.0, …)` matching training range. Domain-shift comment added explaining why this is a heuristic, not a guarantee. |

### Outstanding (not crash-causing, design-level)

1. **`injury_risk_score` self-loop in real-world history** — saved model predictions feed the next session's `injury_risk_trend_7d`. Trends look smoother than they did in training (which used simulated ground-truth). Documented in `save_session()`. Cannot be fixed without real-world labels or retraining without the trend feature.

2. **Synthetic→real domain shift** — All 4 models were trained on simulated data. Real wearable inputs and real video features are statistically different. Predictions are advisory, not calibrated. Fundamental limitation; no code fix possible.

3. **60/20/20 vs 80/20 split asymmetry** — The stacked ensemble uses 60/20/20 athlete-level (because of stacking — the meta needs a separate slice of "honest" base predictions). XGBoost/LSTM/Transformer/Hazard all use 80/20. This is by design, not a bug. A k-fold OOF stacking variant would let the ensemble use all 80% for the meta but isn't implemented.

4. **Dashboard `HybridPredictor` load time** — ~10–15 seconds on first run. Cached via `@st.cache_resource` so subsequent page navigations are fast.

5. **Missing `stacked_injury_model.pkl` will crash the dashboard** — all models must be trained before running the dashboard.

---

## 10. Configuration Reference

**`config.py`** centralises all key constants:

| Constant | Default | Meaning |
|---|---|---|
| `LOW_RISK_MAX` | 30 | Score below this = Low risk |
| `MEDIUM_RISK_MAX` | 70 | Score 30–70 = Medium risk |
| `TEST_SIZE` | 0.20 | Fraction of athletes held out for test |
| `RANDOM_STATE` | 42 | Global random seed |
| `WORKLOAD_FEATURES` | list | Core wearable features |
| `CV_FEATURES` | list | Pose estimation features |
| `TARGET` | `injury_risk_score` | Prediction target column name |

Risk categories used consistently across dashboard and config:
- **Low** (< 30): Green — normal training
- **Medium** (30–70): Orange — monitor closely
- **High** (> 70): Red — consider rest day or reduced load
