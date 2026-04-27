# src/ml/lstm_trainer.py
#
# CURRENT ARCHITECTURE (matches what model.add() actually builds below):
# - Bidirectional(LSTM(64, return_sequences=True))  → Dropout(0.35)
# - LSTM(32)                                        → Dropout(0.35)
# - Dense(32, relu)                                 → Dropout(0.20)
# - Dense(1)
#
# (Earlier versions of this file documented a deeper BiLSTM(128)→BiLSTM(64)
# stack. That deeper variant was reverted because it overfitted on the small
# 100-athlete synthetic set — keeping the comment in sync with the code.)
#
# OTHER NOTES:
# - EarlyStopping patience 12, ReduceLROnPlateau on val_loss.
# - Dense(32) head sized for the ~50 engineered features (v4: workload-only,
#   no CV columns).
#
# LEAKAGE FIXES (retained):
# - Sort by [athlete_id, date], not just date.
# - Athlete-level train/val split (last 20% of athletes as val).
# - injury_risk_score and athlete_id dropped from features.
# - Per-athlete sequences (no cross-athlete boundary crossing).

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.optimizers import Adam

from src.ml.sequence_dataset import create_sequences


DATA_PATH    = Path("data/combined_dataset.csv")
MODEL_PATH   = Path("models/ml/lstm_sequence_model.keras")
META_PATH    = Path("models/ml/lstm_injury_model_meta.pkl")
HISTORY_PATH = Path("results/lstm_training_history.csv")

SEQUENCE_LENGTH  = 30
VAL_ATHLETE_FRAC = 0.20


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    sort_cols = ["athlete_id", "date"] if "date" in df.columns else ["athlete_id"]
    df = df.sort_values(sort_cols).reset_index(drop=True)
    if "date" in df.columns:
        df = df.drop(columns=["date"])
    return df


def athlete_train_val_split(df: pd.DataFrame, val_frac: float = 0.20):
    athletes  = sorted(df["athlete_id"].unique())
    n_val     = max(1, int(len(athletes) * val_frac))
    val_ids   = set(athletes[-n_val:])
    train_ids = set(athletes) - val_ids
    return (
        df[df["athlete_id"].isin(train_ids)].copy(),
        df[df["athlete_id"].isin(val_ids)].copy(),
    )


def train_lstm():
    print("Loading dataset...")
    df = load_dataset()

    feature_drop = [c for c in ["injury_risk_score_next", "injury_risk_score", "athlete_id"]
                    if c in df.columns]

    df_train, df_val = athlete_train_val_split(df, val_frac=VAL_ATHLETE_FRAC)

    print(f"Athletes — train: {df_train['athlete_id'].nunique()}, "
          f"val: {df_val['athlete_id'].nunique()}")

    feature_cols  = [c for c in df.columns if c not in feature_drop]
    train_X_cols  = [c for c in feature_cols if c != "injury_risk_score_next"]

    scaler = StandardScaler()
    df_train = df_train.copy()
    df_val   = df_val.copy()
    df_train[train_X_cols] = scaler.fit_transform(df_train[train_X_cols])
    df_val[train_X_cols]   = scaler.transform(df_val[train_X_cols])

    print("Creating sequences...")
    X_train, y_train = create_sequences(df_train, SEQUENCE_LENGTH, athlete_col="athlete_id")
    X_val,   y_val   = create_sequences(df_val,   SEQUENCE_LENGTH, athlete_col="athlete_id")

    print(f"Train: {X_train.shape}  |  Val: {X_val.shape}")
    n_features = X_train.shape[2]
    print(f"Features per timestep: {n_features}")

    print("Building model...")
    # Reduced architecture to prevent overfitting (was BiLSTM(128)->BiLSTM(64) = 371K params)
    # Now BiLSTM(64)->LSTM(32) ≈ 80K params — better suited for 80 train athletes
    model = Sequential([
        Bidirectional(
            LSTM(64, return_sequences=True),
            input_shape=(SEQUENCE_LENGTH, n_features),
        ),
        Dropout(0.35),
        LSTM(32, return_sequences=False),
        Dropout(0.35),
        Dense(32, activation="relu"),
        Dropout(0.20),
        Dense(1),
    ])

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )
    model.summary()

    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        EarlyStopping(patience=15, restore_best_weights=True, monitor="val_loss"),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4,
                          min_lr=1e-6, verbose=1),
        CSVLogger(str(HISTORY_PATH)),
    ]

    print("Training LSTM (max 150 epochs, early stop patience=15)...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=150,
        batch_size=64,
        callbacks=callbacks,
        verbose=1,
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)

    joblib.dump(
        {
            "scaler":           scaler,
            "features":         train_X_cols,
            "sequence_length":  SEQUENCE_LENGTH,
            "val_athlete_frac": VAL_ATHLETE_FRAC,
        },
        META_PATH,
    )
    print("\nLSTM saved to:", MODEL_PATH)
    print("History saved to:", HISTORY_PATH)


if __name__ == "__main__":
    train_lstm()
