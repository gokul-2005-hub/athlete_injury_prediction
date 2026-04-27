# src/ml/transformer_trainer.py
#
# ARCHITECTURE IMPROVEMENTS (accuracy push):
# 1. ff_dim 64 → 128 per transformer block — larger feed-forward sublayer gives
#    more representation capacity (this was one of the GPT audit recommendations).
# 2. head_size 32 → 48 — slightly larger key/query dimension per attention head.
# 3. EarlyStopping patience 8 → 12 — same rationale as LSTM: give the model more
#    time to recover from LR reductions before stopping.
# 4. Max epochs 60 → 80 to complement the higher patience.
#
# LEAKAGE FIXES (retained from previous version):
# - Sort by [athlete_id, date].
# - Athlete-level train/val split.
# - injury_risk_score and athlete_id dropped.
# - Per-athlete sequences.
# - Sinusoidal positional encoding (so attention knows session ordering).

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Dropout,
    LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D,
    Layer,
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from src.ml.sequence_dataset import create_sequences


DATA_PATH  = Path("data/combined_dataset.csv")
MODEL_PATH = Path("models/ml/transformer_sequence_model.keras")
META_PATH  = Path("models/ml/transformer_meta.pkl")

SEQUENCE_LENGTH  = 30
VAL_ATHLETE_FRAC = 0.20


# ── Sinusoidal Positional Encoding ────────────────────────────────────────────
# FIX: now imported from shared layers.py to avoid duplication with hybrid_predictor.py
from .layers import SinusoidalPositionalEncoding


# ── Transformer block ─────────────────────────────────────────────────────────

def transformer_block(x, head_size: int = 48, num_heads: int = 4,
                      ff_dim: int = 256, dropout: float = 0.12):
    """
    Standard pre-norm transformer encoder block.
    ff_dim 64 → 128 for more representation capacity.
    head_size 32 → 48 for richer attention key/query projections.
    """
    attention = MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(x, x)
    attention = Dropout(dropout)(attention)
    x = LayerNormalization(epsilon=1e-6)(x + attention)

    ff = Dense(ff_dim, activation="relu")(x)
    ff = Dense(x.shape[-1])(ff)
    ff = Dropout(dropout)(ff)
    return LayerNormalization(epsilon=1e-6)(x + ff)


# ── Dataset helpers ───────────────────────────────────────────────────────────

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


# ── Main ──────────────────────────────────────────────────────────────────────

def train_transformer():
    print("Loading dataset...")
    df = load_dataset()

    feature_drop = [c for c in ["injury_risk_score_next", "injury_risk_score", "athlete_id"]
                    if c in df.columns]

    df_train, df_val = athlete_train_val_split(df, val_frac=VAL_ATHLETE_FRAC)

    print(f"Athletes — train: {df_train['athlete_id'].nunique()}, "
          f"val: {df_val['athlete_id'].nunique()}")

    feature_cols = [c for c in df.columns if c not in feature_drop]
    train_X_cols = [c for c in feature_cols if c != "injury_risk_score_next"]

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

    print("Building transformer...")
    inputs = Input(shape=(SEQUENCE_LENGTH, n_features))
    x = SinusoidalPositionalEncoding(max_len=SEQUENCE_LENGTH)(inputs)
    # Reduced from 3 blocks to 2 to prevent overfitting
    x = transformer_block(x, ff_dim=128, dropout=0.20)
    x = transformer_block(x, ff_dim=128, dropout=0.20)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.25)(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.15)(x)
    outputs = Dense(1)(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=3e-4),
        loss="mse",
        metrics=["mae"],
    )
    model.summary()

    callbacks = [
        EarlyStopping(patience=15, restore_best_weights=True, monitor="val_loss"),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4,
                          min_lr=1e-6, verbose=1),
    ]

    print("Training transformer (max 120 epochs, early stop patience=15)...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=120,
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
    print("\nTransformer saved to:", MODEL_PATH)


if __name__ == "__main__":
    train_transformer()
