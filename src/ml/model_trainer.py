# src/ml/model_trainer.py
#
# FIXES applied in this version:
# 1. injury_risk_score and athlete_id dropped from features.
# 2. Athlete-level train/test split (last 20% of athletes as test) instead of
#    row-level shuffle=False split.  Row-level on a date-sorted interleaved
#    dataset gives the last 20% of ALL athletes' records simultaneously — not
#    a true holdout.  Athlete-level split means test athletes are completely
#    unseen during training, consistent with the LSTM/Transformer splits.
# 3. R² and MAE added to performance report.

import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor


DATA_PATH    = Path("data/combined_dataset.csv")
MODEL_PATH   = Path("models/ml/injury_model.pkl")
FEATURE_PATH = Path("models/ml/feature_names.pkl")


def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "combined_dataset.csv not found. Run: python -m src.utils.build_combined_dataset"
        )
    df = pd.read_csv(DATA_PATH)
    sort_cols = ["athlete_id", "date"] if "date" in df.columns else ["athlete_id"]
    df = df.sort_values(sort_cols).reset_index(drop=True)
    if "date" in df.columns:
        df = df.drop(columns=["date"])
    return df



def athlete_split(df: pd.DataFrame, val_frac: float = 0.20):
    """
    Split by athlete — last val_frac of athlete IDs form the test set.
    No athlete appears in both train and test.
    """
    athletes  = sorted(df["athlete_id"].unique())
    n_test    = max(1, int(len(athletes) * val_frac))
    test_ids  = set(athletes[-n_test:])
    train_ids = set(athletes) - test_ids
    return (
        df[df["athlete_id"].isin(train_ids)].copy(),
        df[df["athlete_id"].isin(test_ids)].copy(),
    )


def train_model(df: pd.DataFrame):
    # Athlete-level split
    train_df, test_df = athlete_split(df)

    drop_cols = [c for c in ["injury_risk_score_next", "injury_risk_score", "athlete_id"]
                 if c in df.columns]

    X_train = train_df.drop(columns=drop_cols)
    y_train = train_df["injury_risk_score_next"]
    X_test  = test_df.drop(columns=drop_cols)
    y_test  = test_df["injury_risk_score_next"]

    print(f"Train: {len(X_train):,} rows ({train_df['athlete_id'].nunique()} athletes)")
    print(f"Test:  {len(X_test):,} rows ({test_df['athlete_id'].nunique()} athletes)")
    print(f"Features: {X_train.shape[1]}")

    model = XGBRegressor(
        n_estimators=600,
        max_depth=7,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.75,
        min_child_weight=3,
        gamma=0.1,
        reg_lambda=1.0,
        reg_alpha=0.1,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse  = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2    = float(r2_score(y_test, preds))
    mae   = float(mean_absolute_error(y_test, preds))

    print("\nModel Performance (athlete-level held-out test)")
    print("------------------------------------------------")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²:   {r2:.3f}")
    print(f"MAE:  {mae:.3f}")

    print("\nTop-10 feature importances:")
    fi = pd.Series(model.feature_importances_, index=X_train.columns)
    print(fi.nlargest(10).to_string())

    return model, list(X_train.columns)


def main():
    print("Loading dataset...")
    df = load_dataset()

    print("Training model...")
    model, feature_names = train_model(df)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(feature_names, FEATURE_PATH)
    print("\nModel saved to:", MODEL_PATH)


if __name__ == "__main__":
    main()
