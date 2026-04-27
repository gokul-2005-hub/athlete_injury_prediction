# src/ml/ensemble_model.py
#
# v3 — Athlete-level split + tuned hyperparameters
#
# LEAKAGE FIX: switched from row-level 60/20/20 chronological split to
# athlete-level split. Row-level split leaked within-athlete temporal
# patterns (trained on athlete A's early sessions, tested on A's late
# sessions). Athlete-level split ensures test athletes are completely
# unseen.
#
# ACCURACY IMPROVEMENTS:
# 1. Athlete-level 3-way split (60/20/20 of athletes)
# 2. XGB: n_estimators=700, learning_rate=0.03, max_depth=7
# 3. LGB: n_estimators=700, num_leaves=95
# 4. CatBoost: iterations=600, depth=7
# 5. ExtraTrees: n_estimators=600, max_depth=14
# 6. Ridge meta trained on OOF predictions only
# 7. All base models retrained on train+meta athletes for final artefact

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor


DATA_PATH  = Path("data/combined_dataset.csv")
MODEL_PATH = Path("models/ml/stacked_injury_model.pkl")


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    sort_cols = ["athlete_id", "date"] if "date" in df.columns else ["athlete_id"]
    df = df.sort_values(sort_cols).reset_index(drop=True)
    if "date" in df.columns:
        df = df.drop(columns=["date"])
    return df


def athlete_3way_split(df: pd.DataFrame, base_frac=0.60, meta_frac=0.20):
    """
    Split by athlete ID into 3 groups: base (60%), meta (20%), test (20%).
    No athlete appears in more than one group.
    """
    athletes = sorted(df["athlete_id"].unique())
    n = len(athletes)
    n_base = int(n * base_frac)
    n_meta = int(n * meta_frac)

    base_ids = set(athletes[:n_base])
    meta_ids = set(athletes[n_base:n_base + n_meta])
    test_ids = set(athletes[n_base + n_meta:])

    return (
        df[df["athlete_id"].isin(base_ids)].copy(),
        df[df["athlete_id"].isin(meta_ids)].copy(),
        df[df["athlete_id"].isin(test_ids)].copy(),
    )


def train_ensemble():
    print("Loading dataset...")
    df = load_dataset()

    # Drop leaky and identifier columns
    drop_cols = ["injury_risk_score_next"]
    for col in ["injury_risk_score", "athlete_id"]:
        if col in df.columns:
            drop_cols.append(col)
            print(f"Dropping '{col}' from features.")

    # ── Athlete-level 3-way split ─────────────────────────────────
    df_base, df_meta, df_test = athlete_3way_split(df)

    feature_cols = [c for c in df.columns if c not in drop_cols]

    X_base = df_base[feature_cols]
    y_base = df_base["injury_risk_score_next"]
    X_meta = df_meta[feature_cols]
    y_meta = df_meta["injury_risk_score_next"]
    X_test = df_test[feature_cols]
    y_test = df_test["injury_risk_score_next"]

    print(f"Feature count: {X_base.shape[1]}")
    print(f"Splits — base: {len(X_base)} ({df_base['athlete_id'].nunique()} athletes), "
          f"meta: {len(X_meta)} ({df_meta['athlete_id'].nunique()} athletes), "
          f"test: {len(X_test)} ({df_test['athlete_id'].nunique()} athletes)")

    # ── Train base models ─────────────────────────────────────────
    print("Training base models...")

    xgb_model = xgb.XGBRegressor(
        n_estimators=700,
        max_depth=7,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=3,
        gamma=0.1,
        reg_lambda=1.0,
        reg_alpha=0.1,
        random_state=42,
        n_jobs=-1,
    )
    lgb_model = lgb.LGBMRegressor(
        n_estimators=700,
        learning_rate=0.03,
        num_leaves=95,
        min_child_samples=15,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_lambda=1.0,
        reg_alpha=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    cat_model = CatBoostRegressor(
        iterations=600,
        learning_rate=0.04,
        depth=7,
        l2_leaf_reg=3.0,
        verbose=False,
        random_seed=42,
    )
    et_model = ExtraTreesRegressor(
        n_estimators=600,
        max_depth=14,
        min_samples_leaf=4,
        max_features=0.7,
        random_state=42,
        n_jobs=-1,
    )

    xgb_model.fit(X_base, y_base)
    lgb_model.fit(X_base, y_base)
    cat_model.fit(X_base, y_base)
    et_model.fit(X_base, y_base)

    # Feature importance log
    print("\nTop-10 XGB feature importances:")
    fi = pd.Series(xgb_model.feature_importances_, index=X_base.columns)
    print(fi.nlargest(10).to_string())

    # ── Meta predictions on holdout ───────────────────────────────
    print("\nGenerating meta predictions on meta-holdout split...")
    meta_features = np.column_stack([
        xgb_model.predict(X_meta),
        lgb_model.predict(X_meta),
        cat_model.predict(X_meta),
        et_model.predict(X_meta),
    ])

    meta_model = Ridge(alpha=1.0)
    meta_model.fit(meta_features, y_meta)
    print(f"Ridge meta weights: XGB={meta_model.coef_[0]:.3f}, "
          f"LGB={meta_model.coef_[1]:.3f}, Cat={meta_model.coef_[2]:.3f}, "
          f"ET={meta_model.coef_[3]:.3f}")

    # ── Evaluate on unbiased test split ───────────────────────────
    test_features = np.column_stack([
        xgb_model.predict(X_test),
        lgb_model.predict(X_test),
        cat_model.predict(X_test),
        et_model.predict(X_test),
    ])
    final_pred = meta_model.predict(test_features)
    rmse = float(np.sqrt(mean_squared_error(y_test, final_pred)))
    r2   = float(r2_score(y_test, final_pred))
    mae  = float(mean_absolute_error(y_test, final_pred))

    # Per-model metrics for dashboard
    model_metrics = {}
    for name, m in [("XGBoost", xgb_model), ("LightGBM", lgb_model),
                    ("CatBoost", cat_model), ("ExtraTrees", et_model)]:
        preds = m.predict(X_test)
        model_metrics[name] = {
            "RMSE": float(np.sqrt(mean_squared_error(y_test, preds))),
            "R2": float(r2_score(y_test, preds)),
            "MAE": float(mean_absolute_error(y_test, preds)),
            "predictions": preds.tolist(),
        }
    model_metrics["Stacked_Ensemble"] = {
        "RMSE": rmse, "R2": r2, "MAE": mae,
        "predictions": final_pred.tolist(),
    }

    print("\nStacked Model Performance (athlete-level held-out test)")
    print("-" * 52)
    print(f"RMSE: {rmse:.3f}")
    print(f"R²:   {r2:.3f}")
    print(f"MAE:  {mae:.3f}")

    print("\nPer-model test performance:")
    for name, m in model_metrics.items():
        if name != "Stacked_Ensemble":
            print(f"  {name:12s}: R²={m['R2']:.3f}  RMSE={m['RMSE']:.3f}  MAE={m['MAE']:.3f}")

    # ── Retrain on full 80% for saved artefact ────────────────────
    print("\nRetraining base models on base + meta splits...")
    X_full = pd.concat([X_base, X_meta])
    y_full = pd.concat([y_base, y_meta])

    xgb_model.fit(X_full, y_full)
    lgb_model.fit(X_full, y_full)
    cat_model.fit(X_full, y_full)
    et_model.fit(X_full, y_full)

    # ── Save ──────────────────────────────────────────────────────
    bundle = {
        "xgb":           xgb_model,
        "lgb":           lgb_model,
        "cat":           cat_model,
        "et":            et_model,
        "meta":          meta_model,
        "features":      list(X_base.columns),
        "n_meta_inputs": 4,
        "test_metrics":  {"RMSE": rmse, "R2": r2, "MAE": mae},
        "model_metrics": model_metrics,
        "y_test":        y_test.tolist(),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    print("\nStacked model saved to:", MODEL_PATH)


if __name__ == "__main__":
    train_ensemble()
