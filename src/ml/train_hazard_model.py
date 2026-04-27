# src/ml/train_hazard_model.py
#
# FIX: Consolidated to a single chronological split.
#
# Previous version had a confusing double-split:
# - Split 80/20 externally to compute the 90th-pct threshold on train only ✓
# - Then passed the full labelled dataset to model.train() which split 80/20
#   again internally — creating an ambiguous evaluation where the inner test
#   set overlapped with the outer train set.
#
# New approach:
# - Split once (80/20, no shuffle).
# - Compute threshold on df_train only.
# - Apply threshold to both splits using the locked train threshold.
# - model.train() receives only df_train; model is then evaluated on df_test
#   using model.predict() — giving a single, unambiguous held-out evaluation.

from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score

from src.ml.hazard_model import HazardModel


DATA_PATH  = Path("data/combined_dataset.csv")
MODEL_PATH = Path("models/ml/hazard_model.pkl")


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)
    return df


def main():
    print("Loading dataset...")
    df = load_dataset()

    # Drop injury_risk_score_next — not a feature for the survival model.
    if "injury_risk_score_next" in df.columns:
        df = df.drop(columns=["injury_risk_score_next"])

    # ── Athlete-level split (matches other models) ────────────────────
    # FIX: chronological split leaked within-athlete temporal data.
    # Now we hold out entire athletes so no test athlete is seen in training.
    athlete_ids = sorted(df["athlete_id"].unique())
    n_train_athletes = int(len(athlete_ids) * 0.80)
    train_athletes = athlete_ids[:n_train_athletes]
    test_athletes  = athlete_ids[n_train_athletes:]

    df_train = df[df["athlete_id"].isin(train_athletes)].copy()
    df_test  = df[df["athlete_id"].isin(test_athletes)].copy()

    print(f"Train athletes: {len(train_athletes)}, Test athletes: {len(test_athletes)}")
    print(f"Train rows: {len(df_train):,}, Test rows: {len(df_test):,}")

    model = HazardModel()

    # ── Threshold computed on train only ─────────────────────────────
    df_train, threshold = model.create_event_labels(df_train)
    print(f"Injury event threshold (90th pct, train only): {threshold:.2f}")
    print(f"Event rate — train: "
          f"{df_train['injury_event'].mean():.1%}, "
          f"test: {(df_test['injury_risk_score'] >= threshold).mean():.1%}")

    # Apply locked threshold to test — no distribution leakage
    df_test, _ = model.create_event_labels(df_test, threshold=threshold)

    # ── Build survival datasets ───────────────────────────────────────
    X_train, y_train = model.build_survival_dataset(df_train)
    X_test,  y_test  = model.build_survival_dataset(df_test)

    # ── Fit on training data only ─────────────────────────────────────
    print("Training hazard model on training split...")
    model.model.fit(X_train, y_train)

    # ── Evaluate on held-out test split ──────────────────────────────
    risk_scores = model.model.predict(X_test)
    auc = roc_auc_score(y_test["event"], risk_scores)

    print("\nHazard Model Performance (athlete-level held-out test set)")
    print("ROC-AUC:", round(auc, 4))

    # ── Save percentile-rank reference + bundle ──────────────────────
    # v6 fix: previously we saved just the max raw hazard from the training
    # set and normalised inference scores via raw / max * 100. That made
    # the hazard channel almost always show 0 in the dashboard because
    # the max is dominated by a small number of extreme outliers — every
    # ordinary session lands at < 1% of max.
    #
    # Now we save the SORTED ASC distribution of training raw hazards,
    # and HybridPredictor.predict_hazard() uses np.searchsorted to find
    # the percentile rank. So a session at the 50th percentile of the
    # training distribution shows as 50.0; at the 95th, as 95.0. Properly
    # spread across 0-100, calibrated against the training population.
    train_raw_hazard = model.model.predict(X_train)
    train_hazard_dist = np.sort(train_raw_hazard).astype(np.float64)
    print(f"Train hazard distribution percentiles "
          f"(min/25/50/75/95/max): "
          f"{train_hazard_dist[0]:.2f} / "
          f"{np.percentile(train_hazard_dist, 25):.2f} / "
          f"{np.percentile(train_hazard_dist, 50):.2f} / "
          f"{np.percentile(train_hazard_dist, 75):.2f} / "
          f"{np.percentile(train_hazard_dist, 95):.2f} / "
          f"{train_hazard_dist[-1]:.2f}")

    model._train_threshold   = threshold
    model._train_hazard_dist = train_hazard_dist
    model.save(str(MODEL_PATH))
    print("\nHazard model saved:", MODEL_PATH)


if __name__ == "__main__":
    main()

