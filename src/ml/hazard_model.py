# src/ml/hazard_model.py

import os
import numpy as np
import pandas as pd
import joblib


from sklearn.metrics import roc_auc_score

from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv


class HazardModel:
    """
    Random Survival Forest for injury hazard prediction.

    Fixes applied vs original:
    1. Survival time is computed per-athlete (row rank within each athlete),
       not as a global row index — global index made the RSF learn meaningless
       time values (athlete 2's first session was time=501, not time=1).
    2. The high-risk threshold (90th percentile) is now computed on the
       training set only.  The original computed it on the full dataset,
       leaking test-set distribution into the event labels.
    3. train_test_split uses shuffle=False so the split respects
       chronological order.
    """

    def __init__(self):
        self.model = RandomSurvivalForest(
            n_estimators=200,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features="sqrt",
            n_jobs=-1,
            random_state=42,
        )
        self._train_threshold:    float | None     = None
        # Sorted ascending array of raw cumulative-hazard scores from the
        # training set. predict_hazard() uses np.searchsorted on this to
        # convert a raw score into a 0–100 percentile rank — much better
        # behaved than dividing by the training-set MAX (which crushes
        # everyday sessions to 0 because the max is dominated by a handful
        # of extreme rows).
        self._train_hazard_dist:  np.ndarray | None = None

    # ------------------------------------------------------------------
    # Per-athlete survival time
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_survival_time(df: pd.DataFrame) -> np.ndarray:
        """
        Assign each row its chronological rank within its athlete group
        (1-based).  This is a meaningful approximation of time-to-event
        expressed in training sessions.

        FIX: original used np.arange(len(df)) + 1, which assigned a global
        index and completely ignored athlete_id boundaries.
        """
        if "athlete_id" in df.columns:
            return (
                df.groupby("athlete_id").cumcount() + 1
            ).values.astype(float)
        # Fallback when athlete_id is absent
        return np.arange(len(df), dtype=float) + 1

    # ------------------------------------------------------------------
    # Event labels
    # ------------------------------------------------------------------

    def create_event_labels(
        self,
        df: pd.DataFrame,
        score_column: str = "injury_risk_score",
        threshold: float | None = None,
    ) -> tuple[pd.DataFrame, float]:
        """
        Mark sessions above the 90th-percentile risk score as injury events.

        Args:
            threshold: if supplied (train split threshold), use it directly;
                       otherwise compute from df (use only on train split).
        """
        if threshold is None:
            threshold = float(np.percentile(df[score_column], 90))
        df = df.copy()
        df["injury_event"] = (df[score_column] >= threshold).astype(int)
        return df, threshold

    # ------------------------------------------------------------------
    # Survival dataset
    # ------------------------------------------------------------------

    def build_survival_dataset(
        self,
        df: pd.DataFrame,
        target_column: str = "injury_event",
    ) -> tuple[pd.DataFrame, np.ndarray]:
        survival_time = self._compute_survival_time(df)
        events        = df[target_column].astype(bool)
        y             = Surv.from_arrays(events, survival_time)

        drop = [c for c in ["injury_risk_score", "injury_event", "date",
                             "athlete_id"] if c in df.columns]
        X = df.select_dtypes(include=["number"]).drop(columns=drop, errors="ignore")
        return X, y

    # ------------------------------------------------------------------
    # Train — handled by train_hazard_model.py
    # ------------------------------------------------------------------
    # NOTE: Training is done externally via train_hazard_model.py which
    # calls self.model.fit(X_train, y_train) with a proper athlete-level
    # split.  No internal .train() method needed.

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    # predict() alias kept for compatibility with HybridPredictor
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_risk(X)

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        bundle = {
            "model":              self.model,
            "train_threshold":    self._train_threshold,
            "train_hazard_dist":  self._train_hazard_dist,   # sorted asc
        }
        joblib.dump(bundle, path)
        print("\nHazard model saved to:", path)

    @classmethod
    def load(cls, path: str) -> "HazardModel":
        """Load a saved HazardModel bundle (model + threshold + hazard dist)."""
        raw = joblib.load(path)
        instance = cls()
        if isinstance(raw, dict) and "model" in raw:
            instance.model              = raw["model"]
            instance._train_threshold   = raw.get("train_threshold")
            # New v6 bundles store the sorted training distribution. Older
            # bundles may still ship train_max_hazard — keep a back-compat
            # path so HybridPredictor doesn't crash when loading them, but
            # fall through to the warning in HybridPredictor.__init__.
            instance._train_hazard_dist = raw.get("train_hazard_dist")
        else:
            # Backward compat: very old saves stored the RSF directly
            instance.model = raw
        return instance

