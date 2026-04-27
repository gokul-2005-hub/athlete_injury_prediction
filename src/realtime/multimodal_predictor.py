# src/realtime/multimodal_predictor.py
#
# ╔══════════════════════════════════════════════════════════════════════╗
# ║  DEPRECATED — DO NOT USE IN NEW CODE                                ║
# ║                                                                      ║
# ║  This class is no longer imported anywhere in the project. Use       ║
# ║  src.ml.hybrid_predictor.HybridPredictor instead — it is the         ║
# ║  canonical predictor used by both Streamlit dashboards.              ║
# ║                                                                      ║
# ║  Specific differences vs HybridPredictor that make this file unsafe  ║
# ║  to use as-is:                                                       ║
# ║                                                                      ║
# ║  1. Hazard normalisation here uses 1 - exp(-raw), which produces a   ║
# ║     different scale than the saved-train-max approach used by        ║
# ║     HybridPredictor. Mixing the two will give inconsistent numbers.  ║
# ║                                                                      ║
# ║  2. The Ridge meta in the stacked ensemble was retrained to expect   ║
# ║     4 base-model inputs (XGB + LGB + Cat + ET). Older callers of     ║
# ║     this class may still pass 3 — check predict_stack() before use.  ║
# ║                                                                      ║
# ║  3. No MC-Dropout uncertainty support. No graceful degradation when  ║
# ║     a sequence model can't run.                                      ║
# ║                                                                      ║
# ║  Kept on disk only for reference / archaeology. Will be deleted in a ║
# ║  future cleanup.                                                     ║
# ╚══════════════════════════════════════════════════════════════════════╝

import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from src.realtime.feature_builder import RealtimeFeatureBuilder
from src.cv.biomechanics_features import BiomechanicsFeatureExtractor
from config import LOW_RISK_MAX, MEDIUM_RISK_MAX


class MultiModalPredictor:

    def __init__(self):
        print("Loading models...")

        self.tabular_model = joblib.load("models/ml/stacked_injury_model.pkl")
        self.hazard_model  = joblib.load("models/ml/hazard_model.pkl")
        self.sequence_model = load_model("models/ml/lstm_sequence_model.keras")

        # FIX: use RealtimeFeatureBuilder — correct runtime feature API.
        # FeatureEngineer requires a full DataFrame at construction time and
        # is a batch utility, not a session-by-session builder.
        self.feature_builder    = RealtimeFeatureBuilder(rolling_window=7)
        self.biomech_extractor  = BiomechanicsFeatureExtractor()

        self.base_columns = [
            "athlete_id",
            "training_duration",
            "heart_rate_variability",
            "running_distance",
            "sprint_count",
            "sleep_hours",
            "intensity_rating",
            "previous_injuries",
            "fatigue_level",
            "wellness_score",
            "posture_symmetry",
            "balance_score",
            "movement_smoothness",
        ]

    def build_input_dataframe(self, wearable_dict: dict) -> pd.DataFrame:
        defaults = {
            "athlete_id":         1,
            "intensity_rating":   5,
            "posture_symmetry":   0.85,
            "balance_score":      0.85,
            "movement_smoothness": 0.85,
        }
        row = {
            col: wearable_dict.get(col, defaults.get(col, 0))
            for col in self.base_columns
        }
        return pd.DataFrame([row])

    def process_wearable_data(
        self,
        wearable_dict: dict,
        history_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Build a feature-engineered row for the current session.

        Args:
            wearable_dict: raw session data dict.
            history_df:    previous sessions for this athlete (may be empty).
        """
        current_df = self.build_input_dataframe(wearable_dict)
        session    = current_df.iloc[0].to_dict()
        history    = history_df if history_df is not None else pd.DataFrame()

        # FIX: use RealtimeFeatureBuilder.build_features() — the correct method.
        features_df = self.feature_builder.build_features(session, history)
        return features_df.select_dtypes(include=["number"])

    def process_video_landmarks(self, landmark_sequence):
        return self.biomech_extractor.extract_features_from_sequence(
            landmark_sequence
        )

    def normalize_hazard(self, raw_score: float) -> float:
        """Convert RSF cumulative hazard to [0,1] probability."""
        # FIX: old formula raw_score/max(raw_score,1.0) always returned 1.0
        # for positive scores. Use sigmoid-like transform instead.
        return float(np.clip(1.0 - np.exp(-raw_score), 0, 1))

    def predict(
        self,
        wearable_dict: dict,
        landmark_sequence=None,
        history_df: pd.DataFrame | None = None,
    ) -> dict:

        wearable_df = self.process_wearable_data(wearable_dict, history_df)

        # Stack model expects exactly the features it was trained on
        stack_bundle = self.tabular_model
        feat_cols    = stack_bundle["features"]
        aligned_df   = wearable_df.reindex(columns=feat_cols, fill_value=0)

        # FIX: Include ExtraTrees as 4th model — Ridge meta was trained on 4 inputs
        if "et" in stack_bundle:
            p_stack = np.column_stack([
                stack_bundle["xgb"].predict(aligned_df),
                stack_bundle["lgb"].predict(aligned_df),
                stack_bundle["cat"].predict(aligned_df),
                stack_bundle["et"].predict(aligned_df),
            ])
        else:
            p_stack = np.column_stack([
                stack_bundle["xgb"].predict(aligned_df),
                stack_bundle["lgb"].predict(aligned_df),
                stack_bundle["cat"].predict(aligned_df),
            ])
        base_prediction = float(stack_bundle["meta"].predict(p_stack)[0])

        # Hazard probability
        raw_hazard   = float(self.hazard_model.predict(aligned_df)[0])
        hazard_prob  = self.normalize_hazard(raw_hazard)

        # FIX: risk_level thresholds from config.py — consistent across the app.
        if base_prediction >= MEDIUM_RISK_MAX or hazard_prob >= 0.70:
            risk_level = "HIGH"
        elif base_prediction >= LOW_RISK_MAX:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        results = {
            "injury_risk_score":  float(np.clip(base_prediction, 0, 100)),
            "injury_probability": hazard_prob,
            "risk_level":         risk_level,
        }

        if landmark_sequence is not None:
            biomech_features  = self.process_video_landmarks(landmark_sequence)
            movement_score    = 100 - biomech_features.get("movement_smoothness", 0)
            results["movement_quality_score"] = float(movement_score)

        return results
