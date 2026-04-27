# src/realtime/feature_builder.py
#
# v4 — Option C refactor. Removed the CV-feature defaults block since the
# model no longer trains on CV columns. The video influence is applied
# downstream via HybridPredictor's biomech channel.
#
# REWRITTEN to exactly mirror the training pipeline in build_combined_dataset.py:
#   1. FeatureEngineer(df).run()           → ACWR, EWMA, volatility, interactions, baseline
#   2. add_accuracy_features(df)           → 21 accuracy-boosting features
#
# (v3's step 2 — generate_video_features — was removed in v4 along with the
# matching code here. CV columns are no longer required by the trained model.)

import pandas as pd
import numpy as np
from typing import Dict


class RealtimeFeatureBuilder:
    """
    Builds model-ready features for a single session using athlete history.
    Mirrors the exact training pipeline from build_combined_dataset.py.
    """

    TARGET_COLS = {
        "injuryriskscore_next",
        "injury_risk_score_next",
    }

    def __init__(self, rolling_window: int = 7):
        self.rolling_window = rolling_window

    def build_features(
        self,
        current_session: Dict,
        history_df: pd.DataFrame,
        return_full_history: bool = False,
    ) -> pd.DataFrame:
        """
        Engineer features for the latest session given prior history.

        Args:
            current_session: dict of raw session inputs.
            history_df:      DataFrame of prior sessions (may be empty).
            return_full_history:
                False → return only the latest engineered row.
                True  → return the FULL engineered history (for sequence models).
        """

        current_df = pd.DataFrame([current_session])

        if not history_df.empty:
            combined = pd.concat([history_df, current_df], ignore_index=True)
        else:
            combined = current_df.copy()

        combined = combined.copy()

        if "athlete_id" not in combined.columns:
            combined["athlete_id"] = 1
        if "date" not in combined.columns:
            combined["date"] = pd.date_range("2020-01-01", periods=len(combined)).astype(str)
        if "injury_risk_score" not in combined.columns:
            combined["injury_risk_score"] = 50.0

        combined = combined.sort_values(["athlete_id", "date"]).reset_index(drop=True)

        # ═══════════════════════════════════════════════════════════
        # Step 1: FeatureEngineer features (mirrors feature_engineering.py)
        # ═══════════════════════════════════════════════════════════

        # ACWR
        acute = combined["training_duration"].rolling(7, min_periods=1).mean()
        chronic = combined["training_duration"].rolling(28, min_periods=1).mean()
        combined["acwr"] = acute / (chronic + 1e-6)

        # EWMA
        for span in [7, 14, 28]:
            combined[f"training_duration_ewma_{span}"] = (
                combined["training_duration"].ewm(span=span).mean()
            )

        # Volatility
        for col in ["training_duration", "sleep_hours", "fatigue_level"]:
            if col in combined.columns:
                combined[f"{col}_std_7"] = (
                    combined[col].rolling(7, min_periods=1).std().fillna(0)
                )

        # Acceleration
        combined["training_duration_diff1"] = combined["training_duration"].diff(1).fillna(0)
        combined["training_duration_diff2"] = combined["training_duration"].diff(2).fillna(0)

        # Interactions
        if "fatigue_level" in combined.columns:
            combined["training_fatigue_interaction"] = (
                combined["training_duration"] * combined["fatigue_level"]
            )
        if "sleep_hours" in combined.columns:
            combined["training_sleep_interaction"] = (
                combined["training_duration"] * combined["sleep_hours"]
            )
        if {"heart_rate_variability", "fatigue_level"}.issubset(combined.columns):
            combined["hrv_fatigue_interaction"] = (
                combined["heart_rate_variability"] * combined["fatigue_level"]
            )
        if "acwr" in combined.columns and "fatigue_level" in combined.columns:
            combined["acwr_fatigue_interaction"] = (
                combined["acwr"] * combined["fatigue_level"]
            )

        # Fatigue index
        if "sleep_hours" in combined.columns:
            combined["sleep_debt"] = 8 - combined["sleep_hours"]
        if "acwr" in combined.columns and "fatigue_level" in combined.columns:
            combined["fatigue_index"] = (
                0.5 * combined["acwr"]
                + 0.3 * combined["fatigue_level"]
                + 0.2 * combined.get("sleep_debt", 0)
            )

        # Baseline features
        for col in ["heart_rate_variability", "sleep_hours", "fatigue_level"]:
            if col in combined.columns:
                mean = combined[col].expanding().mean()
                std  = combined[col].expanding().std().fillna(1)
                combined[f"{col}_baseline_dev"] = combined[col] - mean
                combined[f"{col}_baseline_z"]   = (combined[col] - mean) / (std + 1e-6)

        # ═══════════════════════════════════════════════════════════
        # Step 2: Accuracy features (mirrors add_accuracy_features)
        # ═══════════════════════════════════════════════════════════
        # NOTE v4: the CV-defaults block from v3 was removed. The model no
        # longer trains on posture_symmetry / knee_velocity / etc., so we
        # don't need to fabricate them here. Real video influence is applied
        # through HybridPredictor's biomech channel.
        try:
            from src.utils.build_combined_dataset import add_accuracy_features
            combined = add_accuracy_features(combined)
        except Exception:
            self._add_basic_accuracy_features(combined)

        # Strip RAW VIDEO columns from the model-input frame.
        # These come from extract_biomechanics_from_video() and are needed
        # for the biomech-rules channel, but they must not enter the ML
        # predictor's feature space. The predictor's _align() would silently
        # drop them, but explicit removal here keeps the contract clear and
        # avoids polluting downstream debug prints.
        raw_video_cols = [
            # MediaPipe per-frame stats
            "n_frames_analysed", "n_frames", "detection_rate",
            "knee_angle_left_mean", "knee_angle_right_mean",
            "knee_angle_left_std",  "knee_angle_right_std",
            "elbow_angle_left_mean", "elbow_angle_right_mean",
            "elbow_angle_left_std",  "elbow_angle_right_std",
            "posture_symmetry_mean", "posture_symmetry_std",
            "balance_stability_mean", "balance_stability_std",
            "body_symmetry_score_mean", "body_symmetry_score_std",
            "shoulder_symmetry_mean", "shoulder_symmetry_std",
            "movement_fluidity_mean", "movement_fluidity_std",
            "torso_lean_mean", "torso_lean_std",
            "movement_quality_score", "landing_risk_index",
            "running_knee_angle_left_std", "running_knee_angle_right_std",
        ]
        for c in raw_video_cols:
            if c in combined.columns:
                combined = combined.drop(columns=c)

        # ═══════════════════════════════════════════════════════════
        # Extract latest row OR return full engineered history
        # ═══════════════════════════════════════════════════════════
        cols_to_drop = [c for c in list(self.TARGET_COLS) +
                        ["date", "injury_risk_score", "athlete_id"]
                        if c in combined.columns]

        if return_full_history:
            full = combined.copy()
            if cols_to_drop:
                full = full.drop(columns=cols_to_drop)
            full = full.fillna(0)
            return full

        latest_row = combined.iloc[[-1]].copy()
        if cols_to_drop:
            latest_row = latest_row.drop(columns=cols_to_drop)
        latest_row = latest_row.fillna(0)
        return latest_row

    def _add_basic_accuracy_features(self, df: pd.DataFrame):
        """Fallback when the import path fails (e.g. in tests)."""
        df["session_of_season"] = range(1, len(df) + 1)

        for col, alias in [("heart_rate_variability", "hrv"), ("sprint_count", "spri")]:
            if col in df.columns:
                acute   = df[col].rolling(7,  min_periods=1).mean()
                chronic = df[col].rolling(28, min_periods=1).mean()
                df[f"acwr_{alias}"] = acute / (chronic + 1e-6)

        df["injury_risk_trend_7d"] = 0
        df["consecutive_high_load"] = 0
        df["load_spike"] = 0
        df["hrv_trend_7d"] = 0
        df["sleep_trend_7d"] = 0

        if "fatigue_level" in df.columns:
            df["fatigue_diff1"] = df["fatigue_level"].diff(1).fillna(0)
            df["fatigue_diff2"] = df["fatigue_level"].diff(2).fillna(0)

        if all(c in df.columns for c in ["heart_rate_variability", "sleep_hours", "wellness_score"]):
            hrv_norm      = ((df["heart_rate_variability"].clip(20, 100) - 20) / 80) * 100
            sleep_norm    = (df["sleep_hours"].clip(3, 10) - 3) / 7 * 100
            wellness_norm = (df["wellness_score"] / 10) * 100
            df["recovery_score"] = (
                0.40 * hrv_norm + 0.35 * sleep_norm + 0.25 * wellness_norm
            ).clip(0, 100)
        else:
            df["recovery_score"] = 50

        df["strain_recovery_ratio"] = df["training_duration"] / (df["recovery_score"] + 1)
        df["weekly_load_sum"] = df["training_duration"].rolling(7, min_periods=1).sum()

        roll_mean7 = df["training_duration"].rolling(7, min_periods=1).mean()
        roll_std7  = df["training_duration"].rolling(7, min_periods=1).std().fillna(1)
        df["load_monotony"] = roll_mean7 / (roll_std7 + 1e-6)
        df["training_strain"] = df["load_monotony"] * df["weekly_load_sum"]
        df["peak_load_14d"] = df["training_duration"].rolling(14, min_periods=1).max()
        df["days_since_rest"] = 0

        if "intensity_rating" in df.columns:
            session_load = df["training_duration"] * df["intensity_rating"] / 10.0
            df["accumulated_load_7d"] = session_load.rolling(7, min_periods=1).sum()
        else:
            df["accumulated_load_7d"] = df["training_duration"].rolling(7, min_periods=1).sum()

        if all(c in df.columns for c in ["heart_rate_variability", "sleep_hours"]):
            df["hrv_sleep_interaction"] = df["heart_rate_variability"] * df["sleep_hours"]

        rec = df["recovery_score"] if "recovery_score" in df.columns else pd.Series(50.0, index=df.index)
        df["load_recovery_balance"] = df["training_duration"] / (rec + 1)

        fat_d1 = df["fatigue_diff1"] if "fatigue_diff1" in df.columns else pd.Series(0.0, index=df.index)
        df["fatigue_acceleration"] = fat_d1.diff(1).fillna(0)

        if "intensity_rating" in df.columns:
            df["intensity_ewma_7"] = df["intensity_rating"].ewm(span=7).mean()

        if all(c in df.columns for c in ["wellness_score", "fatigue_level"]):
            df["wellness_fatigue_ratio"] = df["wellness_score"] / (df["fatigue_level"] + 0.1)
