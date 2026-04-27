# src/utils/build_combined_dataset.py
#
# v4 — Option C refactor: removed fake CV features from the ML training pipeline
#
# ── WHAT CHANGED VS v3 ──────────────────────────────────────────────────────
# REMOVED: generate_video_features() call. The 9 "biomechanical" columns
#   (posture_symmetry, balance_score, movement_smoothness, joint_variability,
#   knee_velocity, hip_velocity, shoulder_velocity, movement_variability,
#   center_of_mass_stability) were NOT real video data — they were laundered
#   workload features (fatigue_proxy + noise). Keeping them as ML features
#   gave the false impression that video data influenced predictions when it
#   didn't.
#
# Now the ML models train on workload features only (~50 columns instead of
# 62). Video influence comes through a separate biomechanical-rules channel
# in HybridPredictor — see src/cv/biomech_risk_module.py.
#
# The function generate_video_features() is kept in the file for backward
# compatibility with code that explicitly imports it, but it's NO LONGER
# CALLED from build_dataset(). Marked clearly as deprecated.
#
# ── EVERYTHING ELSE PRESERVED ───────────────────────────────────────────────
# All leakage fixes, accuracy features (16-21), causal computations, and
# per-athlete rolling/groupby logic from v3 are unchanged.

import pandas as pd
import numpy as np
from pathlib import Path

from src.ml.feature_engineering import FeatureEngineer


DATA_DIR    = Path("data")
OUTPUT_FILE = DATA_DIR / "combined_dataset.csv"

RNG = np.random.default_rng(42)


# ── DEPRECATED Biomechanical feature generator ───────────────────────────────
# Kept only so existing imports don't break. NOT called from build_dataset().

def generate_video_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    DEPRECATED in v4 (Option C refactor).

    This function used to derive 9 "video" columns from fatigue/HRV/sleep,
    making the model appear to use video data when it really used a
    smoothed workload proxy. It's no longer part of the training pipeline.
    Real video influence is now applied through the rule-based biomechanical
    risk channel in HybridPredictor (src/cv/biomech_risk_module.py).

    Left in the module so any external caller that still imports it doesn't
    crash, but the build_dataset() function below does NOT call it.
    """
    n = len(df)
    fatigue_raw = (
        0.4 * df["training_duration"]
        + 0.3 * df["fatigue_level"]
        - 0.2 * df["heart_rate_variability"]
        - 0.1 * df["sleep_hours"]
    )
    if "athlete_id" in df.columns:
        grp_min = fatigue_raw.groupby(df["athlete_id"]).transform("min")
        grp_max = fatigue_raw.groupby(df["athlete_id"]).transform("max")
        fatigue_proxy = (fatigue_raw - grp_min) / (grp_max - grp_min + 1e-6)
    else:
        fatigue_proxy = (fatigue_raw - fatigue_raw.min()) / (fatigue_raw.max() - fatigue_raw.min() + 1e-6)
    if isinstance(fatigue_proxy, pd.DataFrame):
        fatigue_proxy = fatigue_proxy.iloc[:, 0]

    df["posture_symmetry"]    = np.clip(1 - (0.40 * fatigue_proxy.values + RNG.normal(0, 0.05, n)), 0, 1)
    df["balance_score"]       = np.clip(1 - (0.30 * fatigue_proxy.values + RNG.normal(0, 0.05, n)), 0, 1)
    df["movement_smoothness"] = np.clip(1 - (0.35 * fatigue_proxy.values + RNG.normal(0, 0.05, n)), 0, 1)
    df["joint_variability"]   = np.clip(fatigue_proxy.values              + RNG.normal(0, 0.05, n), 0, 1)
    df["knee_velocity"] = (
        df.groupby("athlete_id")["training_duration"]
          .diff(1).fillna(0) * 0.10
        + RNG.normal(0, 0.02, n)
    )
    df["hip_velocity"] = (
        df.groupby("athlete_id")["running_distance"]
          .diff(1).fillna(0) * 0.05
        + RNG.normal(0, 0.02, n)
    )
    df["shoulder_velocity"] = (
        df.groupby("athlete_id")["sprint_count"]
          .diff(1).fillna(0) * 0.08
        + RNG.normal(0, 0.02, n)
    )
    df["movement_variability"] = (
        df.groupby("athlete_id")["training_duration"]
          .transform(lambda x: x.rolling(5, min_periods=1).std().fillna(0))
    )
    df["center_of_mass_stability"] = np.clip(
        1 - (df["movement_variability"] * 0.2 + RNG.normal(0, 0.02, n)), 0, 1
    )
    return df


# ── Rolling slope helper ──────────────────────────────────────────────────────

def _rolling_slope(series: pd.Series, w: int = 7) -> pd.Series:
    """Compute OLS slope over a rolling window of size w."""
    slopes = []
    vals_list = series.values
    for i in range(len(vals_list)):
        start = max(0, i - w + 1)
        segment = vals_list[start: i + 1]
        if len(segment) < 2:
            slopes.append(0.0)
        else:
            x = np.arange(len(segment), dtype=float)
            slope = float(np.polyfit(x, segment, 1)[0])
            slopes.append(slope)
    return pd.Series(slopes, index=series.index)


# ── Accuracy-boosting features ────────────────────────────────────────────────

def add_accuracy_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 21 causal features per athlete. Computed strictly per-athlete and
    causally (no peeking at the future). Identical to v3 — only the upstream
    pipeline (no CV step) changed.
    """
    df = df.sort_values(["athlete_id", "date"]).copy()

    # 1. Session of season
    df["session_of_season"] = df.groupby("athlete_id").cumcount() + 1

    # 2. ACWR variants for HRV and sprint count
    for col, alias in [("heart_rate_variability", "hrv"), ("sprint_count", "spri")]:
        acute   = df.groupby("athlete_id")[col].transform(lambda x: x.rolling(7,  min_periods=1).mean())
        chronic = df.groupby("athlete_id")[col].transform(lambda x: x.rolling(28, min_periods=1).mean())
        df[f"acwr_{alias}"] = acute / (chronic + 1e-6)

    # 3. Injury risk trend — LEAKAGE FIX: use shift(1)
    df["injury_risk_trend_7d"] = (
        df.groupby("athlete_id", group_keys=False)
          .apply(lambda g: _rolling_slope(g["injury_risk_score"].shift(1).fillna(0), 7))
    )

    # 4. Consecutive high-load sessions
    def _streak(s: pd.Series) -> pd.Series:
        expanding_med = s.expanding().median()
        result, count = [], 0
        for i, v in enumerate(s):
            count = count + 1 if v > expanding_med.iloc[i] else 0
            result.append(count)
        return pd.Series(result, index=s.index)

    df["consecutive_high_load"] = (
        df.groupby("athlete_id")["training_duration"]
          .transform(_streak)
    )

    # 5. Load spike: z-score vs 28-day baseline
    mean28 = df.groupby("athlete_id")["training_duration"].transform(
        lambda x: x.rolling(28, min_periods=1).mean()
    )
    std28 = df.groupby("athlete_id")["training_duration"].transform(
        lambda x: x.rolling(28, min_periods=1).std().fillna(1)
    )
    df["load_spike"] = (df["training_duration"] - mean28) / (std28 + 1e-6)

    # 6. HRV trend
    df["hrv_trend_7d"] = (
        df.groupby("athlete_id", group_keys=False)
          .apply(lambda g: _rolling_slope(g["heart_rate_variability"], 7))
    )

    # 7. Sleep trend
    df["sleep_trend_7d"] = (
        df.groupby("athlete_id", group_keys=False)
          .apply(lambda g: _rolling_slope(g["sleep_hours"], 7))
    )

    # 8. Fatigue acceleration
    df["fatigue_diff1"] = df.groupby("athlete_id")["fatigue_level"].diff(1).fillna(0)
    df["fatigue_diff2"] = df.groupby("athlete_id")["fatigue_level"].diff(2).fillna(0)

    # 9. Recovery score: composite HRV + sleep + wellness (0-100)
    hrv_norm      = ((df["heart_rate_variability"].clip(20, 100) - 20) / 80) * 100
    sleep_norm    = (df["sleep_hours"].clip(3, 10) - 3) / 7 * 100
    wellness_norm = (df["wellness_score"] / 10) * 100
    df["recovery_score"] = (
        0.40 * hrv_norm + 0.35 * sleep_norm + 0.25 * wellness_norm
    ).clip(0, 100)

    # 10. Strain-recovery ratio
    df["strain_recovery_ratio"] = df["training_duration"] / (df["recovery_score"] + 1)

    # 11. Weekly load sum
    df["weekly_load_sum"] = (
        df.groupby("athlete_id")["training_duration"]
          .transform(lambda x: x.rolling(7, min_periods=1).sum())
    )

    # 12. Load monotony (Foster: mean/std of 7-day load)
    roll_mean7 = df.groupby("athlete_id")["training_duration"].transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    )
    roll_std7 = df.groupby("athlete_id")["training_duration"].transform(
        lambda x: x.rolling(7, min_periods=1).std().fillna(1)
    )
    df["load_monotony"] = roll_mean7 / (roll_std7 + 1e-6)

    # 13. Training strain (Foster TRIMP variant)
    df["training_strain"] = df["load_monotony"] * df["weekly_load_sum"]

    # 14. Peak load in last 14 days
    df["peak_load_14d"] = (
        df.groupby("athlete_id")["training_duration"]
          .transform(lambda x: x.rolling(14, min_periods=1).max())
    )

    # 15. Days since last rest
    def _days_since_rest(s: pd.Series) -> pd.Series:
        expanding_q30 = s.expanding().quantile(0.30)
        result, count = [], 0
        for i, v in enumerate(s):
            count = 0 if v <= expanding_q30.iloc[i] else count + 1
            result.append(count)
        return pd.Series(result, index=s.index)

    df["days_since_rest"] = (
        df.groupby("athlete_id")["training_duration"]
          .transform(_days_since_rest)
    )

    # 16. Accumulated load 7d
    df["_session_load"] = df["training_duration"] * df["intensity_rating"] / 10.0
    df["accumulated_load_7d"] = (
        df.groupby("athlete_id")["_session_load"]
          .transform(lambda x: x.rolling(7, min_periods=1).sum())
    )
    df = df.drop(columns=["_session_load"])

    # 17. HRV × Sleep
    df["hrv_sleep_interaction"] = df["heart_rate_variability"] * df["sleep_hours"]

    # 18. Load recovery balance
    df["load_recovery_balance"] = df["training_duration"] / (df["recovery_score"] + 1)

    # 19. Fatigue acceleration
    df["fatigue_acceleration"] = df.groupby("athlete_id")["fatigue_diff1"].diff(1).fillna(0)

    # 20. Intensity EWMA 7-day
    df["intensity_ewma_7"] = (
        df.groupby("athlete_id")["intensity_rating"]
          .transform(lambda x: x.ewm(span=7).mean())
    )

    # 21. Wellness / fatigue ratio
    df["wellness_fatigue_ratio"] = df["wellness_score"] / (df["fatigue_level"] + 0.1)

    return df


# ── Target ────────────────────────────────────────────────────────────────────

def create_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["athlete_id", "date"])
    df["injury_risk_score_next"] = (
        df.groupby("athlete_id")["injury_risk_score"].shift(-1)
    )
    return df


# ── Loader ────────────────────────────────────────────────────────────────────

def load_base_dataset() -> pd.DataFrame:
    csv_path = DATA_DIR / "synthetic" / "workload_data.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            "workload_data.csv not found. Run: python -m src.utils.data_generator"
        )
    return pd.read_csv(csv_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def build_dataset() -> pd.DataFrame:
    print("Loading dataset...")
    df = load_base_dataset()

    print("Running feature engineering (ACWR, EWMA, interactions, baseline)...")
    fe = FeatureEngineer(df)
    df = fe.run()

    # ── v4 OPTION C: NO video feature generation in the training pipeline ──
    # (the call to generate_video_features(df) was here in v3 — removed.)
    # Video influence is now handled separately via the rule-based
    # biomech_risk channel in HybridPredictor.

    print("Adding accuracy-boosting features (21 features)...")
    df = add_accuracy_features(df)

    print("Creating target variable...")
    df = create_target(df)

    df = df.dropna(subset=["injury_risk_score_next"])
    df = df.dropna()

    print(f"Final dataset: {len(df):,} rows × {len(df.columns)} columns")
    print(f"Target range:  {df['injury_risk_score_next'].min():.1f} – "
          f"{df['injury_risk_score_next'].max():.1f}  "
          f"(mean {df['injury_risk_score_next'].mean():.1f})")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print("Dataset saved to:", OUTPUT_FILE)
    return df


if __name__ == "__main__":
    build_dataset()
