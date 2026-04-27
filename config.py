# config.py — Centralised configuration for Athlete Injury Prediction System
#
# v4 — Option C refactor
#   - Two-channel architecture: workload-ML + biomechanical-rules
#   - CV_FEATURES no longer used in ML training pipeline (kept as a constant
#     only for legacy references); video influence flows through
#     BIOMECH_VIDEO_WEIGHT.
#   - Risk distribution recalibrated: synthetic data now spans 5–95.

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DATA_DIR    = BASE_DIR / "data"
MODEL_DIR   = BASE_DIR / "models" / "ml"
RESULTS_DIR = BASE_DIR / "results"

# Active data files
WORKLOAD_CSV      = DATA_DIR / "synthetic" / "workload_data.csv"
COMBINED_CSV      = DATA_DIR / "combined_dataset.csv"
LSTM_HISTORY_CSV  = RESULTS_DIR / "lstm_training_history.csv"

# Model artefacts
XGB_MODEL_PATH         = MODEL_DIR / "injury_model.pkl"
FEATURE_NAMES_PATH     = MODEL_DIR / "feature_names.pkl"
STACKED_MODEL_PATH     = MODEL_DIR / "stacked_injury_model.pkl"
LSTM_MODEL_PATH        = MODEL_DIR / "lstm_sequence_model.keras"
LSTM_META_PATH         = MODEL_DIR / "lstm_injury_model_meta.pkl"
TRANSFORMER_MODEL_PATH = MODEL_DIR / "transformer_sequence_model.keras"
TRANSFORMER_META_PATH  = MODEL_DIR / "transformer_meta.pkl"
HAZARD_MODEL_PATH      = MODEL_DIR / "hazard_model.pkl"

# ── Synthetic data generation ──────────────────────────────────────────────────
NUM_ATHLETES = 100         # number of synthetic athletes
NUM_DAYS     = 365         # days per athlete
DATA_SEED    = 42          # random seed for reproducibility

# ── Injury risk thresholds ─────────────────────────────────────────────────────
LOW_RISK_MAX    = 30       # score < 30  → "Low"   (green)
MEDIUM_RISK_MAX = 70       # score 30-70 → "Medium" (orange)
                           # score > 70  → "High"  (red)

def risk_cat(score: float) -> str:
    """Centralised risk categorisation.  Import this; don't redefine locally."""
    if score < LOW_RISK_MAX:
        return "Low"
    if score < MEDIUM_RISK_MAX:
        return "Medium"
    return "High"


def risk_color(score: float, palette: dict | None = None) -> str:
    default = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}
    p = palette or default
    return p[risk_cat(score)]


# ── ML training settings ───────────────────────────────────────────────────────
TEST_SIZE       = 0.20
RANDOM_STATE    = 42
SEQUENCE_LENGTH = 30
VAL_FRAC        = 0.20

# ── Feature lists ──────────────────────────────────────────────────────────────
WORKLOAD_FEATURES = [
    "training_duration",
    "heart_rate_variability",
    "running_distance",
    "sprint_count",
    "sleep_hours",
    "intensity_rating",
    "previous_injuries",
    "fatigue_level",
    "wellness_score",
]

# DEPRECATED in v4 — these columns are no longer part of the ML feature set.
# Kept as a constant only so legacy code that imports CV_FEATURES doesn't crash.
# Video influence is applied via the rule-based biomechanical risk channel.
CV_FEATURES = [
    "posture_symmetry",
    "balance_score",
    "movement_smoothness",
    "joint_variability",
    "knee_velocity",
    "hip_velocity",
    "shoulder_velocity",
    "movement_variability",
    "center_of_mass_stability",
]

ENGINEERED_FEATURES = [
    "acwr",
    "training_duration_ewma_7",
    "training_duration_ewma_14",
    "training_duration_ewma_28",
    "training_duration_std_7",
    "sleep_hours_std_7",
    "fatigue_level_std_7",
    "training_duration_diff1",
    "training_duration_diff2",
    "training_fatigue_interaction",
    "training_sleep_interaction",
    "hrv_fatigue_interaction",
    "acwr_fatigue_interaction",
    "sleep_debt",
    "fatigue_index",
    "heart_rate_variability_baseline_dev",
    "heart_rate_variability_baseline_z",
    "sleep_hours_baseline_dev",
    "sleep_hours_baseline_z",
    "fatigue_level_baseline_dev",
    "fatigue_level_baseline_z",
]

ACCURACY_FEATURES = [
    "session_of_season",
    "acwr_hrv",
    "acwr_spri",
    "injury_risk_trend_7d",
    "consecutive_high_load",
    "load_spike",
    "hrv_trend_7d",
    "sleep_trend_7d",
    "fatigue_diff1",
    "fatigue_diff2",
    "recovery_score",
    "strain_recovery_ratio",
    "weekly_load_sum",
    "load_monotony",
    "training_strain",
    "peak_load_14d",
    "days_since_rest",
    "accumulated_load_7d",
    "hrv_sleep_interaction",
    "load_recovery_balance",
    "fatigue_acceleration",
    "intensity_ewma_7",
    "wellness_fatigue_ratio",
]

TARGET = "injury_risk_score_next"

# ── Hazard model ───────────────────────────────────────────────────────────────
HAZARD_EVENT_PERCENTILE = 90

# ── Dashboard ──────────────────────────────────────────────────────────────────
DASHBOARD_TITLE     = "Athlete Injury Risk System"
DASHBOARD_ICON      = "⚽"
DASHBOARD_LAYOUT    = "wide"

# ── Hybrid blend weights (workload channel) ────────────────────────────────────
# These are the 4 weights inside the WORKLOAD-ML channel of the hybrid.
FALLBACK_BLEND_WEIGHTS = {
    "stack":       0.35,
    "lstm":        0.30,
    "transformer": 0.20,
    "hazard":      0.15,
}

# ── Two-channel hybrid blend (NEW in v4) ───────────────────────────────────────
#
#   final_risk = (1 - α_eff) * workload_risk + α_eff * biomech_risk
#   α_eff      = clamp(BIOMECH_VIDEO_WEIGHT * confidence,
#                      BIOMECH_ALPHA_MIN, BIOMECH_ALPHA_MAX)
#
# - No video uploaded         → confidence=0 → α_eff=0     → workload only
# - Full clean video uploaded → confidence≈1 → α_eff=0.40 → strong biomech voice
#
# Tune BIOMECH_VIDEO_WEIGHT to give the video channel more or less authority.
BIOMECH_VIDEO_WEIGHT = 0.40   # max contribution of biomech channel (0-1)
BIOMECH_ALPHA_MIN    = 0.00   # floor on α when video present (0 = optional)
BIOMECH_ALPHA_MAX    = 0.40   # absolute ceiling — never give video > 40%

# How much of the InjuryTypeClassifier's max risk to mix into biomech_risk
# (alongside the pure rule-based score). 0 = rules only, 1 = classifier only.
# Implements decision (c) from the design discussion: classifier reused as a
# secondary biomech signal, while remaining a per-injury display panel.
BIOMECH_CLASSIFIER_BLEND = 0.30
