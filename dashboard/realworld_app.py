"""
realworld_app.py  — Dashboard 2: Real-World Athlete Data
─────────────────────────────────────────────────────────
Enter real athlete session data, get injury risk predictions,
and track athlete history with full hybrid model breakdown.

Run from project root:
    streamlit run dashboard/realworld_app.py

Model architecture (v4 — two-channel hybrid)
────────────────────────────────────────────
Channel 1 (Workload-ML):
    Stacked Ensemble (XGB + LGB + CatBoost + ExtraTrees → Ridge)
  + LSTM Sequence Model (BiLSTM(64)→LSTM(32), 30-step per-athlete sequences)
  + Transformer Sequence Model (sinusoidal positional encoding, 2 blocks)
  + Random Survival Forest (Hazard)
  → workload_risk

Channel 2 (Biomechanical-Rules — NEW in v4):
    Rule-based score from MediaPipe pose statistics
    → biomech_risk (0-100)

Final = (1 - α_eff) * workload_risk + α_eff * biomech_risk
α_eff = BIOMECH_VIDEO_WEIGHT (0.40) × video confidence

Pages
─────
  👥 Athlete Manager    — Create / select / delete real athletes
  ➕ New Session        — Enter session data (workload + video + biometric CSV upload)
  🎯 Prediction Results — 5-model gauge, uncertainty band, injury types, recommendations
  📊 Athlete Dashboard  — Full history: trends, load, MQS, radar, risk velocity
  🏟️  Squad Overview    — All real athletes compared side-by-side
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, date
from pathlib import Path
from typing import Optional

# ── Paths ──────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent
SRC_DIR   = BASE_DIR / "src"
HIST_DIR  = BASE_DIR / "data" / "history"
META_FILE = BASE_DIR / "data" / "realworld" / "athletes_meta.json"
META_FILE.parent.mkdir(parents=True, exist_ok=True)
HIST_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(BASE_DIR))  # Also add project root so "from src.X import Y" resolves inside loaded modules

# ── Page config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Real-World Dashboard | Athlete Injury Risk",
    page_icon="🏟️",
    layout="wide",
)

# ── Colours ─────────────────────────────────────────────────────────
C = {
    "Low":    "#2ecc71",
    "Medium": "#f39c12",
    "High":   "#e74c3c",
    "Blue":   "#3498db",
    "Dark":   "#2c3e50",
    "Purple": "#9b59b6",
}

SPORTS    = ["Tennis", "Badminton", "Running",
             "Squat / Strength", "Generic"]
POSITIONS = ["Forward","Midfielder","Defender","Goalkeeper",
             "Sprinter","Distance","All-round","N/A"]

# ══════════════════════════════════════════════════════════════════
# STREAMLIT VERSION COMPATIBILITY
# ══════════════════════════════════════════════════════════════════

def _rerun():
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()   # type: ignore[attr-defined]

# ══════════════════════════════════════════════════════════════════
# UTILITY
# ══════════════════════════════════════════════════════════════════

def risk_cat(s: float) -> str:
    return "Low" if s < 30 else ("Medium" if s < 70 else "High")

def compute_mqs(row) -> float:
    def sg(k, d): return float(row.get(k, d) or d)
    stab = max(0, min(100, 100 - (sg("knee_angle_left_std",5) + sg("knee_angle_right_std",5)) * 1.8))
    mqs  = (stab                                    * 0.25 +
            sg("posture_symmetry_mean",   0.8)*100  * 0.20 +
            sg("balance_stability_mean",  0.8)*100  * 0.20 +
            sg("movement_fluidity_mean",  0.8)*100  * 0.20 +
            sg("shoulder_symmetry_mean",  0.8)*100  * 0.10 +
            max(0, min(100, 100 - sg("torso_lean_mean",0.1)*150)) * 0.05)
    return round(float(np.clip(mqs, 0, 100)), 1)

def mqs_grade(s: float) -> str:
    if s >= 90: return "Excellent ✅"
    if s >= 70: return "Good 🟢"
    if s >= 50: return "Needs Improvement 🟡"
    return "Injury Risk Zone 🔴"

def injury_types_for_row(row, acwr_val: float = 1.0) -> list:
    try:
        from cv.injury_type_classifier import InjuryTypeClassifier
        w = {k: float(row.get(k, d)) for k, d in [
            ("training_duration", 90), ("heart_rate_variability", 60),
            ("running_distance", 8),   ("sprint_count", 10),
            ("sleep_hours", 7),        ("intensity_rating", 5),
            ("previous_injuries", 0),  ("fatigue_level", 5),
            ("wellness_score", 7),
        ]}
        w["acwr"] = acwr_val
        kls  = float(row.get("knee_angle_left_std",  5))
        krs  = float(row.get("knee_angle_right_std", 5))
        bsym = float(row.get("body_symmetry_score_mean", 0.8))
        v = {
            "knee_angle_left_std":         kls,
            "knee_angle_right_std":        krs,
            "landing_risk_index":          (1-bsym)*50 + max(kls,krs)*0.5,
            "body_symmetry_score_mean":    bsym,
            "posture_symmetry_mean":       float(row.get("posture_symmetry_mean",    0.8)),
            "balance_stability_mean":      float(row.get("balance_stability_mean",   0.8)),
            "movement_fluidity_mean":      float(row.get("movement_fluidity_mean",   0.8)),
            "shoulder_symmetry_mean":      float(row.get("shoulder_symmetry_mean",   0.8)),
            "torso_lean_mean":             float(row.get("torso_lean_mean",          0.1)),
            "elbow_angle_left_std":        float(row.get("elbow_angle_left_std",     5)),
            "elbow_angle_right_std":       float(row.get("elbow_angle_right_std",    5)),
            "running_knee_angle_left_std": kls,
        }
        return InjuryTypeClassifier().classify(w, v)
    except Exception:
        return []

# ══════════════════════════════════════════════════════════════════
# VIDEO BIOMECHANICS EXTRACTOR
# ══════════════════════════════════════════════════════════════════

@st.cache_resource
def _get_pose_estimator():
    try:
        from cv.pose_estimator import PoseEstimator
        return PoseEstimator(confidence_threshold=0.5)
    except Exception:
        return None


def extract_biomechanics_from_video(uploaded_file) -> tuple[dict, str]:
    estimator = _get_pose_estimator()
    if estimator is None:
        return {}, "❌ PoseEstimator could not load (check mediapipe / opencv install)"

    suffix = Path(uploaded_file.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        raw = estimator.process_video(tmp_path, skip_frames=2)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if raw is None:
        return {}, "❌ No pose landmarks detected in this video. Check lighting and that the full body is visible."

    n_frames = raw.get("frames_processed", 0)
    if n_frames < 10:
        return {}, f"⚠️ Only {n_frames} usable frames detected — video may be too short or poorly lit."

    # v4 OPTION C: return RAW pose statistics only. NO mapping to fake CV
    # feature names (posture_symmetry / knee_velocity / etc.) — the ML model
    # is no longer trained on those columns. Real video influence is applied
    # downstream via HybridPredictor's biomech channel.
    features = {
        "posture_symmetry_mean":      raw.get("posture_symmetry_mean",     0.85),
        "posture_symmetry_std":       raw.get("posture_symmetry_std",      0.05),
        "balance_stability_mean":     raw.get("balance_stability_mean",    0.85),
        "balance_stability_std":      raw.get("balance_stability_std",     0.05),
        "movement_fluidity_mean":     raw.get("movement_fluidity_mean",    0.85),
        "movement_fluidity_std":      raw.get("movement_fluidity_std",     0.05),
        "shoulder_symmetry_mean":     raw.get("shoulder_symmetry_mean",    0.85),
        "shoulder_symmetry_std":      raw.get("shoulder_symmetry_std",     0.05),
        "torso_lean_mean":            raw.get("torso_lean_mean",           0.10),
        "torso_lean_std":             raw.get("torso_lean_std",            0.02),
        "body_symmetry_score_mean":   raw.get("body_symmetry_score_mean",  0.85),
        "body_symmetry_score_std":    raw.get("body_symmetry_score_std",   0.05),
        "knee_angle_left_mean":       raw.get("knee_angle_left_mean",      160.0),
        "knee_angle_right_mean":      raw.get("knee_angle_right_mean",     160.0),
        "knee_angle_left_std":        raw.get("knee_angle_left_std",       5.0),
        "knee_angle_right_std":       raw.get("knee_angle_right_std",      5.0),
        "elbow_angle_left_std":       raw.get("elbow_angle_left_std",      5.0),
        "elbow_angle_right_std":      raw.get("elbow_angle_right_std",     5.0),
        "movement_quality_score":     raw.get("movement_quality_score",    70.0),
        # KEY for biomech channel (read by compute_biomech_risk)
        "frames_processed":           n_frames,
        "_frames_processed":          n_frames,  # back-compat
    }

    status = (
        f"✅ Analysed **{n_frames} frames** via MediaPipe.  "
        f"MQS: **{features['movement_quality_score']:.1f}/100** ({estimator.mqs_grade(features['movement_quality_score'])})"
    )
    return features, status

# ══════════════════════════════════════════════════════════════════
# ATHLETE META STORE
# ══════════════════════════════════════════════════════════════════

def load_athletes_meta() -> dict:
    if META_FILE.exists():
        try: return json.loads(META_FILE.read_text())
        except Exception: return {}
    return {}

def save_athletes_meta(meta: dict):
    META_FILE.write_text(json.dumps(meta, indent=2))

def next_athlete_id(meta: dict) -> int:
    if not meta: return 1
    return max(int(k) for k in meta.keys()) + 1

# ══════════════════════════════════════════════════════════════════
# SESSION HISTORY
# ══════════════════════════════════════════════════════════════════

def history_path(aid: int) -> Path:
    return HIST_DIR / f"athlete_{aid}.csv"

def load_history(aid: int) -> pd.DataFrame:
    p = history_path(aid)
    if not p.exists(): return pd.DataFrame()
    try:
        df = pd.read_csv(p, parse_dates=["date"])
        return df.sort_values("date").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()

def save_session(aid: int, session: dict) -> pd.DataFrame:
    """
    Append a new session to the athlete's history CSV.

    NOTE on the `injury_risk_score` self-loop:
    The model's predicted risk for this session is saved back into
    `session["injury_risk_score"]` (see the New Session page where this
    function is called). On the *next* prediction, RealtimeFeatureBuilder
    runs `add_accuracy_features`, which computes `injury_risk_trend_7d` from
    the rolling slope of past `injury_risk_score` values — and those past
    values are themselves model outputs.

    This is a known divergence from training: during training the trend
    feature was computed from simulated *ground-truth* risk; at inference it
    is computed from the model's *own* past predictions. Trends will look
    smoother than they did in training because the model's outputs vary less
    than the simulated truth. There is no clean fix without either
    (a) collecting real injury-event labels, or (b) dropping the trend
    feature altogether and retraining. Documenting it here so future readers
    know why predictions seem to anchor near 50 for a few sessions before
    the trend starts to move.
    """
    existing = load_history(aid)
    new_row  = pd.DataFrame([session])
    new_row["date"] = pd.to_datetime(new_row["date"])
    updated  = pd.concat([existing, new_row], ignore_index=True)
    updated  = updated.sort_values("date").reset_index(drop=True)
    updated.to_csv(history_path(aid), index=False)
    return updated

def delete_last_session(aid: int):
    df = load_history(aid)
    if not df.empty:
        df = df.iloc[:-1]
        df.to_csv(history_path(aid), index=False)

# ══════════════════════════════════════════════════════════════════
# MODEL LOADER
# FIX: register SinusoidalPositionalEncoding as a custom object so
# Keras can deserialise the transformer model after the training fix
# ══════════════════════════════════════════════════════════════════

@st.cache_resource
def load_predictor():
    try:
        from ml.hybrid_predictor import HybridPredictor
        return HybridPredictor()
    except Exception as e:
        st.sidebar.warning(f"Model load issue: {e}")
        return None

predictor = load_predictor()

# ── Helper: get actual blend weights from predictor ────────────────
def get_blend_weights() -> dict:
    if predictor is not None and hasattr(predictor, "_weights"):
        w = predictor._weights
        return {
            "Ensemble":    round(float(w[0]) * 100, 1),
            "LSTM":        round(float(w[1]) * 100, 1),
            "Transformer": round(float(w[2]) * 100, 1),
            "Hazard":      round(float(w[3]) * 100, 1),
        }
    return {"Ensemble": 35.0, "LSTM": 30.0, "Transformer": 20.0, "Hazard": 15.0}

# ══════════════════════════════════════════════════════════════════
# FEATURE BUILDER + PREDICT
# FIX: NaN crash — was concat-ing raw history with a single engineered
# row, leaving NaN in engineered columns of historical rows. ExtraTrees
# and RSF rejected those NaNs and fell back to 50.0. Fixed by using the
# full-history mode of RealtimeFeatureBuilder.
# ══════════════════════════════════════════════════════════════════

def build_features_and_predict(
    session_dict: dict,
    history_df: pd.DataFrame,
    mc_samples: int = 1,
    video_features: dict | None = None,
    classifier_results: list | None = None,
    biomech_sport: str | None = None,
):
    """
    Build engineered features for the session, combine with history,
    and run the two-channel HybridPredictor.

    v4: now passes video_features and classifier_results into predictor.predict()
    so the biomech-rules channel actually contributes to the final risk score.

    Args:
        session_dict:       raw session data
        history_df:         previous sessions for this athlete
        mc_samples:         MC-Dropout samples for uncertainty (1 = deterministic)
        video_features:     dict of pose statistics (from extract_biomechanics_from_video)
                            or None if no video uploaded.
        classifier_results: list of InjuryTypeRisk from InjuryTypeClassifier,
                            blended into biomech_risk per BIOMECH_CLASSIFIER_BLEND.
        biomech_sport:      Sport label/key for sport-specific biomech rule
                            thresholds (Phase B). Accepts "Tennis", "Badminton",
                            "Running", "Squat / Strength", "Generic", or any
                            of the underlying keys. Default = "generic".

    Returns:
        (result_dict, acwr_value, feature_row_df)
    """
    full_engineered = pd.DataFrame()
    feat_row        = pd.DataFrame()

    try:
        from realtime.feature_builder import RealtimeFeatureBuilder
        builder = RealtimeFeatureBuilder()
        full_engineered = builder.build_features(
            session_dict, history_df, return_full_history=True
        )
        feat_row = full_engineered.iloc[[-1]].copy()
    except Exception as e:
        st.warning(f"Feature builder error: {e}")
        feat_row = pd.DataFrame([session_dict])
        full_engineered = feat_row.copy()

    if not full_engineered.empty:
        full_engineered = full_engineered.fillna(0)

    result = {
        "ensemble": None, "lstm": None, "transformer": None, "hazard": None,
        "workload_risk": 50.0, "biomech_risk": 0.0, "biomech_confidence": 0.0,
        "biomech_rules": [], "biomech_components": {}, "biomech_n_frames": 0,
        "alpha_video": 0.0, "final_risk_score": 50.0,
        "lstm_std": 0.0, "transformer_std": 0.0, "uncertainty_band": 0.0,
        "effective_weights": {"stack": 0.35, "lstm": 0.30, "transformer": 0.20,
                              "hazard": 0.15, "biomech": 0.0},
    }
    if predictor is not None and not full_engineered.empty:
        try:
            # Forward sport selection through to predictor.predict() if it
            # accepts it; older predictors fall back to default thresholds.
            kwargs = dict(
                mc_samples=mc_samples,
                video_features=video_features,
                classifier_results=classifier_results,
            )
            try:
                result = predictor.predict(
                    full_engineered, biomech_sport=biomech_sport, **kwargs,
                )
            except TypeError:
                # Old HybridPredictor without biomech_sport kwarg
                result = predictor.predict(full_engineered, **kwargs)
        except Exception as e:
            st.warning(f"Predictor error: {e}")

    # Compute ACWR for display
    acwr_val = 1.0
    if not feat_row.empty and "acwr" in feat_row.columns:
        try:
            acwr_val = float(feat_row["acwr"].values[0])
        except (ValueError, TypeError):
            acwr_val = 1.0
    elif not history_df.empty and "training_duration" in history_df.columns:
        dur = pd.concat([
            history_df[["training_duration"]],
            pd.DataFrame([{"training_duration": session_dict.get("training_duration", 90)}]),
        ], ignore_index=True)["training_duration"]
        acwr_val = float(dur.tail(7).mean() / (dur.tail(28).mean() + 1e-9))

    return result, acwr_val, feat_row

# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════

meta = load_athletes_meta()
bw   = get_blend_weights()

with st.sidebar:
    st.markdown("## 🏟️ Real-World Dashboard")
    st.caption("Enter real athlete data · Full hybrid model predictions")
    st.divider()

    page = st.radio("Navigate to", [
        "👥 Athlete Manager",
        "➕ New Session",
        "🎯 Prediction Results",
        "📊 Athlete Dashboard",
        "🏟️ Squad Overview",
    ])

    st.divider()
    if meta:
        options = {f"{v['name']} (ID {k})": k for k, v in meta.items()}
        sel = st.selectbox("Active Athlete", list(options.keys()))
        st.session_state["active_athlete_id"]   = int(options[sel])
        st.session_state["active_athlete_meta"] = meta[options[sel]]
    else:
        st.info("No athletes yet. Go to 'Athlete Manager' to create one.")
        st.session_state["active_athlete_id"]   = None
        st.session_state["active_athlete_meta"] = None

    st.divider()
    with st.expander("🤖 Model Architecture (2-channel)"):
        st.write(f"Stacked Ensemble:  {'✅ Loaded' if predictor else '❌ Missing'}")
        st.write(f"LSTM Sequence:     {'✅ Loaded' if predictor else '❌ Missing'}")
        st.write(f"Transformer:       {'✅ Loaded' if predictor else '❌ Missing'}")
        st.write(f"Hazard Survival:   {'✅ Loaded' if predictor else '❌ Missing'}")
        try:
            from cv.biomech_risk_module import compute_biomech_risk as _b
            biomech_ok = _b is not None
        except Exception:
            biomech_ok = False
        st.write(f"Biomech Rules:     {'✅ Loaded' if biomech_ok else '❌ Missing'}")
        if predictor:
            st.info(
                f"**Channel 1 — Workload-ML:**  \n"
                f"Ensemble {bw['Ensemble']}% · LSTM {bw['LSTM']}%  \n"
                f"Transformer {bw['Transformer']}% · Hazard {bw['Hazard']}%  \n\n"
                f"**Channel 2 — Biomech:**  \n"
                f"α_max = 40% (scaled by video confidence)"
            )
            st.write(f"✅ Per-athlete sequences")
            st.write(f"✅ Sinusoidal PE (Transformer)")
            st.write(f"✅ Rule-based biomech channel")
        else:
            st.info("Active: **No model loaded**")

    pose_ok = _get_pose_estimator() is not None
    with st.expander("🎥 Video Analysis"):
        st.write(f"PoseEstimator (MediaPipe): {'✅ Ready' if pose_ok else '❌ Not available'}")
        if not pose_ok:
            st.caption("Install: `pip install mediapipe opencv-python`")

    # ── Biomech channel weight slider (live tuning) ──────────────────
    # Calls hybrid_predictor.set_biomech_weight() so the next predict()
    # call uses the new α target without restarting the app. The default
    # is BIOMECH_VIDEO_WEIGHT (0.40) from config.py.
    with st.expander("🎚️ Biomech (video) weight α"):
        try:
            from src.ml.hybrid_predictor import (
                set_biomech_weight, get_biomech_weight,
            )
            current = get_biomech_weight()
        except Exception:
            set_biomech_weight = None
            current = 0.40

        new_w = st.slider(
            "Video influence on final risk score",
            min_value=0.00, max_value=0.60, value=float(current), step=0.05,
            help=(
                "How much weight the biomechanical channel gets in the final "
                "blend, when video is present and confidence is high. "
                "Default 0.40 = video accounts for up to 40 % of the final "
                "score. Set to 0 to disable the video channel entirely. "
                "Above 0.6, video starts to dominate and small pose-detection "
                "errors swing the risk score significantly."
            ),
        )
        if set_biomech_weight is not None and abs(new_w - current) > 1e-9:
            set_biomech_weight(new_w)
        st.caption(
            f"Effective max α = {new_w:.2f}  ·  "
            f"actual α applied = α_max × video_confidence"
        )

    st.caption(
        "ℹ️ **Training note:** Models trained on synthetic data (100 athletes, 365 days). "
        "No retraining needed until ~1 000+ real sessions."
    )

active_id   = st.session_state.get("active_athlete_id")
active_meta = st.session_state.get("active_athlete_meta")

# ══════════════════════════════════════════════════════════════════
# PAGE 1 — ATHLETE MANAGER
# ══════════════════════════════════════════════════════════════════
if page == "👥 Athlete Manager":
    st.title("👥 Athlete Manager")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("➕ Create New Athlete")
        with st.form("create_athlete"):
            name   = st.text_input("Full Name *")
            age    = st.number_input("Age", 14, 50, 22)
            sport  = st.selectbox("Sport", SPORTS)
            pos    = st.selectbox("Position", POSITIONS)
            height = st.number_input("Height (cm)", 140, 220, 175)
            weight = st.number_input("Weight (kg)",  40, 150,  75)
            notes  = st.text_area("Notes (injuries, history, etc.)", height=70)
            sub    = st.form_submit_button("Create Athlete", type="primary")

        if sub:
            if not name.strip():
                st.error("Name is required.")
            else:
                meta = load_athletes_meta()
                aid  = next_athlete_id(meta)
                meta[str(aid)] = {
                    "name": name.strip(), "age": age, "sport": sport,
                    "position": pos, "height_cm": height, "weight_kg": weight,
                    "notes": notes, "created": str(datetime.now().date()),
                }
                save_athletes_meta(meta)
                st.success(f"✅ Athlete '{name}' created with ID {aid}.")
                _rerun()

    with c2:
        st.subheader("📋 Current Roster")
        meta = load_athletes_meta()
        if meta:
            roster = []
            for aid, m in meta.items():
                hist = load_history(int(aid))
                roster.append({
                    "ID":       int(aid),
                    "Name":     m["name"],
                    "Age":      m["age"],
                    "Sport":    m["sport"],
                    "Sessions": len(hist),
                    "Last Session": str(hist["date"].max().date()) if not hist.empty and "date" in hist.columns else "—",
                })
            st.dataframe(pd.DataFrame(roster), use_container_width=True, hide_index=True)

            st.subheader("🗑️ Delete Athlete")
            del_opt = {f"{v['name']} (ID {k})": k for k, v in meta.items()}
            del_sel = st.selectbox("Select to delete", list(del_opt.keys()))
            if st.button("Delete Athlete & History", type="secondary"):
                did = del_opt[del_sel]
                meta.pop(did, None)
                save_athletes_meta(meta)
                hp = history_path(int(did))
                if hp.exists(): hp.unlink()
                st.success("Deleted.")
                _rerun()
        else:
            st.info("No athletes yet. Create one on the left.")

# ══════════════════════════════════════════════════════════════════
# PAGE 2 — NEW SESSION
# ══════════════════════════════════════════════════════════════════
elif page == "➕ New Session":
    st.title("➕ Enter New Session")

    if not active_id:
        st.warning("No athlete selected. Go to Athlete Manager first.")
        st.stop()

    st.subheader(f"Athlete: **{active_meta['name']}** | {active_meta['sport']} | Age {active_meta['age']}")
    hist = load_history(active_id)
    st.caption(f"Sessions recorded: {len(hist)}")

    # ── Uncertainty settings ──────────────────────
    with st.expander("⚙️ Prediction Settings"):
        mc_samples = st.slider(
            "MC-Dropout samples for uncertainty estimation",
            min_value=1, max_value=50, value=1,
            help="1 = fast deterministic. 20-50 = uncertainty band (±). Requires ~2s extra per model."
        )
        if mc_samples > 1:
            st.caption(f"Running {mc_samples} stochastic forward passes. Results include a confidence band (±).")

    with st.form("new_session"):
        st.markdown("### 📅 Session Details")
        session_date = st.date_input("Session Date", value=date.today())

        st.markdown("### 💪 Workload Metrics")
        w1, w2, w3 = st.columns(3)
        with w1:
            training_duration = st.number_input("Training Duration (min)", 0.0, 300.0, 90.0, step=5.0)
            running_distance  = st.number_input("Running Distance (km)",    0.0,  30.0,  8.0, step=0.5)
            sprint_count      = st.number_input("Sprint Count",             0,    60,    10)
        with w2:
            intensity_rating = st.slider("Intensity Rating (1-10)", 1.0, 10.0, 6.0, step=0.5)
            fatigue_level    = st.slider("Fatigue Level (1-10)",    1.0, 10.0, 5.0, step=0.5)
            wellness_score   = st.slider("Wellness Score (1-10)",   1.0, 10.0, 7.0, step=0.5)
        with w3:
            heart_rate_variability = st.number_input("HRV", 10.0, 120.0, 55.0, step=1.0)
            sleep_hours            = st.number_input("Sleep Last Night (h)",  3.0,  12.0,  7.5, step=0.25)
            previous_injuries      = st.number_input("Previous Injury Count", 0,    10,     0)

        # ── VIDEO BIOMECHANICS ─────────────────────────────────────
        st.markdown("### 🎥 Training Video — Biomechanics Extraction")
        st.caption(
            "Upload a training or movement video (MP4, MOV, AVI, MKV, WebM). "
            "MediaPipe will automatically extract joint angles, symmetry, balance and fluidity. "
            "Maximum upload size: **2 GB** (set in `.streamlit/config.toml`). "
            "Full-body visibility and good lighting give the best results."
        )

        video_file = st.file_uploader(
            "Upload training video",
            type=["mp4", "mov", "avi", "mkv", "webm"],
            help="Full-body video works best. Ensure good lighting and that the whole body is visible throughout.",
        )

        # ── Phase B: sport context for biomech-rule thresholds ──────────
        # Different sports have different healthy ranges for knee variability,
        # forward lean, etc. Tennis legitimately produces 28° knee std from
        # lunging — that's not an ACL signal. Running with 28° knee std IS.
        # The selector below routes through to BiomechRiskAssessor's sport
        # threshold table. Pre-filled from the athlete's sport metadata.
        from src.cv.biomech_risk_module import SPORT_LABEL_TO_KEY
        BIOMECH_SPORT_LABELS = list(SPORT_LABEL_TO_KEY.keys())

        athlete_sport_raw = (active_meta or {}).get("sport", "") or ""
        athlete_sport_l   = athlete_sport_raw.lower()
        # Pre-fill: match the athlete's sport string against our 5 sports
        prefill_idx = BIOMECH_SPORT_LABELS.index("Generic")
        for i, label in enumerate(BIOMECH_SPORT_LABELS):
            key = SPORT_LABEL_TO_KEY[label]
            if key in athlete_sport_l or athlete_sport_l in key or label.lower() in athlete_sport_l:
                prefill_idx = i
                break

        biomech_sport_label = st.selectbox(
            "Sport context for biomech analysis",
            options=BIOMECH_SPORT_LABELS,
            index=prefill_idx,
            help=(
                "Each sport has its own healthy range for knee variability, "
                "torso lean, etc. Tennis allows ~35° knee std (lunging is "
                "normal); Running expects ≤18°. Picking the wrong sport "
                "here either over-flags healthy motion (e.g. tennis under "
                "Running) or under-flags real concerns. Defaults to the "
                "athlete's registered sport, but can be overridden per "
                "video. If the sport isn't listed, choose Generic."
            ),
            key="biomech_sport_select",
        )

        # ── BIOMETRIC CSV UPLOAD ───────────────────────────────────
        st.markdown("### 📊 Biometric Data — Upload CSV")
        st.caption(
            "Upload a CSV exported from a wearable device (Garmin, Whoop, Oura, etc.) "
            "or any spreadsheet with biometric columns. Supported columns: "
            "`heart_rate_variability`, `sleep_hours`, `fatigue_level`, `wellness_score`, "
            "`training_duration`, `running_distance`, `sprint_count`, `intensity_rating`. "
            "Matching columns will override the manually entered values above."
        )

        biometric_csv = st.file_uploader(
            "Upload biometric CSV",
            type=["csv"],
            help="CSV with one row per session or a single-row summary. Column names are matched case-insensitively.",
        )

        session_notes = st.text_area("Session Notes")
        submit = st.form_submit_button("💾 Save Session & Predict", type="primary")

    # ── VIDEO PROCESSING ─────────────────────────────────────────
    bio_data   = {}
    bio_status = ""

    if video_file is not None:
        with st.spinner(f"🎥 Analysing video with MediaPipe ({video_file.size / 1_048_576:.1f} MB)…"):
            bio_data, bio_status = extract_biomechanics_from_video(video_file)

    # ── BIOMETRIC CSV PROCESSING ──────────────────────────────────
    csv_overrides = {}
    if biometric_csv is not None:
        try:
            bio_csv_df = pd.read_csv(biometric_csv)
            # Normalise column names: lowercase, strip whitespace, replace spaces with _
            bio_csv_df.columns = [c.strip().lower().replace(" ", "_") for c in bio_csv_df.columns]
            # Use the last row (most recent session) if multiple rows
            csv_row = bio_csv_df.iloc[-1].to_dict()
            BIOMETRIC_COLS = [
                "heart_rate_variability", "sleep_hours", "fatigue_level",
                "wellness_score", "training_duration", "running_distance",
                "sprint_count", "intensity_rating", "previous_injuries",
            ]
            matched = []
            for col in BIOMETRIC_COLS:
                if col in csv_row and pd.notna(csv_row[col]):
                    try:
                        csv_overrides[col] = float(csv_row[col])
                        matched.append(col)
                    except (ValueError, TypeError):
                        pass
            if matched:
                st.success(f"✅ Biometric CSV loaded — {len(matched)} columns matched: `{'`, `'.join(matched)}`")
                ovr_df = pd.DataFrame([{k: round(v, 2) for k, v in csv_overrides.items()}])
                st.dataframe(ovr_df, use_container_width=True, hide_index=True)
            else:
                st.warning("⚠️ Biometric CSV loaded but no matching columns found. Check column names.")
        except Exception as e:
            st.warning(f"⚠️ Could not parse biometric CSV: {e}")

    if video_file is not None:
        if bio_status.startswith("✅"):
            st.success(bio_status)

            display_keys = {
                "posture_symmetry_mean":    "Posture Symmetry",
                "balance_stability_mean":   "Balance Stability",
                "body_symmetry_score_mean": "Body Symmetry",
                "movement_fluidity_mean":   "Movement Fluidity",
                "shoulder_symmetry_mean":   "Shoulder Symmetry",
                "torso_lean_mean":          "Torso Lean",
                "knee_angle_left_mean":     "Knee Angle Left (°)",
                "knee_angle_right_mean":    "Knee Angle Right (°)",
                "knee_angle_left_std":      "Knee L Variability",
                "knee_angle_right_std":     "Knee R Variability",
            }
            rows = []
            for key, label in display_keys.items():
                val = bio_data.get(key)
                if val is not None:
                    rows.append({"Feature": label, "Value": round(val, 3)})

            if rows:
                b_df = pd.DataFrame(rows)
                st.markdown("**Extracted Biomechanical Features:**")

                def colour_val(v, key=""):
                    if "angle" in key.lower() or "variability" in key.lower():
                        return f"📐 {v:.3f}"
                    if v > 0.8:  return f"🟢 {v:.3f}"
                    if v > 0.6:  return f"🟡 {v:.3f}"
                    return f"🔴 {v:.3f}"

                b_df["Status"] = [
                    colour_val(r["Value"], r["Feature"]) if r["Value"] <= 1.0 else f"📐 {r['Value']:.1f}"
                    for r in rows
                ]
                st.dataframe(b_df[["Feature","Status"]], use_container_width=True, hide_index=True)

                mqs_v = bio_data.get("movement_quality_score", compute_mqs(bio_data))
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("🎯 Movement Quality Score", f"{mqs_v:.1f}/100", mqs_grade(mqs_v))
                mc2.metric("📐 Knee Variability (L)", f"{bio_data.get('knee_angle_left_std',0):.1f}°",
                           "Low = stable" if bio_data.get('knee_angle_left_std',0) < 8 else "High — review mechanics")
                mc3.metric("📐 Knee Variability (R)", f"{bio_data.get('knee_angle_right_std',0):.1f}°",
                           "Low = stable" if bio_data.get('knee_angle_right_std',0) < 8 else "High — review mechanics")
        else:
            if bio_status:
                st.warning(bio_status)

    if submit:
        session = {
            "date":                   str(session_date),
            "training_duration":      training_duration,
            "heart_rate_variability": heart_rate_variability,
            "running_distance":       running_distance,
            "sprint_count":           sprint_count,
            "sleep_hours":            sleep_hours,
            "intensity_rating":       intensity_rating,
            "previous_injuries":      previous_injuries,
            "fatigue_level":          fatigue_level,
            "wellness_score":         wellness_score,
            "session_notes":          session_notes,
        }
        # Apply biometric CSV overrides (wearable data takes precedence over sliders)
        if csv_overrides:
            session.update(csv_overrides)

        # Merge raw video pose statistics into the saved session record
        # (skip private keys starting with _). These are NOT ML features any
        # more — they are stored for display + the biomech-rules channel.
        if bio_data:
            session.update({k: v for k, v in bio_data.items() if not k.startswith("_")})

        hist_df = load_history(active_id)

        # ── Duplicate-date guard ──────────────────────────────────────
        duplicate_date_count = 0
        if not hist_df.empty and "date" in hist_df.columns:
            try:
                existing_dates = pd.to_datetime(hist_df["date"]).dt.date
                duplicate_date_count = int((existing_dates == session_date).sum())
            except Exception:
                duplicate_date_count = 0
        if duplicate_date_count > 0:
            st.warning(
                f"⚠️  This athlete already has **{duplicate_date_count}** "
                f"session(s) on {session_date}. Time-based charts will collapse "
                f"if every session shares the same date — vary the date for "
                f"meaningful trend visualisation."
            )

        # ── v4: compute classifier_results + pass video into predict() ──
        # The InjuryTypeClassifier runs on the SAME video features, and its
        # max risk_score is blended into biomech_risk per BIOMECH_CLASSIFIER_BLEND.
        # If no video, we still compute classifier results from workload only —
        # but the rule-based biomech score will be 0 (no video keys present).
        video_features_for_predict = bio_data if bio_data else None

        # Compute ACWR estimate for the classifier (uses 7d/28d rolling means)
        acwr_for_cls = 1.0
        if not hist_df.empty and "training_duration" in hist_df.columns:
            dur_series = pd.concat([
                hist_df[["training_duration"]],
                pd.DataFrame([{"training_duration": training_duration}]),
            ], ignore_index=True)["training_duration"]
            acwr_for_cls = float(dur_series.tail(7).mean() /
                                 (dur_series.tail(28).mean() + 1e-9))

        classifier_results = injury_types_for_row(session, acwr_for_cls) if bio_data else None

        result, acwr_val, feat_row = build_features_and_predict(
            session, hist_df,
            mc_samples=mc_samples,
            video_features=video_features_for_predict,
            classifier_results=classifier_results,
            biomech_sport=biomech_sport_label,
        )

        final_score = result["final_risk_score"]
        session["injury_risk_score"] = final_score
        session["acwr"]              = acwr_val
        if bio_data:
            session["has_video_biomechanics"] = True
            # FIX: persist MQS so it survives reload (was nan/100 on revisit)
            session["movement_quality_score"] = bio_data.get(
                "movement_quality_score",
                compute_mqs(bio_data),
            )

        save_session(active_id, session)

        st.session_state["last_prediction"] = {
            "session":     session,
            "result":      result,
            "acwr":        acwr_val,
            "feat_row":    feat_row.to_dict(orient="records")[0] if not feat_row.empty else {},
            "history_len": len(hist_df),
            "has_bio":     bool(bio_data),
            "mc_samples":  mc_samples,
        }

        st.success("✅ Session saved! Go to **🎯 Prediction Results** to see the output.")

        cat_ = risk_cat(final_score)
        uncertainty = result.get("uncertainty_band", 0.0)
        alpha_v     = result.get("alpha_video", 0.0)
        biomech_r   = result.get("biomech_risk", 0.0)
        workload_r  = result.get("workload_risk", final_score)

        uncertainty_str = f"  |  Uncertainty: **±{uncertainty:.1f}**" if mc_samples > 1 and uncertainty > 0 else ""
        if bio_data and alpha_v > 0:
            blend_str = (
                f"  |  Workload: **{workload_r:.1f}** · "
                f"Biomech: **{biomech_r:.1f}** "
                f"(α_video={alpha_v*100:.0f}%)"
            )
        else:
            blend_str = ""

        st.markdown(
            f"<div style='padding:15px;background:{C[cat_]}22;border-left:5px solid {C[cat_]};border-radius:6px;'>"
            f"<h3>Predicted Risk: <strong style='color:{C[cat_]};'>{final_score:.1f}/100 — {cat_}</strong></h3>"
            f"ACWR: <strong>{acwr_val:.2f}</strong>"
            + (f"  |  🎥 Video biomechanics included" if bio_data else "")
            + blend_str
            + uncertainty_str
            + "</div>",
            unsafe_allow_html=True,
        )

# ══════════════════════════════════════════════════════════════════
# PAGE 3 — PREDICTION RESULTS
# ══════════════════════════════════════════════════════════════════
elif page == "🎯 Prediction Results":
    st.title("🎯 Prediction Results")

    if "last_prediction" not in st.session_state:
        st.info("No prediction yet. Enter a session on the **➕ New Session** page.")
        st.stop()

    pred       = st.session_state["last_prediction"]
    sess       = pred["session"]
    result     = pred["result"]
    acwr       = pred["acwr"]
    hist_len   = pred["history_len"]
    has_bio    = pred.get("has_bio", False)
    mc_samples = pred.get("mc_samples", 1)

    ensemble   = result.get("ensemble")
    lstm_val   = result.get("lstm")
    transf_val = result.get("transformer")
    hazard_val = result.get("hazard")
    final_val  = result.get("final_risk_score", 50.0)
    workload_r = result.get("workload_risk", final_val)
    biomech_r  = result.get("biomech_risk", 0.0)
    biomech_conf = result.get("biomech_confidence", 0.0)
    biomech_rules = result.get("biomech_rules", [])
    biomech_components = result.get("biomech_components", {})
    alpha_v    = result.get("alpha_video", 0.0)
    lstm_std   = result.get("lstm_std",  0.0)
    transf_std = result.get("transformer_std", 0.0)
    uncertainty = result.get("uncertainty_band", 0.0)

    name = active_meta["name"] if active_meta else "Athlete"
    st.subheader(f"Results for: **{name}** — {sess.get('date','')}")
    if has_bio:
        st.caption(
            f"🎥 Video biomechanics included · "
            f"α_video = **{alpha_v*100:.0f}%** "
            f"(confidence {biomech_conf*100:.0f}%, "
            f"{result.get('biomech_n_frames', 0)} frames)"
        )
    if mc_samples > 1:
        st.caption(f"🎲 MC-Dropout: {mc_samples} samples · Uncertainty band: ±{uncertainty:.1f}")

    # ══ THREE-GAUGE LAYOUT: Workload | Final | Biomech ════════════════
    cat_final = risk_cat(final_val)
    cat_work  = risk_cat(workload_r)
    cat_bio   = risk_cat(biomech_r)

    g1, g2, g3 = st.columns(3)

    with g1:
        fig_w = go.Figure(go.Indicator(
            mode="gauge+number",
            value=workload_r,
            title={"text": "💪 Workload Risk<br><span style='font-size:0.7em'>(ML ensemble)</span>"},
            gauge={
                "axis":  {"range": [0, 100]},
                "bar":   {"color": C["Blue"], "thickness": 0.28},
                "steps": [
                    {"range": [0,  30], "color": C["Low"]},
                    {"range": [30, 70], "color": C["Medium"]},
                    {"range": [70,100], "color": C["High"]},
                ],
            },
        ))
        fig_w.update_layout(height=280, margin=dict(t=70,b=10,l=15,r=15))
        st.plotly_chart(fig_w, use_container_width=True)
        st.caption(f"Stack/LSTM/TF/Hazard blend → **{workload_r:.1f}**")

    with g2:
        fig_f = go.Figure(go.Indicator(
            mode="gauge+number",
            value=final_val,
            title={"text": f"🔀 Final Hybrid Risk<br><span style='font-size:0.7em'>{cat_final}</span>"},
            gauge={
                "axis":  {"range": [0, 100]},
                "bar":   {"color": C["Dark"], "thickness": 0.32},
                "steps": [
                    {"range": [0,  30], "color": C["Low"]},
                    {"range": [30, 70], "color": C["Medium"]},
                    {"range": [70,100], "color": C["High"]},
                ],
                "threshold": {"line": {"color": "red", "width": 4}, "value": 70},
            },
        ))
        fig_f.update_layout(height=280, margin=dict(t=70,b=10,l=15,r=15))
        st.plotly_chart(fig_f, use_container_width=True)
        st.markdown(
            f"<div style='text-align:center;background:{C[cat_final]};color:white;"
            f"padding:6px;border-radius:8px;font-weight:bold;font-size:14px;'>"
            f"{cat_final} Risk &nbsp;·&nbsp; ({1-alpha_v:.0%} workload + {alpha_v:.0%} biomech)</div>",
            unsafe_allow_html=True,
        )
        if mc_samples > 1 and uncertainty > 0:
            low_b  = max(0,   final_val - uncertainty)
            high_b = min(100, final_val + uncertainty)
            st.caption(f"Confidence: {low_b:.1f} – {high_b:.1f}  (±{uncertainty:.1f})")

    with g3:
        if has_bio:
            fig_b = go.Figure(go.Indicator(
                mode="gauge+number",
                value=biomech_r,
                title={"text": "🎥 Biomech Risk<br><span style='font-size:0.7em'>(rule-based)</span>"},
                gauge={
                    "axis":  {"range": [0, 100]},
                    "bar":   {"color": C["Purple"], "thickness": 0.28},
                    "steps": [
                        {"range": [0,  30], "color": C["Low"]},
                        {"range": [30, 70], "color": C["Medium"]},
                        {"range": [70,100], "color": C["High"]},
                    ],
                },
            ))
            fig_b.update_layout(height=280, margin=dict(t=70,b=10,l=15,r=15))
            st.plotly_chart(fig_b, use_container_width=True)
            n_rules = len(biomech_rules)
            st.caption(
                f"{n_rules} rule(s) triggered · confidence **{biomech_conf*100:.0f}%**"
            )
        else:
            st.markdown(
                "<div style='height:280px;display:flex;align-items:center;"
                "justify-content:center;border:2px dashed #ccc;border-radius:12px;"
                "padding:20px;text-align:center;'>"
                "<div>🎥<br><strong>Biomech Risk</strong><br>"
                "<span style='color:#888;font-size:0.85em'>"
                "Upload a training video on the<br>New Session page to enable<br>"
                "the biomechanical channel."
                "</span></div></div>",
                unsafe_allow_html=True,
            )
            st.caption("α_video = 0% (no video uploaded)")

    # ── 5-Voice Breakdown (4 ML voices + biomech) ─────────────────
    st.subheader("5-Voice Breakdown")
    cols = st.columns(5)

    def _metric(col, label, val, std=0.0):
        if val is not None:
            delta_str = f"±{std:.1f}" if std > 0 else f"({risk_cat(val)})"
            col.metric(label, f"{val:.1f}", delta_str)
        else:
            col.metric(label, "N/A", "Need more sessions")

    _metric(cols[0], "Stacked\nEnsemble",   ensemble)
    _metric(cols[1], f"LSTM\n(±{lstm_std:.1f})" if lstm_std > 0 else "LSTM\nTemporal",
            lstm_val, lstm_std)
    _metric(cols[2], f"Transformer\n(±{transf_std:.1f})" if transf_std > 0 else "Transformer\nModel",
            transf_val, transf_std)
    _metric(cols[3], "Hazard\nSurvival",     hazard_val)
    if has_bio:
        cols[4].metric("🎥 Biomech\nRules", f"{biomech_r:.1f}",
                       f"α={alpha_v*100:.0f}%")
    else:
        cols[4].metric("🎥 Biomech\nRules", "N/A", "No video")

    # Effective blend display — shows true contributions including biomech
    eff = result.get("effective_weights", {})
    if eff:
        st.caption(
            f"**Effective blend (this prediction):** "
            f"Stack {eff.get('stack',0)*100:.1f}% · "
            f"LSTM {eff.get('lstm',0)*100:.1f}% · "
            f"Transformer {eff.get('transformer',0)*100:.1f}% · "
            f"Hazard {eff.get('hazard',0)*100:.1f}% · "
            f"🎥 Biomech {eff.get('biomech',0)*100:.1f}%"
        )

    acwr_icon  = "🔴" if acwr > 1.5 else "🟡" if acwr > 1.2 else "🟢"
    acwr_label = "Danger" if acwr > 1.5 else "Caution" if acwr > 1.2 else "Safe"
    st.metric(f"{acwr_icon} ACWR ({acwr_label})", f"{acwr:.2f}",
              help="< 1.2 Safe · 1.2–1.5 Caution · > 1.5 Danger")

    st.subheader("Session Summary")
    sm1, sm2, sm3, sm4 = st.columns(4)
    sm1.metric("⏱ Duration", f"{sess.get('training_duration',0):.0f} min")
    sm2.metric("💓 HRV",     f"{sess.get('heart_rate_variability',0):.0f}")
    sm3.metric("💤 Sleep",   f"{sess.get('sleep_hours',0):.1f} h")
    sm4.metric("😓 Fatigue", f"{sess.get('fatigue_level',0):.1f}/10")

    seq_len = getattr(predictor, "lstm_seq_len", 30) if predictor else 30
    if hist_len < seq_len:
        sessions_so_far = hist_len + 1
        st.info(
            f"ℹ️ LSTM and Transformer use a **{seq_len}-session window**. "
            f"You have **{sessions_so_far}** — until {seq_len - sessions_so_far} "
            f"more sessions are recorded, those two voices contribute little."
        )

    st.divider()

    # ══ TRANSPARENT BIOMECH RULES PANEL (only when video uploaded) ════
    if has_bio:
        st.subheader("🎥 Biomechanical Risk Breakdown")
        st.caption(
            "Rule-based score derived from MediaPipe pose statistics. "
            "Each rule below contributed points to the biomech_risk above. "
            "Rules sourced from sports-medicine literature on injury mechanics."
        )

        # ── Sport + phase summary banner ────────────────────────────
        sport_used      = result.get("biomech_sport", "generic").replace("_", " ").title()
        phase_summary   = result.get("biomech_phase_summary", {}) or {}
        if phase_summary:
            phase_blurb = " · ".join(
                f"{p.title()}: {n}" for p, n in phase_summary.items() if n > 0
            ) or "no phase frames detected"
            st.info(
                f"**Sport context:** {sport_used} · "
                f"**Phase distribution:** {phase_blurb}"
            )
        else:
            st.info(f"**Sport context:** {sport_used}")

        if biomech_rules:
            rules_rows = []
            for r in biomech_rules:
                sev_color = (
                    C["High"] if r["severity"] == "High"
                    else C["Medium"] if r["severity"] == "Medium"
                    else C["Low"]
                )
                phase_label = r.get("phase", "global")
                if r.get("fallback"):
                    phase_label = f"global (fallback)"
                rules_rows.append({
                    "Rule":      r["name"],
                    "Severity":  r["severity"],
                    "Points":    f"+{r['points']:.1f}",
                    "Observed":  f"{r['value']:.2f}",
                    "Threshold": f"{r['threshold']:.2f}",
                    "Phase":     phase_label,
                    "_color":    sev_color,
                })
            rules_df = pd.DataFrame(rules_rows).drop(columns=["_color"])
            st.dataframe(
                rules_df,
                use_container_width=True,
                hide_index=True,
            )

            for r in biomech_rules:
                phase_suffix = ""
                if r.get("fallback"):
                    phase_suffix = "  ·  ⚠️ scored on global stats (insufficient phase frames)"
                elif r.get("phase") and r["phase"] != "global":
                    phase_suffix = f"  ·  scored on **{r['phase']}** phase"
                with st.expander(
                    f"{r['name']} — {r['severity']} (+{r['points']:.1f} points)",
                    expanded=(r["severity"] == "High"),
                ):
                    st.markdown(r["description"] + phase_suffix)
        else:
            st.success(
                "✅ No biomechanical rules triggered — pose statistics are within "
                "safe ranges. The video did not add any risk to the workload score."
            )

        # Stacked-bar showing biomech components
        if biomech_components:
            comp_df = pd.DataFrame([
                {"Rule": k.replace("_", " ").title(), "Points": v}
                for k, v in biomech_components.items()
            ])
            fig_comp = px.bar(
                comp_df.sort_values("Points", ascending=True),
                x="Points", y="Rule", orientation="h",
                color="Points", color_continuous_scale="Reds",
                title="Biomechanical risk contributions (points)",
                range_x=[0, 25],
            )
            fig_comp.update_layout(height=280, showlegend=False)
            st.plotly_chart(fig_comp, use_container_width=True)

        st.divider()

    # ── Hazard gauge ───────────────────────────────────────────────
    if hazard_val is not None:
        st.subheader("☠️ Hazard Survival Model")
        hc1, hc2 = st.columns([1, 2])
        with hc1:
            fig_haz = go.Figure(go.Indicator(
                mode="gauge+number",
                value=hazard_val,
                title={"text": "Injury Hazard Probability"},
                gauge={
                    "axis":  {"range": [0, 100]},
                    "bar":   {"color": C["Dark"], "thickness": 0.28},
                    "steps": [
                        {"range": [0,  30], "color": C["Low"]},
                        {"range": [30, 70], "color": C["Medium"]},
                        {"range": [70,100], "color": C["High"]},
                    ],
                },
            ))
            fig_haz.update_layout(height=240, margin=dict(t=55,b=10,l=15,r=15))
            st.plotly_chart(fig_haz, use_container_width=True)
        with hc2:
            hz = hazard_val or 0
            if hz > 70:
                st.error("🔴 High cumulative hazard — athlete may be approaching injury threshold")
            elif hz > 40:
                st.warning("🟡 Moderate hazard — monitor closely")
            else:
                st.success("🟢 Low cumulative hazard")

    # ── MQS (from video or manual) ─────────────────────────────────
    bio_cols = ["posture_symmetry_mean","balance_stability_mean","body_symmetry_score_mean",
                "movement_fluidity_mean","shoulder_symmetry_mean"]
    if any(k in sess for k in bio_cols):
        # FIX v4: tolerate missing/NaN MQS
        mqs_raw = sess.get("movement_quality_score")
        try:
            mqs_v = float(mqs_raw) if mqs_raw is not None and not pd.isna(mqs_raw) else None
        except (TypeError, ValueError):
            mqs_v = None
        if mqs_v is None or pd.isna(mqs_v):
            mqs_v = compute_mqs(sess)
        mc1, mc2 = st.columns(2)
        with mc1:
            source_label = "🎥 Video MQS" if has_bio else "🎯 Movement Quality Score"
            st.metric(source_label, f"{mqs_v:.1f}/100", mqs_grade(mqs_v))
        with mc2:
            bio_vals = {k: float(sess.get(v, 0.8)) for k, v in {
                "Posture": "posture_symmetry_mean",
                "Balance": "balance_stability_mean",
                "Body Sym": "body_symmetry_score_mean",
                "Fluidity": "movement_fluidity_mean",
                "Shoulder": "shoulder_symmetry_mean",
            }.items() if v in sess}
            if bio_vals:
                fig_b = go.Figure(go.Bar(
                    x=list(bio_vals.keys()), y=list(bio_vals.values()),
                    marker_color=["#2ecc71" if v > 0.8 else "#f39c12" if v > 0.6 else "#e74c3c"
                                  for v in bio_vals.values()],
                ))
                fig_b.add_hline(y=0.8, line_dash="dash", line_color="green")
                fig_b.add_hline(y=0.6, line_dash="dot",  line_color="orange")
                fig_b.update_layout(yaxis=dict(range=[0,1]), height=200,
                                    margin=dict(t=20,b=20,l=20,r=20),
                                    title="Biomechanics Quality" + (" (from video)" if has_bio else ""))
                st.plotly_chart(fig_b, use_container_width=True)

    # ── Injury type breakdown ──────────────────────────────────────
    st.subheader("🦴 Injury Type Risk Breakdown")
    inj = injury_types_for_row(sess, acwr)
    if inj:
        inj_df = pd.DataFrame({
            "Injury Type": [i.name for i in inj],
            "Risk Score":  [i.risk_score for i in inj],
            "Risk Level":  [i.risk_level for i in inj],
        })
        fig_i = px.bar(inj_df, x="Risk Score", y="Injury Type", orientation="h",
                       color="Risk Level", color_discrete_map=C,
                       title="Injury Type Risk Breakdown", range_x=[0,100])
        fig_i.add_vline(x=35, line_dash="dot",  line_color=C["Medium"])
        fig_i.add_vline(x=65, line_dash="dash", line_color=C["High"])
        st.plotly_chart(fig_i, use_container_width=True)

        for i_ in inj:
            with st.expander(f"{i_.name} — {i_.risk_level} ({i_.risk_score:.0f}/100)",
                             expanded=i_.risk_score >= 35):
                for sig in i_.key_signals:
                    st.write(f"• {sig}")
                st.info(i_.advice)
    else:
        st.info("InjuryTypeClassifier not available.")

    # ── Recommendations ────────────────────────────────────────────
    st.subheader("📋 Personalised Recommendations")
    recs = []
    fat = float(sess.get("fatigue_level", 5))
    slp = float(sess.get("sleep_hours",  8))
    hrv = float(sess.get("heart_rate_variability", 60))
    wel = float(sess.get("wellness_score", 7))
    if acwr > 1.5:     recs.append(("🔴", "ACWR critical",   "Reduce training load immediately — danger of overuse injury."))
    if acwr > 1.2:     recs.append(("🟡", "ACWR elevated",   "Monitor load closely. Avoid additional high-intensity sessions."))
    if fat > 7:        recs.append(("🟡", "High fatigue",    "Include a full rest day or active recovery session."))
    if slp < 6:        recs.append(("🔴", "Sleep deficit",   "Prioritise 7–9 hours of sleep — critical for recovery and HRV."))
    if hrv < 45:       recs.append(("🟡", "Low HRV",         "Recovery incomplete. Reduce intensity and prioritise sleep."))
    if wel < 4:        recs.append(("🔴", "Low wellness",    "Consider medical check — wellness score significantly below normal."))
    if final_val > 70: recs.append(("🔴", "High overall risk","Training session should be modified or cancelled today."))
    if hazard_val and hazard_val > 70:
        recs.append(("🔴", "High hazard signal", "Survival model flags elevated cumulative injury risk."))
    if mc_samples > 1 and uncertainty > 10:
        recs.append(("🟡", "High model uncertainty",
                     f"±{uncertainty:.1f} uncertainty from MC-Dropout. Prediction less reliable — use caution."))
    if has_bio:
        kl_std = float(sess.get("knee_angle_left_std", 0))
        kr_std = float(sess.get("knee_angle_right_std", 0))
        if max(kl_std, kr_std) > 12:
            recs.append(("🟡", "High knee variability (video)", "Video analysis shows unstable knee mechanics — review landing technique."))
        bsym = float(sess.get("body_symmetry_score_mean", 1.0))
        if bsym < 0.7:
            recs.append(("🟡", "Low body symmetry (video)", "Asymmetric movement patterns detected — check for compensatory mechanics."))
    if not recs:
        recs.append(("🟢", "All clear", "Risk metrics are within safe ranges. Maintain current training plan."))

    for icon, title, detail in recs:
        st.markdown(f"{icon} **{title}** — {detail}")

# ══════════════════════════════════════════════════════════════════
# PAGE 4 — ATHLETE DASHBOARD
# ══════════════════════════════════════════════════════════════════
elif page == "📊 Athlete Dashboard":
    st.title("📊 Athlete Dashboard")

    if not active_id:
        st.warning("No athlete selected.")
        st.stop()

    hist = load_history(active_id)
    name = active_meta["name"] if active_meta else "Athlete"
    st.subheader(f"**{name}** — {active_meta.get('sport','')} | Age {active_meta.get('age','')} | {active_meta.get('position','')}")

    if hist.empty:
        st.info("No sessions yet. Use **➕ New Session** to add data.")
        st.stop()

    if "date" in hist.columns:
        hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
        hist = hist.sort_values("date").reset_index(drop=True)

    if "injury_risk_score" in hist.columns:
        hist["risk_category"] = hist["injury_risk_score"].apply(risk_cat)
        hist["risk_7d_avg"]   = hist["injury_risk_score"].rolling(7, min_periods=1).mean()
        hist["risk_velocity"] = hist["injury_risk_score"].diff()

    if "training_duration" in hist.columns:
        acute   = hist["training_duration"].rolling(7,  min_periods=1).mean()
        chronic = hist["training_duration"].rolling(28, min_periods=1).mean()
        hist["acwr_computed"] = acute / (chronic + 1e-9)

    acwr_col = "acwr" if "acwr" in hist.columns else "acwr_computed"
    latest   = hist.iloc[-1]
    risk_    = float(latest.get("injury_risk_score", 50))

    # ── KPIs ──────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Sessions",     len(hist))
    k2.metric("Avg Risk",     f"{hist['injury_risk_score'].mean():.1f}" if "injury_risk_score" in hist.columns else "—")
    k3.metric("Max Risk",     f"{hist['injury_risk_score'].max():.1f}"  if "injury_risk_score" in hist.columns else "—")
    k4.metric("Latest Risk",  f"{risk_:.1f}")
    k5.metric("ACWR (latest)",f"{float(latest.get(acwr_col, 1)):.2f}" if acwr_col in latest.index else "—")

    if "has_video_biomechanics" in hist.columns:
        video_count = int(hist["has_video_biomechanics"].sum())
        if video_count > 0:
            st.caption(f"🎥 {video_count} of {len(hist)} sessions include video biomechanics")

    # ── Sequence model readiness ───────────────────────────────────
    seq_len = getattr(predictor, "lstm_seq_len", 30) if predictor else 30
    if len(hist) < seq_len:
        remaining = seq_len - len(hist)
        st.info(
            f"📈 **Sequence models still warming up ({len(hist)}/{seq_len} sessions).**  "
            f"LSTM and Transformer use a {seq_len}-step window — the older slots are "
            f"currently padded, so predictions carry limited temporal signal until "
            f"{remaining} more sessions are logged. Stacked Ensemble + Hazard are "
            f"unaffected and produce row-level predictions from session 1."
        )
    else:
        st.success(
            f"✅ All 4 model components have full context "
            f"({len(hist)} sessions ≥ {seq_len} required for full sequence window)."
        )

    st.divider()

    # ── Gauge ──────────────────────────────────────────────────────
    cg, cm = st.columns([1, 2])
    with cg:
        prev_r = float(hist.iloc[-2]["injury_risk_score"]) if len(hist) > 1 and "injury_risk_score" in hist.columns else risk_
        fig_g  = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk_,
            delta={"reference": prev_r},
            title={"text": f"{name}<br>Latest Risk"},
            gauge={
                "axis":  {"range": [0,100]},
                "bar":   {"color": C["Dark"], "thickness": 0.28},
                "steps": [
                    {"range":[0,30],  "color": C["Low"]},
                    {"range":[30,70], "color": C["Medium"]},
                    {"range":[70,100],"color": C["High"]},
                ],
                "threshold": {"line": {"color": "red", "width": 4}, "value": 70},
            },
        ))
        fig_g.update_layout(height=280, margin=dict(t=60,b=10,l=20,r=20))
        st.plotly_chart(fig_g, use_container_width=True)

    with cm:
        st.subheader("Latest Session Metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("⏱ Training",  f"{latest.get('training_duration',0):.0f} min")
        m2.metric("💓 HRV",      f"{latest.get('heart_rate_variability',0):.0f}")
        m3.metric("💤 Sleep",    f"{latest.get('sleep_hours',0):.1f} h")
        m4.metric("😓 Fatigue",  f"{latest.get('fatigue_level',0):.1f}/10")
        m5, m6, m7, m8 = st.columns(4)
        m5.metric("⚡ Intensity", f"{latest.get('intensity_rating',0):.1f}/10")
        m6.metric("🏃 Sprints",   f"{int(latest.get('sprint_count',0))}")
        m7.metric("📏 Distance",  f"{latest.get('running_distance',0):.1f} km")
        m8.metric("🌟 Wellness",  f"{latest.get('wellness_score',0):.1f}/10")

        bio_check_cols = ["posture_symmetry_mean","movement_fluidity_mean"]
        if any(c in hist.columns for c in bio_check_cols):
            # FIX v4: handle NaN MQS gracefully — was showing "nan/100" on reload
            # because earlier sessions saved before the column existed produce NaN.
            mqs_raw = latest.get("movement_quality_score")
            try:
                mqs_v = float(mqs_raw) if mqs_raw is not None and not pd.isna(mqs_raw) else None
            except (TypeError, ValueError):
                mqs_v = None
            if mqs_v is None or pd.isna(mqs_v):
                mqs_v = compute_mqs(latest)
            src = "🎥 MQS (video)" if latest.get("has_video_biomechanics") else "🎯 MQS"
            st.metric(src, f"{mqs_v:.1f}/100", mqs_grade(mqs_v))

    # ── Risk trend ─────────────────────────────────────────────────
    if "injury_risk_score" in hist.columns:
        st.subheader("Risk Score Trend")
        ycols = ["injury_risk_score"]
        if "risk_7d_avg" in hist.columns: ycols.append("risk_7d_avg")
        fig_rt = px.line(hist, x="date", y=ycols,
                         color_discrete_map={"injury_risk_score": C["Blue"], "risk_7d_avg": C["High"]},
                         title=f"{name} — Risk Over Time")
        fig_rt.add_hrect(y0=70, y1=100, fillcolor="red",    opacity=0.08)
        fig_rt.add_hrect(y0=30, y1=70,  fillcolor="orange", opacity=0.05)
        if "has_video_biomechanics" in hist.columns:
            vid_sess = hist[hist["has_video_biomechanics"] == True]
            if not vid_sess.empty and "injury_risk_score" in vid_sess.columns:
                fig_rt.add_trace(go.Scatter(
                    x=vid_sess["date"], y=vid_sess["injury_risk_score"],
                    mode="markers", marker=dict(symbol="diamond", size=10, color=C["Purple"]),
                    name="🎥 Video session",
                ))
        st.plotly_chart(fig_rt, use_container_width=True)

    # ── Risk velocity ──────────────────────────────────────────────
    if "risk_velocity" in hist.columns and len(hist) > 2:
        st.subheader("⚡ Risk Velocity (Early Warning Signal)")
        fig_vel = go.Figure()
        colors  = [C["High"] if v > 5 else C["Medium"] if v > 0 else C["Low"]
                   for v in hist["risk_velocity"].fillna(0)]
        fig_vel.add_trace(go.Bar(x=hist["date"], y=hist["risk_velocity"],
                                 marker_color=colors, name="Risk Velocity"))
        fig_vel.add_hline(y=5,  line_dash="dash", line_color="orange", annotation_text="+5 Caution")
        fig_vel.add_hline(y=10, line_dash="dash", line_color="red",    annotation_text="+10 Alert")
        fig_vel.add_hline(y=0,  line_dash="dot",  line_color="gray")
        fig_vel.update_layout(title="Risk Score Velocity (Δ per session)", yaxis_title="Δ Risk Score")
        st.plotly_chart(fig_vel, use_container_width=True)

    # ── ACWR ───────────────────────────────────────────────────────
    if acwr_col in hist.columns:
        st.subheader("ACWR Trend")
        fig_acwr = go.Figure()
        fig_acwr.add_trace(go.Scatter(x=hist["date"], y=hist[acwr_col],
                                      mode="lines+markers", line=dict(color=C["Blue"], width=2)))
        fig_acwr.add_hrect(y0=0,   y1=1.2, fillcolor="green",  opacity=0.07)
        fig_acwr.add_hrect(y0=1.2, y1=1.5, fillcolor="orange", opacity=0.10)
        fig_acwr.add_hrect(y0=1.5, y1=3.0, fillcolor="red",    opacity=0.10)
        fig_acwr.add_hline(y=1.5, line_dash="dash", line_color="red",    annotation_text="1.5 Danger")
        fig_acwr.add_hline(y=1.2, line_dash="dot",  line_color="orange", annotation_text="1.2 Caution")
        fig_acwr.update_layout(title="ACWR Over Time", yaxis_title="ACWR")
        st.plotly_chart(fig_acwr, use_container_width=True)

    # ── Load components ────────────────────────────────────────────
    lcols = [c for c in ["training_duration","running_distance","sprint_count","intensity_rating"]
             if c in hist.columns and len(hist[c].dropna()) > 0]
    if lcols:
        st.subheader("Training Load Components")
        fig_l = make_subplots(rows=2, cols=2, subplot_titles=lcols[:4])
        for (col_, (r_,c_)) in zip(lcols, [(1,1),(1,2),(2,1),(2,2)]):
            fig_l.add_trace(go.Scatter(x=hist["date"], y=hist[col_],
                                       mode="lines+markers", line=dict(width=1.5)), row=r_, col=c_)
        fig_l.update_layout(height=360, showlegend=False, title="Load Components Over Time")
        st.plotly_chart(fig_l, use_container_width=True)

    # ── Biomechanics radar ─────────────────────────────────────────
    bio_map = {
        "Posture Sym.":  "posture_symmetry_mean",
        "Balance":       "balance_stability_mean",
        "Body Sym.":     "body_symmetry_score_mean",
        "Fluidity":      "movement_fluidity_mean",
        "Shoulder Sym.": "shoulder_symmetry_mean",
    }
    avail = {k: v for k, v in bio_map.items() if v in hist.columns}
    if len(avail) >= 3:
        st.subheader("🎯 Biomechanics Radar")
        bio_hist = hist[[v for v in avail.values()]].dropna(how="all")
        if not bio_hist.empty:
            cats  = list(avail.keys())
            v_lat = [float(latest.get(v, 0.8)) for v in avail.values()]
            v_all = [float(hist[v].mean()) for v in avail.values()]
            fig_r = go.Figure()
            for vals, nm, col_ in [(v_lat,"Latest Session",C["High"]), (v_all,"All-time avg",C["Blue"])]:
                fig_r.add_trace(go.Scatterpolar(
                    r=vals+[vals[0]], theta=cats+[cats[0]],
                    fill="toself", name=nm, line_color=col_,
                ))
            fig_r.update_layout(polar=dict(radialaxis=dict(range=[0.4,1.0])),
                                title="Biomechanics Quality")
            st.plotly_chart(fig_r, use_container_width=True)

    # ── Injury types ───────────────────────────────────────────────
    st.subheader("🦴 Injury Type Risk (Latest Session)")
    acwr_v = float(latest.get(acwr_col, 1.0)) if acwr_col in latest.index else 1.0
    inj = injury_types_for_row(latest, acwr_v)
    if inj:
        inj_df = pd.DataFrame({
            "Injury Type": [i.name for i in inj],
            "Risk Score":  [i.risk_score for i in inj],
            "Risk Level":  [i.risk_level for i in inj],
        })
        fig_i = px.bar(inj_df, x="Risk Score", y="Injury Type", orientation="h",
                       color="Risk Level", color_discrete_map=C,
                       title="Injury Type Risk", range_x=[0,100])
        st.plotly_chart(fig_i, use_container_width=True)

    # ── Session history table ──────────────────────────────────────
    st.subheader("Session History")
    show_cols = [c for c in ["date","injury_risk_score","risk_category","training_duration",
                              "fatigue_level","sleep_hours","heart_rate_variability",
                              acwr_col,"has_video_biomechanics"]
                 if c in hist.columns]
    st.dataframe(
        hist[show_cols].sort_values("date", ascending=False)
        .style.background_gradient(
            subset=["injury_risk_score"] if "injury_risk_score" in show_cols else [],
            cmap="RdYlGn_r", vmin=0, vmax=100,
        ),
        use_container_width=True,
    )

    if st.button("🗑️ Delete Last Session", type="secondary"):
        delete_last_session(active_id)
        st.success("Last session deleted.")
        _rerun()

# ══════════════════════════════════════════════════════════════════
# PAGE 5 — SQUAD OVERVIEW
# ══════════════════════════════════════════════════════════════════
elif page == "🏟️ Squad Overview":
    st.title("🏟️ Real-World Squad Overview")

    meta = load_athletes_meta()
    if not meta:
        st.info("No athletes yet. Create athletes in Athlete Manager.")
        st.stop()

    summary = []
    for aid, m in meta.items():
        hist = load_history(int(aid))
        if hist.empty:
            summary.append({
                "ID": int(aid), "Name": m["name"], "Sport": m["sport"],
                "Age": m["age"], "Sessions": 0,
                "Avg Risk": None, "Latest Risk": None, "Latest ACWR": None, "Video Sessions": 0,
            })
            continue
        avg_r = float(hist["injury_risk_score"].mean()) if "injury_risk_score" in hist.columns else None
        lat_r = float(hist["injury_risk_score"].iloc[-1]) if "injury_risk_score" in hist.columns else None
        acwr_v = None
        if "acwr" in hist.columns:
            acwr_v = float(hist["acwr"].iloc[-1])
        elif "training_duration" in hist.columns and len(hist) >= 2:
            acwr_v = float(hist["training_duration"].tail(7).mean() /
                           (hist["training_duration"].tail(28).mean() + 1e-9))
        vid_count = int(hist["has_video_biomechanics"].sum()) if "has_video_biomechanics" in hist.columns else 0
        summary.append({
            "ID": int(aid), "Name": m["name"], "Sport": m["sport"],
            "Age": m["age"], "Sessions": len(hist),
            "Avg Risk":    round(avg_r, 1) if avg_r is not None else None,
            "Latest Risk": round(lat_r, 1) if lat_r is not None else None,
            "Latest ACWR": round(acwr_v, 2) if acwr_v is not None else None,
            "Video Sessions": vid_count,
        })

    sq_df    = pd.DataFrame(summary)
    sq_valid = sq_df[sq_df["Sessions"] > 0].copy()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Athletes",     len(sq_df))
    k2.metric("Athletes with Data", len(sq_valid))
    if not sq_valid.empty and "Latest Risk" in sq_valid.columns:
        high_count = int((sq_valid["Latest Risk"].dropna() > 70).sum())
        k3.metric("🔴 High Risk Now", high_count)
    if not sq_valid.empty and "Video Sessions" in sq_valid.columns:
        k4.metric("🎥 Video Sessions", int(sq_valid["Video Sessions"].sum()))

    st.divider()

    st.subheader("Squad Summary")
    if not sq_valid.empty and "Latest Risk" in sq_valid.columns:
        st.dataframe(
            sq_df.style.background_gradient(
                subset=["Latest Risk","Avg Risk"], cmap="RdYlGn_r", vmin=0, vmax=100,
            ),
            use_container_width=True,
        )

        sq_sorted = sq_valid.sort_values("Latest Risk", ascending=False)
        fig_sq = px.bar(sq_sorted, x="Name", y="Latest Risk",
                        color="Latest Risk", color_continuous_scale="RdYlGn_r",
                        title="Latest Risk Score by Athlete",
                        labels={"Latest Risk":"Risk Score"}, range_y=[0,100])
        fig_sq.add_hline(y=70, line_dash="dash", line_color="red",   annotation_text="High Risk")
        fig_sq.add_hline(y=30, line_dash="dot",  line_color="green", annotation_text="Low Risk")
        st.plotly_chart(fig_sq, use_container_width=True)

        acwr_df = sq_valid[sq_valid["Latest ACWR"].notna()].sort_values("Latest ACWR", ascending=False)
        if not acwr_df.empty:
            fig_acwr = px.bar(acwr_df, x="Name", y="Latest ACWR",
                              color="Latest ACWR", color_continuous_scale="RdYlGn_r",
                              color_continuous_midpoint=1.2,
                              title="ACWR by Athlete (Latest Session)")
            fig_acwr.add_hline(y=1.5, line_dash="dash", line_color="red",    annotation_text="1.5 Danger")
            fig_acwr.add_hline(y=1.2, line_dash="dot",  line_color="orange", annotation_text="1.2 Caution")
            st.plotly_chart(fig_acwr, use_container_width=True)
    else:
        st.dataframe(sq_df, use_container_width=True)
        st.info("Enter sessions for athletes to see risk comparisons.")

# ── Footer ──────────────────────────────────────────────────────────
st.divider()
st.caption(
    "🏟️ Athlete Injury Risk Prediction System — Real-World Data Dashboard (v4)  ·  "
    "Stacked Ensemble + LSTM + Transformer + Hazard + 🎥 Biomech Rules  ·  "
    "Two-channel hybrid · MC-Dropout uncertainty · MediaPipe Video Biomechanics"
)
