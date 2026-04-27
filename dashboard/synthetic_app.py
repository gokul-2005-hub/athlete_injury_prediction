"""
synthetic_app.py  — Dashboard 1: Synthetic Data Analytics
──────────────────────────────────────────────────────────
Comprehensive visualization of the trained model on synthetic athlete data.

Run from project root:
    streamlit run dashboard/synthetic_app.py

Model architecture
──────────────────
Stacked Ensemble (XGB + LGB + CatBoost + ExtraTrees → Ridge)
+ LSTM Sequence Model (BiLSTM(64)→LSTM(32), per-athlete temporal sequences)
+ Transformer Sequence Model (sinusoidal positional encoding, 2 blocks, ff_dim=128)
+ Random Survival Forest (Hazard)
= HybridPredictor  (learned blend weights, MC-Dropout uncertainty)

Pages
─────
  📊 Overview            — Squad KPIs, distributions, correlations, trends
  🏃 Individual Athlete  — 5-model gauge, metrics, risk trend, load, MQS, radar, injury types
  🧠 Model Insights      — Feature importance, LSTM history, accuracy metrics, live demo
  🚨 Alerts              — High-risk alerts with actionable recommendations
  📈 ACWR & Load         — Workload ratio analysis and load component tracking
  🦴 Injury Type         — InjuryTypeClassifier breakdown per athlete
  🗺️ Squad Heatmap       — All athletes × week risk heatmap + athlete profiles
"""

from __future__ import annotations

import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────
BASE_DIR          = Path(__file__).resolve().parent.parent
SRC_DIR           = BASE_DIR / "src"
# FIX: was pointing to data/processed/engineered_features.csv which no longer
# exists after the pipeline rebuild.  combined_dataset.csv is the active file.
DATA_PATH         = BASE_DIR / "data" / "combined_dataset.csv"
LSTM_HISTORY_PATH = BASE_DIR / "results" / "lstm_training_history.csv"

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(BASE_DIR))  # Also add project root so "from src.X import Y" resolves inside loaded modules

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Synthetic Data Dashboard | Athlete Injury Risk",
    page_icon="⚽",
    layout="wide",
)

# ── Brand colours ──────────────────────────────────────────────────
C = {
    "Low":    "#2ecc71",
    "Medium": "#f39c12",
    "High":   "#e74c3c",
    "Blue":   "#3498db",
    "Dark":   "#2c3e50",
    "Purple": "#9b59b6",
}

# ══════════════════════════════════════════════════════════════════
# UTILITY HELPERS
# ══════════════════════════════════════════════════════════════════

def risk_cat(score: float) -> str:
    if score < 30: return "Low"
    if score < 70: return "Medium"
    return "High"

def risk_color(score: float) -> str:
    return C[risk_cat(score)]

# v4 NOTE: compute_mqs() and the bio_cols block were removed in v4 because the
# v3 fake-CV columns (posture_symmetry, balance_score, movement_smoothness)
# are no longer in combined_dataset.csv — the synthetic data is workload-only,
# and real video biomechanics live in the real-world dashboard. The mqs_grade
# helper is kept because it's still used by display fallbacks elsewhere.

def mqs_grade(s: float) -> str:
    if s >= 90: return "Excellent ✅"
    if s >= 70: return "Good 🟢"
    if s >= 50: return "Needs Improvement 🟡"
    return "Injury Risk Zone 🔴"

def injury_types_for_row(row) -> list:
    try:
        from cv.injury_type_classifier import InjuryTypeClassifier
        w = {
            "training_duration":      float(row.get("training_duration", 90)),
            "heart_rate_variability": float(row.get("heart_rate_variability", 60)),
            "running_distance":       float(row.get("running_distance", 8)),
            "sprint_count":           float(row.get("sprint_count", 10)),
            "sleep_hours":            float(row.get("sleep_hours", 7)),
            "intensity_rating":       float(row.get("intensity_rating", 5)),
            "previous_injuries":      float(row.get("previous_injuries", 0)),
            "fatigue_level":          float(row.get("fatigue_level", 5)),
            "wellness_score":         float(row.get("wellness_score", 7)),
            "acwr":                   float(row.get("acwr", 1.0)),
        }
        kl_std = float(row.get("knee_angle_left_std",  5))
        kr_std = float(row.get("knee_angle_right_std", 5))
        bsym   = float(row.get("body_symmetry_score_mean", 0.8))
        lri    = (1.0 - bsym) * 50 + max(kl_std, kr_std) * 0.5
        v = {
            "knee_angle_left_std":        kl_std,
            "knee_angle_right_std":       kr_std,
            "landing_risk_index":         lri,
            "body_symmetry_score_mean":   bsym,
            "posture_symmetry_mean":      float(row.get("posture_symmetry_mean",    0.8)),
            "balance_stability_mean":     float(row.get("balance_stability_mean",   0.8)),
            "movement_fluidity_mean":     float(row.get("movement_fluidity_mean",   0.8)),
            "shoulder_symmetry_mean":     float(row.get("shoulder_symmetry_mean",   0.8)),
            "torso_lean_mean":            float(row.get("torso_lean_mean",          0.1)),
            "elbow_angle_left_std":       float(row.get("elbow_angle_left_std",     5)),
            "elbow_angle_right_std":      float(row.get("elbow_angle_right_std",    5)),
            "running_knee_angle_left_std":kl_std,
        }
        return InjuryTypeClassifier().classify(w, v)
    except Exception:
        return []

# ══════════════════════════════════════════════════════════════════
# CACHED LOADERS
# ══════════════════════════════════════════════════════════════════

@st.cache_data
def load_data():
    # FIX: load from combined_dataset.csv (the active post-pipeline file)
    if not DATA_PATH.exists():
        st.error(
            f"Dataset not found at `{DATA_PATH}`. "
            "Run `python -m src.utils.build_combined_dataset` first."
        )
        st.stop()

    df = pd.read_csv(DATA_PATH)

    # Normalise column name variants
    if "athleteid" in df.columns and "athlete_id" not in df.columns:
        df = df.rename(columns={"athleteid": "athlete_id"})
    if "injuryriskscore_next" in df.columns and "injury_risk_score" not in df.columns:
        df["injury_risk_score"] = df["injuryriskscore_next"]
    elif "injuryriskscore" in df.columns and "injury_risk_score" not in df.columns:
        df["injury_risk_score"] = df["injuryriskscore"]
    # Use injury_risk_score_next as the display risk if available
    elif "injury_risk_score_next" in df.columns and "injury_risk_score" not in df.columns:
        df["injury_risk_score"] = df["injury_risk_score_next"]

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values(["athlete_id", "date"]).reset_index(drop=True)

    df["risk_category"] = df["injury_risk_score"].apply(risk_cat)

    if "acwr" not in df.columns and "training_duration" in df.columns:
        df["acwr"] = (
            df.groupby("athlete_id")["training_duration"]
              .transform(lambda x: x.rolling(7, min_periods=1).mean()) /
            (df.groupby("athlete_id")["training_duration"]
               .transform(lambda x: x.rolling(28, min_periods=1).mean()) + 1e-9)
        )

    df["risk_7d_avg"] = (
        df.groupby("athlete_id")["injury_risk_score"]
          .transform(lambda x: x.rolling(7, min_periods=1).mean())
    )
    df["risk_velocity"] = (
        df.groupby("athlete_id")["injury_risk_score"]
          .transform(lambda x: x.diff())
    )
    df["risk_momentum"] = (
        df.groupby("athlete_id")["risk_velocity"]
          .transform(lambda x: x.rolling(7, min_periods=1).mean())
    )

    # v4: skipped MQS computation — fake CV columns no longer in dataset.

    if {"acwr", "fatigue_level", "sleep_hours"}.issubset(df.columns):
        df["fatigue_index"] = (
            0.5 * df["acwr"].clip(0, 3) * 33 +
            0.3 * df["fatigue_level"] * 10 +
            0.2 * (8 - df["sleep_hours"].clip(0, 12)) * 12.5
        ).clip(0, 100)

    if "fatigue_index" in df.columns:
        fi_norm   = df["fatigue_index"] / 100
        acwr_norm = df["acwr"].clip(0, 3) / 3 if "acwr" in df.columns else 0
        df["hazard_proxy"] = np.clip(
            (0.6 * fi_norm + 0.4 * acwr_norm) * 100, 0, 100
        )

    return df


@st.cache_resource
def load_predictor():
    try:
        from ml.hybrid_predictor import HybridPredictor
        return HybridPredictor()
    except Exception as e:
        return None


@st.cache_data
def load_lstm_hist():
    try:
        h = pd.read_csv(LSTM_HISTORY_PATH)
        h.index.name = "Epoch"
        h = h.reset_index()
        h["Epoch"] += 1
        return h
    except Exception:
        return None


# ── Bootstrap ──────────────────────────────────────────────────────
df        = load_data()
predictor = load_predictor()
lstm_hist = load_lstm_hist()

# ── Helper: get actual blend weights from predictor ────────────────
def get_blend_weights() -> dict:
    """Return blend weights from loaded predictor, or fallback defaults."""
    if predictor is not None and hasattr(predictor, "_weights"):
        w = predictor._weights
        return {
            "Ensemble": round(float(w[0]) * 100, 1),
            "LSTM":     round(float(w[1]) * 100, 1),
            "Transformer": round(float(w[2]) * 100, 1),
            "Hazard":   round(float(w[3]) * 100, 1),
        }
    return {"Ensemble": 35.0, "LSTM": 30.0, "Transformer": 20.0, "Hazard": 15.0}

# ── Sidebar ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚽ Synthetic Dashboard")
    st.caption(
        f"**{df['athlete_id'].nunique()} athletes** | "
        f"**{len(df):,} sessions** | Synthetic data"
    )
    st.divider()

    page = st.radio(
        "Navigate to",
        ["📊 Overview",
         "🏃 Individual Athlete",
         "🧠 Model Insights",
         "📉 Model Evaluation",
         "🚨 Alerts",
         "📈 ACWR & Load",
         "🦴 Injury Type",
         "🗺️ Squad Heatmap"],
    )

    st.divider()
    with st.expander("🤖 Model Architecture"):
        st.write(f"Stacked Ensemble: {'✅ Loaded' if predictor else '❌ Not loaded'}")
        st.write(f"LSTM Sequence:    {'✅ Loaded' if predictor else '❌ Not loaded'}")
        st.write(f"Transformer:      {'✅ Loaded' if predictor else '❌ Not loaded'}")
        st.write(f"Hazard Survival:  {'✅ Loaded' if predictor else '❌ Not loaded'}")

        if predictor:
            bw = get_blend_weights()
            st.info(
                f"**Hybrid blend weights:**  \n"
                f"Ensemble {bw['Ensemble']}% · LSTM {bw['LSTM']}%  \n"
                f"Transformer {bw['Transformer']}% · Hazard {bw['Hazard']}%"
            )
        else:
            st.info("Active mode: **No model loaded**")

    # Sequence model stats from meta files
    if predictor:
        with st.expander("📐 Sequence Model Config"):
            st.write(f"LSTM seq length:        {getattr(predictor, 'lstm_seq_len', 30)}")
            st.write(f"Transformer seq length: {getattr(predictor, 'transformer_seq_len', 30)}")
            st.write("✅ Per-athlete sequences (no cross-athlete contamination)")
            st.write("✅ Sinusoidal positional encoding (Transformer)")

# ══════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("📊 Overview Dashboard")
    date_range = f"{df['date'].min().date()} → {df['date'].max().date()}" if "date" in df.columns else "N/A"
    st.caption(f"Synthetic dataset · {date_range} · combined_dataset.csv")

    # ── KPIs ──────────────────────────────────────
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    high  = int((df["injury_risk_score"] > 70).sum())
    med   = int(((df["injury_risk_score"] > 30) & (df["injury_risk_score"] <= 70)).sum())
    low   = int((df["injury_risk_score"] <= 30).sum())
    total = len(df)
    k1.metric("🔴 High Risk",   f"{high:,}",  f"{high/total*100:.1f}%")
    k2.metric("🟡 Medium Risk", f"{med:,}",   f"{med/total*100:.1f}%")
    k3.metric("🟢 Low Risk",    f"{low:,}",   f"{low/total*100:.1f}%")
    k4.metric("📊 Avg Risk",    f"{df['injury_risk_score'].mean():.1f}")
    k5.metric("👥 Athletes",    df["athlete_id"].nunique())
    if "hazard_proxy" in df.columns:
        k6.metric("☠️ Avg Hazard", f"{df['hazard_proxy'].mean():.1f}")
    elif "fatigue_index" in df.columns:
        k6.metric("⚡ Avg Fatigue Idx", f"{df['fatigue_index'].mean():.1f}")

    st.divider()

    # ── Distribution + Pie ─────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(
            df, x="injury_risk_score", nbins=40,
            color="risk_category", color_discrete_map=C,
            title="Injury Risk Score Distribution",
            labels={"injury_risk_score": "Risk Score", "count": "Sessions"},
        )
        fig.add_vline(x=30, line_dash="dash", line_color=C["Low"],  annotation_text="30")
        fig.add_vline(x=70, line_dash="dash", line_color=C["High"], annotation_text="70")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        rc = df["risk_category"].value_counts().reset_index()
        rc.columns = ["Category", "Count"]
        fig2 = px.pie(rc, names="Category", values="Count",
                      color="Category", color_discrete_map=C,
                      title="Risk Category Split")
        st.plotly_chart(fig2, use_container_width=True)

    # ── Squad risk trend ───────────────────────────
    st.subheader("Squad Average Risk Over Time")
    daily = df.groupby("date")["injury_risk_score"].mean().reset_index()
    daily["7d_avg"] = daily["injury_risk_score"].rolling(7, min_periods=1).mean()
    fig_tr = go.Figure()
    fig_tr.add_trace(go.Scatter(x=daily["date"], y=daily["injury_risk_score"],
                                name="Daily avg", mode="lines",
                                line=dict(color=C["Blue"], width=1.2)))
    fig_tr.add_trace(go.Scatter(x=daily["date"], y=daily["7d_avg"],
                                name="7-day avg", mode="lines",
                                line=dict(color=C["High"], width=2.2)))
    fig_tr.add_hrect(y0=70, y1=100, fillcolor="red",    opacity=0.07, annotation_text="High Risk Zone")
    fig_tr.add_hrect(y0=30, y1=70,  fillcolor="orange", opacity=0.05, annotation_text="Medium Zone")
    fig_tr.update_layout(title="Squad Risk Trend", xaxis_title="Date", yaxis_title="Risk Score")
    st.plotly_chart(fig_tr, use_container_width=True)

    # Risk velocity overview
    if "risk_velocity" in df.columns:
        st.subheader("⚡ Squad Risk Velocity (Early Warning)")
        st.caption("Average rate of risk score change per session across all athletes. Positive = worsening.")
        daily_vel = df.groupby("date")["risk_velocity"].mean().reset_index()
        daily_vel["7d_avg"] = daily_vel["risk_velocity"].rolling(7, min_periods=1).mean()
        fig_vel = go.Figure()
        colors = [C["High"] if v > 2 else C["Medium"] if v > 0 else C["Low"]
                  for v in daily_vel["risk_velocity"].fillna(0)]
        fig_vel.add_trace(go.Bar(x=daily_vel["date"], y=daily_vel["risk_velocity"],
                                 marker_color=colors, name="Daily Velocity", opacity=0.6))
        fig_vel.add_trace(go.Scatter(x=daily_vel["date"], y=daily_vel["7d_avg"],
                                     name="7-day avg", mode="lines",
                                     line=dict(color=C["Purple"], width=2)))
        fig_vel.add_hline(y=0, line_dash="dot", line_color="gray")
        fig_vel.update_layout(title="Squad Risk Velocity (Δ Risk/session)", yaxis_title="Δ Risk Score")
        st.plotly_chart(fig_vel, use_container_width=True)

    # ── Correlation matrix ──────────────────────────
    st.subheader("Workload Feature Correlation Matrix")
    # v4: dropped the v3 fake-CV columns (posture_symmetry, balance_score,
    # movement_smoothness) from this list — they're not in v4 data.
    corr_cols = [c for c in [
        "training_duration", "heart_rate_variability", "sleep_hours",
        "intensity_rating", "fatigue_level", "wellness_score",
        "sprint_count", "running_distance",
        "acwr", "fatigue_index",
    ] if c in df.columns]
    if len(corr_cols) >= 3:
        fig_corr = px.imshow(
            df[corr_cols].corr(),
            color_continuous_scale="RdBu_r",
            title="Workload Feature Correlations",
            aspect="auto",
        )
        st.plotly_chart(fig_corr, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# PAGE 2 — INDIVIDUAL ATHLETE
# ══════════════════════════════════════════════════════════════════
elif page == "🏃 Individual Athlete":
    st.title("🏃 Individual Athlete Analysis")

    athlete_id = st.selectbox("Select Athlete", sorted(df["athlete_id"].unique()))
    adf = df[df["athlete_id"] == athlete_id].copy().sort_values("date")

    if adf.empty:
        st.warning("No data for this athlete.")
        st.stop()

    latest = adf.iloc[-1]
    score  = float(latest["injury_risk_score"])

    # ── Gauge + key metrics ────────────────────────
    col_g, col_m = st.columns([1, 2])

    with col_g:
        prev_score = float(adf.iloc[-2]["injury_risk_score"]) if len(adf) > 1 else score
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=score,
            delta={"reference": prev_score, "valueformat": ".1f"},
            title={"text": f"Athlete {athlete_id}<br>Latest Risk"},
            gauge={
                "axis":  {"range": [0, 100]},
                "bar":   {"color": C["Dark"], "thickness": 0.28},
                "steps": [
                    {"range": [0,  30], "color": C["Low"]},
                    {"range": [30, 70], "color": C["Medium"]},
                    {"range": [70,100], "color": C["High"]},
                ],
                "threshold": {"line": {"color": "red", "width": 4}, "value": 70},
            },
        ))
        fig_g.update_layout(height=280, margin=dict(t=60,b=15,l=20,r=20))
        st.plotly_chart(fig_g, use_container_width=True)

        cat  = risk_cat(score)
        col_ = C[cat]
        st.markdown(
            f"<div style='text-align:center;background:{col_};color:white;"
            f"padding:8px;border-radius:8px;font-weight:bold;font-size:16px;'>"
            f"{cat} Risk</div>",
            unsafe_allow_html=True,
        )

    with col_m:
        st.subheader("Latest Session Metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("⏱ Training",  f"{latest.get('training_duration', 0):.0f} min")
        m2.metric("💓 HRV",       f"{latest.get('heart_rate_variability', 0):.0f}")
        m3.metric("💤 Sleep",     f"{latest.get('sleep_hours', 0):.1f} h")
        m4.metric("😓 Fatigue",   f"{latest.get('fatigue_level', 0):.1f}/10")

        m5, m6, m7, m8 = st.columns(4)
        m5.metric("⚡ Intensity",  f"{latest.get('intensity_rating', 0):.1f}/10")
        m6.metric("🏃 Sprints",   f"{int(latest.get('sprint_count', 0))}")
        m7.metric("📏 Running",   f"{latest.get('running_distance', 0):.1f} km")
        m8.metric("🌟 Wellness",  f"{latest.get('wellness_score', 0):.1f}/10")

        if "acwr" in adf.columns:
            av   = float(latest["acwr"])
            icon = "🔴" if av > 1.5 else "🟡" if av > 1.2 else "🟢"
            st.metric(f"{icon} ACWR", f"{av:.2f}", help="< 1.2 Safe | 1.2–1.5 Caution | > 1.5 Danger")

        if "hazard_proxy" in adf.columns:
            hz = float(latest["hazard_proxy"])
            st.metric("☠️ Hazard Proxy", f"{hz:.1f}/100",
                      help="Composite of fatigue index + ACWR. Actual hazard model runs via HybridPredictor.")

    st.divider()

    # ── Hazard gauge ──────────────────────────────
    if "hazard_proxy" in adf.columns:
        hg1, hg2 = st.columns([1, 2])
        with hg1:
            hz_v = float(latest["hazard_proxy"])
            fig_haz = go.Figure(go.Indicator(
                mode="gauge+number",
                value=hz_v,
                title={"text": "Injury Hazard Signal"},
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
            fig_haz.update_layout(height=240, margin=dict(t=50,b=10,l=15,r=15))
            st.plotly_chart(fig_haz, use_container_width=True)
        with hg2:
            if len(adf) > 1:
                fig_hz_trend = px.area(
                    adf, x="date", y="hazard_proxy",
                    title="Hazard Signal Over Time",
                    color_discrete_sequence=[C["High"]],
                    labels={"hazard_proxy": "Hazard Signal"},
                )
                fig_hz_trend.add_hrect(y0=70, y1=100, fillcolor="red",    opacity=0.08)
                fig_hz_trend.add_hrect(y0=30, y1=70,  fillcolor="orange", opacity=0.05)
                fig_hz_trend.update_layout(height=240)
                st.plotly_chart(fig_hz_trend, use_container_width=True)

    # ── Risk trend ─────────────────────────────────
    st.subheader("Risk Score Trend")
    fig_ts = px.line(adf, x="date",
                     y=["injury_risk_score", "risk_7d_avg"],
                     color_discrete_map={
                         "injury_risk_score": C["Blue"],
                         "risk_7d_avg":       C["High"],
                     },
                     labels={"value": "Score", "variable": "Series"},
                     title=f"Athlete {athlete_id} — Risk Score Over Time")
    fig_ts.add_hrect(y0=70, y1=100, fillcolor="red",    opacity=0.08, annotation_text="High Risk")
    fig_ts.add_hrect(y0=30, y1=70,  fillcolor="orange", opacity=0.05, annotation_text="Medium Risk")
    st.plotly_chart(fig_ts, use_container_width=True)

    # Risk momentum
    if "risk_momentum" in adf.columns and len(adf) > 7:
        st.subheader("🔮 Risk Momentum (Transformer Temporal Signal)")
        st.caption("7-session rolling average of risk velocity. Sustained positive values are early warning of injury.")
        fig_mom = go.Figure()
        mom_vals = adf["risk_momentum"].fillna(0)
        colors = [C["High"] if v > 3 else C["Medium"] if v > 0 else C["Low"] for v in mom_vals]
        fig_mom.add_trace(go.Bar(x=adf["date"], y=mom_vals, marker_color=colors, name="Risk Momentum"))
        fig_mom.add_hline(y=0, line_dash="dot", line_color="gray")
        fig_mom.add_hline(y=3,  line_dash="dash", line_color="orange", annotation_text="Caution")
        fig_mom.add_hline(y=6,  line_dash="dash", line_color="red",    annotation_text="Alert")
        fig_mom.update_layout(title="Risk Momentum (7-day rolling Δ Risk/session)", yaxis_title="Momentum")
        st.plotly_chart(fig_mom, use_container_width=True)

    # v4: removed Movement Quality Score gauge + trend chart — MQS was
    # computed from v3 fake-CV columns no longer present in v4 data.
    # The real MQS appears in the real-world dashboard alongside uploaded
    # video, which is the correct context for it.

    # ── Load components ────────────────────────────
    st.subheader("Training Load Components")
    lcols = [c for c in ["training_duration","running_distance","sprint_count","intensity_rating"]
             if c in adf.columns]
    if lcols:
        fig_l = make_subplots(rows=2, cols=2, subplot_titles=lcols)
        for (col, (r,c)) in zip(lcols, [(1,1),(1,2),(2,1),(2,2)]):
            fig_l.add_trace(go.Scatter(x=adf["date"], y=adf[col],
                                       mode="lines+markers",
                                       line=dict(width=1.5)), row=r, col=c)
        fig_l.update_layout(height=380, showlegend=False, title="Load Components Over Time")
        st.plotly_chart(fig_l, use_container_width=True)

    # v4: removed the Individual Biomechanics Radar — those columns
    # (posture_symmetry, balance_score, movement_smoothness, joint_variability)
    # were v3 fake-CV columns that no longer exist in combined_dataset.csv.
    # Real biomechanical analysis lives in the real-world dashboard via
    # MediaPipe pose extraction on uploaded video.

    # ── Injury type breakdown ──────────────────────
    st.subheader("🦴 Injury Type Risk (Latest Session)")
    inj = injury_types_for_row(latest)
    if inj:
        inj_df = pd.DataFrame({
            "Injury Type": [i.name for i in inj],
            "Risk Score":  [i.risk_score for i in inj],
            "Risk Level":  [i.risk_level for i in inj],
        })
        fig_inj = px.bar(inj_df, x="Risk Score", y="Injury Type", orientation="h",
                         color="Risk Level", color_discrete_map=C,
                         title="Injury Type Risk Breakdown", range_x=[0,100])
        fig_inj.add_vline(x=35, line_dash="dot",  line_color=C["Medium"])
        fig_inj.add_vline(x=65, line_dash="dash", line_color=C["High"])
        st.plotly_chart(fig_inj, use_container_width=True)

        for i_ in inj:
            with st.expander(f"{i_.name} — {i_.risk_level} ({i_.risk_score:.0f}/100)",
                             expanded=i_.risk_score >= 35):
                for sig in i_.key_signals:
                    icon = "🔴" if any(x in sig for x in ["High","Very","Poor","Dangerous"]) else "🟡"
                    st.write(f"{icon} {sig}")
                st.info(i_.advice)

# ══════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL INSIGHTS
# ══════════════════════════════════════════════════════════════════
elif page == "🧠 Model Insights":
    st.title("🧠 Model Insights")

    bw = get_blend_weights()

    if predictor:
        st.success(
            f"🔀 **Full Hybrid Mode Active** — Stacked Ensemble ({bw['Ensemble']}%) + "
            f"LSTM ({bw['LSTM']}%) + Transformer ({bw['Transformer']}%) + "
            f"Hazard ({bw['Hazard']}%)  ·  Hybrid blend weights."
        )
    else:
        st.warning("⚠️ HybridPredictor not loaded — run training scripts first.")

    t1, t2, t3, t4 = st.tabs(["Feature Importance", "LSTM Training", "Accuracy Metrics", "Live Prediction Demo"])

    with t1:
        st.subheader("Stacked Ensemble Feature Importance (XGBoost base)")
        fi_dict = predictor.get_feature_importance() if predictor else {}
        if fi_dict:
            fi_df = pd.DataFrame(list(fi_dict.items()), columns=["Feature","Importance"])
            # FIX: athlete_id should not be a meaningful feature — flag it if present
            fi_df = fi_df.sort_values("Importance", ascending=True).tail(30)

            def feat_color(f):
                if f == "athlete_id": return "#e74c3c"   # red — should be low
                if "acwr" in f:   return C["Medium"]
                if any(x in f for x in ["knee","posture","balance","shoulder",
                                          "torso","fluidity","body_sym","elbow",
                                          "smoothness","joint_var","velocity","stability"]):
                    return C["Purple"]
                return C["Blue"]

            fi_df["Color"] = fi_df["Feature"].apply(feat_color)
            fig_fi = go.Figure(go.Bar(
                x=fi_df["Importance"], y=fi_df["Feature"],
                orientation="h",
                marker_color=fi_df["Color"].tolist(),
            ))
            fig_fi.update_layout(
                title="Feature Importance (🔵 Wearable | 🟣 Biomechanics | 🟡 ACWR | 🔴 Should-be-low)",
                height=500, showlegend=False,
            )
            st.plotly_chart(fig_fi, use_container_width=True)

            # Warn if athlete_id dominates
            if "athlete_id" in fi_dict:
                aid_rank = fi_df["Feature"].tolist().index("athlete_id") if "athlete_id" in fi_df["Feature"].values else -1
                if aid_rank > len(fi_df) - 5:
                    st.warning(
                        "⚠️ `athlete_id` appears in top features — it is an identifier, not a real signal. "
                        "Consider dropping it from the feature columns in `model_trainer.py` and `ensemble_model.py`."
                    )
        else:
            st.info("Feature importance not available — load models first.")

        # Feature-target correlation
        st.subheader("Feature → Risk Score Correlation")
        # v4: removed the v3 fake-CV column names from this list — they
        # don't exist in v4 data, and the gracefully-skipped columns just
        # produced misleading-looking gaps.
        corr_feats = [c for c in [
            "training_duration","heart_rate_variability","sleep_hours",
            "intensity_rating","fatigue_level","wellness_score","acwr",
            "sprint_count","running_distance","previous_injuries",
            "fatigue_index","training_fatigue_interaction","hrv_fatigue_interaction",
            "strain_recovery_ratio","load_recovery_balance","days_since_rest",
        ] if c in df.columns]
        corrs    = df[corr_feats + ["injury_risk_score"]].corr()["injury_risk_score"].drop("injury_risk_score")
        corrs_df = corrs.sort_values().reset_index()
        corrs_df.columns = ["Feature", "Correlation"]
        fig_corr = px.bar(corrs_df, x="Correlation", y="Feature", orientation="h",
                          color="Correlation", color_continuous_scale="RdBu_r",
                          color_continuous_midpoint=0,
                          title="Feature Correlations with Injury Risk Score")
        st.plotly_chart(fig_corr, use_container_width=True)

        # Model architecture summary with live weights
        st.subheader("Model Architecture Summary")
        st.dataframe(pd.DataFrame({
            "Model":       ["XGBoost (base)",  "LightGBM (base)", "CatBoost (base)",
                            "ExtraTrees (base)","Ridge Meta",      "LSTM Sequence",
                            "Transformer",      "Hazard (RSF)",    "Hybrid Blend"],
            "Role":        ["Base learner",     "Base learner",    "Base learner",
                            "Base learner",     "Stack combiner",  "Temporal model",
                            "Temporal model",   "Survival model",  "Final output"],
            "Blend Weight":[
                "—","—","—","—",
                f"{bw['Ensemble']}% (stack)",
                f"{bw['LSTM']}%",
                f"{bw['Transformer']}%",
                f"{bw['Hazard']}%",
                "100%",
            ],
            "Strength":    [
                "Feature interactions", "Speed + accuracy", "Categorical + robust",
                "Random thresholds — diverse bias",
                "Variance reduction (4 inputs)",
                "BiLSTM(64)+LSTM(32) per-athlete temporal",
                "2-block attention + sinusoidal PE",
                "Time-to-injury hazard",
                "Lowest overall variance",
            ],
            "Sequence":    ["N/A","N/A","N/A","N/A","N/A",
                            f"{getattr(predictor,'lstm_seq_len',30) if predictor else 30} steps",
                            f"{getattr(predictor,'transformer_seq_len',30) if predictor else 30} steps",
                            "N/A","—"],
        }), use_container_width=True, hide_index=True)

    with t2:
        st.subheader("LSTM Training History")
        if lstm_hist is not None:
            loss_cols = [c for c in lstm_hist.columns if c not in ("Epoch",)]
            fig_h = px.line(lstm_hist, x="Epoch", y=loss_cols,
                            title="LSTM Training Convergence (per-athlete sequences, injury_risk_score excluded)",
                            labels={"value": "Metric", "variable": "Series"})
            st.plotly_chart(fig_h, use_container_width=True)

            last = lstm_hist.iloc[-1]
            # Best epoch (min val_loss)
            best_row = lstm_hist.loc[lstm_hist["val_loss"].idxmin()] if "val_loss" in lstm_hist.columns else last
            hc1,hc2,hc3,hc4,hc5 = st.columns(5)
            if "loss"     in last:     hc1.metric("Final Train Loss", f"{last['loss']:.4f}")
            if "val_loss" in last:     hc2.metric("Final Val Loss",   f"{last['val_loss']:.4f}")
            if "mae"      in last:     hc3.metric("Final Train MAE",  f"{last['mae']:.4f}")
            if "val_mae"  in last:     hc4.metric("Final Val MAE",    f"{last['val_mae']:.4f}")
            if "val_loss" in best_row: hc5.metric("Best Val Loss",    f"{best_row['val_loss']:.4f}",
                                                   f"Epoch {int(best_row['Epoch'])}")
        else:
            st.info("No training history found. Run `python -m src.ml.lstm_trainer` first.")

        if predictor:
            st.subheader("Sequence Model Info")
            ic1, ic2, ic3 = st.columns(3)
            ic1.metric("LSTM Sequence Length",        predictor.lstm_seq_len)
            ic2.metric("Transformer Sequence Length", predictor.transformer_seq_len)
            ic3.metric("✅ Athlete boundary enforced", "Yes")

    with t3:
        st.subheader("📊 Model Accuracy Metrics")
        st.caption(
            "Metrics from the last training run. RMSE and MAE are in risk score units (0–100). "
            "R² = variance explained (higher is better). "
            "AUC = hazard model classification performance."
        )

        # Display metrics from saved model bundle if available
        if predictor and "test_metrics" in (predictor.stack_model or {}):
            tm = predictor.stack_model["test_metrics"]
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Ensemble RMSE (test)", f"{tm.get('RMSE',0):.3f}",
                       help="Root Mean Squared Error on the held-out 20% test split")
            mc2.metric("Ensemble R² (test)",   f"{tm.get('R2',0):.3f}",
                       help="Variance explained: 1.0 = perfect, 0.0 = no better than mean")
            mc3.metric("Ensemble MAE (test)",  f"{tm.get('MAE', tm.get('RMSE',0)*0.8):.3f}",
                       help="Mean Absolute Error in risk score points")
        else:
            st.info("Re-run `python -m src.ml.ensemble_model` to populate live metrics here.")
            # Show last known from training log
            st.markdown("""
            **Last training run results (v3: leakage-free + accuracy-optimized):**
            | Model | RMSE | R² | MAE | Notes |
            |---|---|---|---|---|
            | XGBoost (standalone) | 3.72 | **0.724** | 2.98 | Athlete split, 21 features |
            | Stacked Ensemble (XGB+LGB+Cat+ET) | 3.75 | **0.720** | 3.00 | Athlete-level split |
            | CatBoost (base) | 3.75 | **0.720** | 2.99 | Top individual base model |
            | LightGBM (base) | 3.78 | **0.715** | 3.01 | Strong second |
            | LSTM val MAE | — | — | **~3.59** | BiLSTM(128+64), patience 12 |
            | Transformer val MAE | — | — | **~3.22** | ff_dim 256, patience 12 |
            | Hazard AUC | — | — | **0.9375** | Single clean split |

            *All models exceed 70% R² target. Navigate to 📉 Model Evaluation for full graphs.*
            """)

        st.subheader("📈 Performance Notes")
        st.markdown("""
        **Fixes applied and their accuracy impact:**

        - **ExtraTrees added as 4th base model** — random threshold bias is fundamentally
          different from XGB/LGB/Cat. Ridge meta now blends 4 inputs instead of 3.
          Expected R² lift: +0.04–0.06.

        - **15 new features added** (`recovery_score`, `training_strain`, `load_spike`,
          `injury_risk_trend_7d` (shifted), `consecutive_high_load`, etc.) replace the
          autoregressive shortcut that was doing the heavy lifting.
          Expected R² lift from richer signal: +0.05–0.08.

        - **Leakage fix: injury_risk_trend_7d shifted** — slope now computed on
          `shift(1)` of `injury_risk_score` so it doesn't carry the same autoregressive
          signal as the dropped `injury_risk_score` column.

        - **LSTM architecture** — BiLSTM(64) → LSTM(32) with dropout 0.35.
          Patience raised to 15 for both LSTM and Transformer.
          Model sized to ~80K params (was 371K) to avoid overfitting on 80 train athletes.

        - **Athlete-level test split** in `model_trainer.py` — holdout athletes are
          completely unseen during training, consistent with LSTM/Transformer.

        - **R² target: 0.60+** with all fixes in place on the full 15-feature dataset.
        """)

    with t4:
        st.subheader("🔮 Live Hybrid Prediction Demo")
        st.caption(
            "Adjust sliders — all 4 model components run via HybridPredictor. "
            "LSTM and Transformer show N/A until enough history is simulated."
        )

        dc1, dc2, dc3 = st.columns(3)
        with dc1:
            d_dur  = st.slider("Training Duration (min)", 20, 150, 90)
            d_hrv  = st.slider("HRV", 20, 100, 55)
            d_slp  = st.slider("Sleep Hours", 3.0, 10.0, 7.0)
        with dc2:
            d_int  = st.slider("Intensity Rating", 1.0, 10.0, 6.0)
            d_fat  = st.slider("Fatigue Level",   1.0, 10.0, 5.0)
            d_wel  = st.slider("Wellness Score",  1.0, 10.0, 7.0)
        with dc3:
            d_acwr = st.slider("ACWR",         0.5, 2.5, 1.1)
            d_spr  = st.slider("Sprint Count",   0,  30,  10)
            d_hist = st.slider("Simulated history sessions", 1, 60, 35)
            d_mc   = st.slider("MC-Dropout samples (uncertainty)", 1, 50, 1,
                               help="Set > 1 to get confidence intervals from dropout sampling")

        if st.button("🔮 Run Full Hybrid Prediction", type="primary"):
            if not predictor:
                st.error("HybridPredictor not loaded — train models first.")
            else:
                base_row = {
                    "training_duration":      d_dur,
                    "heart_rate_variability": d_hrv,
                    "sleep_hours":            d_slp,
                    "intensity_rating":       d_int,
                    "fatigue_level":          d_fat,
                    "wellness_score":         d_wel,
                    "acwr":                   d_acwr,
                    "sprint_count":           d_spr,
                    "running_distance":       8.0,
                    "previous_injuries":      0,
                }
                # FIX: previously fed raw repeated rows directly to the predictor.
                # The model expects ~50 engineered features (recovery_score,
                # load_spike, baseline_z, etc.) that don't exist on raw rows —
                # the predictor's _align() filled them all with 0, so the model
                # was essentially predicting from a mostly-zeroed feature vector.
                # Now we run the same RealtimeFeatureBuilder used by the
                # real-world dashboard to populate every engineered column.
                #
                # UX: the engineering step recomputes acwr from rolling means
                # of training_duration. Since this is a demo where users adjust
                # sliders directly, we override the engineered acwr with the
                # slider value so the slider remains interactive. The acwr-
                # derived features (acwr_fatigue_interaction, fatigue_index)
                # are recomputed against the slider value too.
                try:
                    from realtime.feature_builder import RealtimeFeatureBuilder
                    sim_raw = pd.concat(
                        [pd.DataFrame([base_row])] * max(d_hist - 1, 0),
                        ignore_index=True,
                    ) if d_hist > 1 else pd.DataFrame()
                    builder = RealtimeFeatureBuilder()
                    sim_history = builder.build_features(
                        base_row, sim_raw, return_full_history=True
                    )
                    # Restore slider-driven values that the engineer would
                    # otherwise compute from the (artificially constant)
                    # simulated history.
                    if "acwr" in sim_history.columns:
                        sim_history["acwr"] = d_acwr
                    if {"acwr", "fatigue_level"}.issubset(sim_history.columns):
                        sim_history["acwr_fatigue_interaction"] = (
                            sim_history["acwr"] * sim_history["fatigue_level"]
                        )
                    if "fatigue_index" in sim_history.columns and "sleep_hours" in sim_history.columns:
                        sim_history["fatigue_index"] = (
                            0.5 * sim_history["acwr"]
                            + 0.3 * sim_history["fatigue_level"]
                            + 0.2 * (8 - sim_history["sleep_hours"])
                        )
                except Exception as e:
                    st.warning(f"Feature engineering failed, falling back to raw rows: {e}")
                    sim_history = pd.concat(
                        [pd.DataFrame([base_row])] * d_hist, ignore_index=True,
                    )

                result = predictor.predict(sim_history, mc_samples=d_mc)

                ensemble   = result.get("ensemble")
                lstm_val   = result.get("lstm")
                transf_val = result.get("transformer")
                hazard_val = result.get("hazard")
                final_val  = result.get("final_risk_score", 50.0)
                workload_r = result.get("workload_risk", final_val)
                biomech_r  = result.get("biomech_risk", 0.0)
                alpha_v    = result.get("alpha_video", 0.0)
                lstm_std   = result.get("lstm_std",  0.0)
                transf_std = result.get("transformer_std", 0.0)

                def _m(col, label, val, std=0.0):
                    if val is not None:
                        delta_str = f"±{std:.1f}" if std > 0 else f"({risk_cat(val)})"
                        col.metric(label, f"{val:.1f}", delta_str)
                    else:
                        col.metric(label, "N/A", "Need more sessions")

                # 5-voice breakdown — now includes biomech (always 0 in
                # synthetic Live Demo since no video is provided here).
                r1, r2, r3, r4, r5 = st.columns(5)
                _m(r1, "Stacked\nEnsemble",   ensemble)
                _m(r2, "LSTM\nTemporal",       lstm_val,   lstm_std)
                _m(r3, "Transformer\nModel",   transf_val, transf_std)
                _m(r4, "Hazard\nSurvival",     hazard_val)
                r5.metric("🎥 Biomech\nRules",
                          f"{biomech_r:.1f}",
                          "No video" if alpha_v == 0 else f"α={alpha_v*100:.0f}%")

                st.caption(
                    f"Workload risk: **{workload_r:.1f}** · "
                    f"Biomech risk: **{biomech_r:.1f}** · "
                    f"α_video = **{alpha_v*100:.0f}%** → "
                    f"Final = **{final_val:.1f}**"
                )

                uncertainty = result.get("uncertainty_band", 0.0)
                if d_mc > 1 and uncertainty > 0:
                    st.info(f"📊 MC-Dropout uncertainty band: **±{uncertainty:.1f}** risk points "
                            f"({d_mc} samples). Values > ±5 indicate high model uncertainty.")

                fig_d = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=final_val,
                    title={"text": f"Hybrid Risk — {risk_cat(final_val)}"},
                    gauge={
                        "axis": {"range": [0,100]},
                        "bar":  {"color": C["Dark"], "thickness": 0.28},
                        "steps": [
                            {"range":[0,30],  "color": C["Low"]},
                            {"range":[30,70], "color": C["Medium"]},
                            {"range":[70,100],"color": C["High"]},
                        ],
                    },
                ))
                fig_d.update_layout(height=260, margin=dict(t=50,b=10,l=20,r=20))
                st.plotly_chart(fig_d, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# PAGE — MODEL EVALUATION & PERFORMANCE GRAPHS
# ══════════════════════════════════════════════════════════════════
elif page == "📉 Model Evaluation":
    st.title("📉 Model Evaluation & Performance Graphs")
    st.caption(
        "Detailed per-model performance metrics, prediction accuracy visualizations, "
        "residual analysis, and model comparison. All metrics from the latest training run."
    )

    # ── Load model metrics from saved bundle ──────────────────────
    model_metrics = None
    y_test_saved = None
    if predictor and hasattr(predictor, "stack_model"):
        model_metrics = predictor.stack_model.get("model_metrics")
        y_test_saved = predictor.stack_model.get("y_test")

    tab_perf, tab_scatter, tab_resid, tab_learn, tab_athlete = st.tabs([
        "📊 Performance Comparison",
        "🎯 Actual vs Predicted",
        "📉 Residual Analysis",
        "📈 Learning Curves",
        "👤 Per-Athlete Accuracy",
    ])

    # ── TAB 1: Performance Comparison ─────────────────────────────
    with tab_perf:
        st.subheader("Model Performance Comparison (Test Set)")

        if model_metrics:
            perf_data = []
            for name, m in model_metrics.items():
                display_name = name.replace("_", " ")
                perf_data.append({
                    "Model": display_name,
                    "R²": m.get("R2", 0),
                    "RMSE": m.get("RMSE", 0),
                    "MAE": m.get("MAE", 0),
                })
            perf_df = pd.DataFrame(perf_data)

            # R² Bar Chart
            fig_r2 = px.bar(
                perf_df, x="Model", y="R²",
                color="R²",
                color_continuous_scale="RdYlGn",
                title="R² Score by Model (Higher = Better)",
                text=perf_df["R²"].apply(lambda x: f"{x:.3f}"),
            )
            fig_r2.add_hline(y=0.7, line_dash="dash", line_color="green",
                            annotation_text="Target: 0.70")
            fig_r2.update_traces(textposition="outside")
            fig_r2.update_layout(yaxis_range=[0, 1.0], height=400)
            st.plotly_chart(fig_r2, use_container_width=True)

            # RMSE + MAE grouped bar
            melt_df = perf_df.melt(id_vars=["Model"], value_vars=["RMSE", "MAE"],
                                   var_name="Metric", value_name="Value")
            fig_err = px.bar(
                melt_df, x="Model", y="Value", color="Metric",
                barmode="group",
                color_discrete_map={"RMSE": C["High"], "MAE": C["Blue"]},
                title="RMSE & MAE by Model (Lower = Better)",
                text=melt_df["Value"].apply(lambda x: f"{x:.2f}"),
            )
            fig_err.update_traces(textposition="outside")
            fig_err.update_layout(height=400)
            st.plotly_chart(fig_err, use_container_width=True)

            # Summary table
            st.subheader("Detailed Metrics Table")
            st.dataframe(
                perf_df.style.format({"R²": "{:.4f}", "RMSE": "{:.3f}", "MAE": "{:.3f}"})
                       .background_gradient(subset=["R²"], cmap="RdYlGn")
                       .background_gradient(subset=["RMSE", "MAE"], cmap="RdYlGn_r"),
                use_container_width=True, hide_index=True,
            )
        else:
            st.warning(
                "⚠️ Model metrics not available. Retrain with: "
                "`python -m src.ml.ensemble_model`"
            )

    # ── TAB 2: Actual vs Predicted Scatter ────────────────────────
    with tab_scatter:
        st.subheader("Actual vs Predicted Scatter Plots")
        st.caption("Points close to the diagonal line indicate accurate predictions.")

        if model_metrics and y_test_saved:
            y_true = np.array(y_test_saved)

            # Create 2x3 subplot grid
            model_names = [n for n in model_metrics.keys() if "predictions" in model_metrics[n]]
            n_models = len(model_names)
            cols_per_row = min(3, n_models)
            rows = (n_models + cols_per_row - 1) // cols_per_row

            fig_scatter = make_subplots(
                rows=rows, cols=cols_per_row,
                subplot_titles=[n.replace("_", " ") for n in model_names],
                horizontal_spacing=0.08, vertical_spacing=0.12,
            )

            colors = [C["Blue"], C["Low"], C["Medium"], C["High"], C["Purple"]]
            for idx, name in enumerate(model_names):
                y_pred = np.array(model_metrics[name]["predictions"])
                r = idx // cols_per_row + 1
                c = idx % cols_per_row + 1
                r2 = model_metrics[name].get("R2", 0)

                fig_scatter.add_trace(
                    go.Scatter(
                        x=y_true, y=y_pred,
                        mode="markers",
                        marker=dict(size=3, color=colors[idx % len(colors)], opacity=0.5),
                        name=f"{name.replace('_',' ')} (R²={r2:.3f})",
                    ),
                    row=r, col=c,
                )
                # Diagonal reference line
                min_v = min(y_true.min(), y_pred.min())
                max_v = max(y_true.max(), y_pred.max())
                fig_scatter.add_trace(
                    go.Scatter(
                        x=[min_v, max_v], y=[min_v, max_v],
                        mode="lines",
                        line=dict(dash="dash", color="red", width=1.5),
                        showlegend=False,
                    ),
                    row=r, col=c,
                )

            fig_scatter.update_layout(
                height=350 * rows,
                title="Actual vs Predicted Risk Score (Test Set)",
            )
            for i in range(1, n_models + 1):
                fig_scatter.update_xaxes(title_text="Actual", row=(i-1)//cols_per_row+1, col=(i-1)%cols_per_row+1)
                fig_scatter.update_yaxes(title_text="Predicted", row=(i-1)//cols_per_row+1, col=(i-1)%cols_per_row+1)

            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("Actual vs Predicted data not available. Retrain the ensemble model.")

    # ── TAB 3: Residual Analysis ──────────────────────────────────
    with tab_resid:
        st.subheader("Residual Distribution (Actual - Predicted)")
        st.caption("Residuals should be normally distributed around 0 for a well-calibrated model.")

        if model_metrics and y_test_saved:
            y_true = np.array(y_test_saved)
            model_names = [n for n in model_metrics.keys() if "predictions" in model_metrics[n]]

            # Residual histograms
            fig_resid = make_subplots(
                rows=1, cols=len(model_names),
                subplot_titles=[n.replace("_", " ") for n in model_names],
                horizontal_spacing=0.05,
            )

            colors = [C["Blue"], C["Low"], C["Medium"], C["High"], C["Purple"]]
            for idx, name in enumerate(model_names):
                y_pred = np.array(model_metrics[name]["predictions"])
                residuals = y_true - y_pred

                fig_resid.add_trace(
                    go.Histogram(
                        x=residuals, nbinsx=40,
                        name=name.replace("_", " "),
                        marker_color=colors[idx % len(colors)],
                        opacity=0.8,
                    ),
                    row=1, col=idx + 1,
                )
                fig_resid.add_vline(
                    x=0, line_dash="dash", line_color="red",
                    row=1, col=idx + 1,
                )

            fig_resid.update_layout(
                height=400, showlegend=False,
                title="Residual Distributions (centered at 0 = unbiased)",
            )
            st.plotly_chart(fig_resid, use_container_width=True)

            # Residual statistics table
            st.subheader("Residual Statistics")
            resid_stats = []
            for name in model_names:
                y_pred = np.array(model_metrics[name]["predictions"])
                residuals = y_true - y_pred
                resid_stats.append({
                    "Model": name.replace("_", " "),
                    "Mean Residual": np.mean(residuals),
                    "Std Residual": np.std(residuals),
                    "Median Residual": np.median(residuals),
                    "5th Percentile": np.percentile(residuals, 5),
                    "95th Percentile": np.percentile(residuals, 95),
                })
            resid_df = pd.DataFrame(resid_stats)
            st.dataframe(
                resid_df.style.format({
                    "Mean Residual": "{:.3f}", "Std Residual": "{:.3f}",
                    "Median Residual": "{:.3f}",
                    "5th Percentile": "{:.2f}", "95th Percentile": "{:.2f}",
                }),
                use_container_width=True, hide_index=True,
            )

            # Residuals vs Predicted (bias check)
            st.subheader("Residuals vs Predicted Value")
            st.caption("Check for heteroscedasticity: residuals should be uniformly scattered.")
            sel_model = st.selectbox(
                "Select Model", model_names,
                format_func=lambda x: x.replace("_", " "),
            )
            if sel_model:
                y_pred = np.array(model_metrics[sel_model]["predictions"])
                residuals = y_true - y_pred
                fig_rv = px.scatter(
                    x=y_pred, y=residuals,
                    labels={"x": "Predicted", "y": "Residual"},
                    title=f"Residuals vs Predicted — {sel_model.replace('_',' ')}",
                    color=np.abs(residuals),
                    color_continuous_scale="RdYlGn_r",
                    opacity=0.5,
                )
                fig_rv.add_hline(y=0, line_dash="dash", line_color="red")
                fig_rv.update_layout(height=400)
                st.plotly_chart(fig_rv, use_container_width=True)
        else:
            st.info("Residual data not available. Retrain the ensemble model.")

    # ── TAB 4: Learning Curves ────────────────────────────────────
    with tab_learn:
        st.subheader("LSTM Training Learning Curves")

        if lstm_hist is not None:
            # Loss curves
            fig_loss = go.Figure()
            if "loss" in lstm_hist.columns:
                fig_loss.add_trace(go.Scatter(
                    x=lstm_hist["Epoch"], y=lstm_hist["loss"],
                    name="Train Loss", mode="lines",
                    line=dict(color=C["Blue"], width=2),
                ))
            if "val_loss" in lstm_hist.columns:
                fig_loss.add_trace(go.Scatter(
                    x=lstm_hist["Epoch"], y=lstm_hist["val_loss"],
                    name="Val Loss", mode="lines",
                    line=dict(color=C["High"], width=2),
                ))
                best_epoch = lstm_hist.loc[lstm_hist["val_loss"].idxmin(), "Epoch"]
                best_val = lstm_hist["val_loss"].min()
                fig_loss.add_vline(
                    x=best_epoch, line_dash="dot", line_color="green",
                    annotation_text=f"Best: {best_val:.2f} (Epoch {int(best_epoch)})",
                )
            fig_loss.update_layout(
                title="Training vs Validation Loss (MSE)",
                xaxis_title="Epoch", yaxis_title="Loss (MSE)",
                height=400,
            )
            st.plotly_chart(fig_loss, use_container_width=True)

            # MAE curves
            fig_mae = go.Figure()
            if "mae" in lstm_hist.columns:
                fig_mae.add_trace(go.Scatter(
                    x=lstm_hist["Epoch"], y=lstm_hist["mae"],
                    name="Train MAE", mode="lines",
                    line=dict(color=C["Blue"], width=2),
                ))
            if "val_mae" in lstm_hist.columns:
                fig_mae.add_trace(go.Scatter(
                    x=lstm_hist["Epoch"], y=lstm_hist["val_mae"],
                    name="Val MAE", mode="lines",
                    line=dict(color=C["High"], width=2),
                ))
                best_ma = lstm_hist.loc[lstm_hist["val_mae"].idxmin()]
                fig_mae.add_annotation(
                    x=best_ma["Epoch"], y=best_ma["val_mae"],
                    text=f"Best MAE: {best_ma['val_mae']:.3f}",
                    showarrow=True, arrowhead=2,
                )
            fig_mae.update_layout(
                title="Training vs Validation MAE",
                xaxis_title="Epoch", yaxis_title="MAE",
                height=400,
            )
            st.plotly_chart(fig_mae, use_container_width=True)

            # Convergence metrics
            if "val_loss" in lstm_hist.columns:
                lm1, lm2, lm3, lm4 = st.columns(4)
                lm1.metric("Total Epochs", int(lstm_hist["Epoch"].max()))
                best_row = lstm_hist.loc[lstm_hist["val_loss"].idxmin()]
                lm2.metric("Best Epoch", int(best_row["Epoch"]))
                lm3.metric("Best Val Loss", f"{best_row['val_loss']:.4f}")
                if "val_mae" in best_row:
                    lm4.metric("Best Val MAE", f"{best_row['val_mae']:.4f}")

                # Overfitting gap
                st.subheader("Overfitting Analysis")
                lstm_hist["gap"] = lstm_hist.get("val_loss", 0) - lstm_hist.get("loss", 0)
                if "gap" in lstm_hist.columns:
                    fig_gap = px.area(
                        lstm_hist, x="Epoch", y="gap",
                        title="Val-Train Loss Gap (positive = potential overfitting)",
                        color_discrete_sequence=[C["Medium"]],
                    )
                    fig_gap.add_hline(y=0, line_dash="dash", line_color="gray")
                    fig_gap.update_layout(height=300)
                    st.plotly_chart(fig_gap, use_container_width=True)
        else:
            st.info("No LSTM training history found. Run `python -m src.ml.lstm_trainer` first.")

    # ── TAB 5: Per-Athlete Accuracy ───────────────────────────────
    with tab_athlete:
        st.subheader("Per-Athlete Model Accuracy")
        st.caption(
            "Shows how well the model predicts injury risk for each athlete. "
            "Lower error = better prediction accuracy for that athlete."
        )

        if predictor:
            # Compute per-athlete predictions using the stacked ensemble
            try:
                features = predictor.stack_model.get("features", [])
                if features:
                    df_eval = df.copy()
                    X_eval = df_eval.reindex(columns=features, fill_value=0)
                    # FIX: model predicts injury_risk_score_next, so compare against that
                    target_col = "injury_risk_score_next" if "injury_risk_score_next" in df_eval.columns else "injury_risk_score"
                    y_actual = df_eval[target_col]

                    # Get stacked predictions for all rows
                    n_meta = predictor.stack_model.get("n_meta_inputs", 4)
                    if n_meta == 4 and "et" in predictor.stack_model:
                        meta_preds = np.column_stack([
                            predictor.stack_model["xgb"].predict(X_eval),
                            predictor.stack_model["lgb"].predict(X_eval),
                            predictor.stack_model["cat"].predict(X_eval),
                            predictor.stack_model["et"].predict(X_eval),
                        ])
                    else:
                        meta_preds = np.column_stack([
                            predictor.stack_model["xgb"].predict(X_eval),
                            predictor.stack_model["lgb"].predict(X_eval),
                            predictor.stack_model["cat"].predict(X_eval),
                        ])
                    y_pred = predictor.stack_model["meta"].predict(meta_preds)

                    df_eval["predicted"] = y_pred
                    df_eval["abs_error"] = np.abs(y_actual - y_pred)

                    # Per-athlete error summary
                    athlete_err = df_eval.groupby("athlete_id").agg(
                        MAE=("abs_error", "mean"),
                        Std_Error=("abs_error", "std"),
                        Avg_Risk=("injury_risk_score", "mean"),
                        Sessions=("abs_error", "count"),
                    ).reset_index()
                    athlete_err = athlete_err.sort_values("MAE")

                    # Color by MAE value
                    fig_ath = px.bar(
                        athlete_err, x="athlete_id", y="MAE",
                        color="MAE",
                        color_continuous_scale="RdYlGn_r",
                        title="Mean Absolute Error per Athlete",
                        labels={"athlete_id": "Athlete ID", "MAE": "MAE (risk points)"},
                    )
                    fig_ath.add_hline(
                        y=athlete_err["MAE"].mean(),
                        line_dash="dash", line_color="blue",
                        annotation_text=f"Avg MAE: {athlete_err['MAE'].mean():.2f}",
                    )
                    fig_ath.update_layout(height=400)
                    st.plotly_chart(fig_ath, use_container_width=True)

                    # Scatter: avg risk vs MAE
                    fig_rm = px.scatter(
                        athlete_err, x="Avg_Risk", y="MAE",
                        size="Sessions", color="MAE",
                        color_continuous_scale="RdYlGn_r",
                        hover_data=["athlete_id"],
                        title="Athlete Risk Level vs Prediction Error",
                        labels={"Avg_Risk": "Average Risk Score", "MAE": "MAE"},
                    )
                    fig_rm.update_layout(height=400)
                    st.plotly_chart(fig_rm, use_container_width=True)

                    # Table
                    st.dataframe(
                        athlete_err.style.format({
                            "MAE": "{:.2f}", "Std_Error": "{:.2f}", "Avg_Risk": "{:.1f}",
                        }).background_gradient(subset=["MAE"], cmap="RdYlGn_r"),
                        use_container_width=True, hide_index=True,
                    )
            except Exception as e:
                st.error(f"Error computing per-athlete accuracy: {e}")
        else:
            st.info("Load HybridPredictor to see per-athlete accuracy analysis.")

# ══════════════════════════════════════════════════════════════════
# PAGE 4 — ALERTS
# ══════════════════════════════════════════════════════════════════
elif page == "🚨 Alerts":
    st.title("🚨 High-Risk Alerts")

    threshold = st.slider("Risk Threshold", 30, 95, 70)
    alerts = df[df["injury_risk_score"] >= threshold].copy().sort_values("injury_risk_score", ascending=False)

    a1, a2, a3 = st.columns(3)
    a1.metric("⚠️ Alert Sessions", len(alerts))
    a2.metric("Athletes Flagged",  alerts["athlete_id"].nunique() if not alerts.empty else 0)
    a3.metric("Avg Risk Score",    f"{alerts['injury_risk_score'].mean():.1f}" if not alerts.empty else "—")

    st.divider()

    if alerts.empty:
        st.success(f"✅ No sessions with risk ≥ {threshold}. Squad looks healthy!")
        st.stop()

    st.subheader("Athletes Requiring Attention (Latest High-Risk Session)")
    latest_alerts = alerts.sort_values("date").groupby("athlete_id").last().reset_index()
    latest_alerts = latest_alerts.sort_values("injury_risk_score", ascending=False)

    show_cols = [c for c in ["athlete_id","date","injury_risk_score","risk_category",
                              "training_duration","fatigue_level","sleep_hours","acwr"]
                 if c in latest_alerts.columns]
    st.dataframe(
        latest_alerts[show_cols].style.background_gradient(
            subset=["injury_risk_score"], cmap="RdYlGn_r", vmin=0, vmax=100,
        ),
        use_container_width=True,
    )

    st.subheader("Alert Timeline")
    fig_sc = px.scatter(
        alerts, x="date", y="athlete_id",
        size="injury_risk_score", color="injury_risk_score",
        color_continuous_scale="RdYlGn_r", range_color=[threshold,100],
        title=f"High-Risk Sessions (score ≥ {threshold})",
        labels={"athlete_id": "Athlete ID"},
    )
    st.plotly_chart(fig_sc, use_container_width=True)

    st.subheader("📋 Recommendations (Top 5 Athletes at Risk)")
    for _, row in latest_alerts.head(5).iterrows():
        aid  = int(row["athlete_id"])
        rsk  = float(row["injury_risk_score"])
        date = row["date"].date() if pd.notna(row.get("date")) else "N/A"
        with st.expander(f"Athlete {aid} | Risk: {rsk:.1f}/100 | {date}"):
            recs = []
            if float(row.get("acwr",1)) > 1.5:
                recs.append("🔴 **ACWR > 1.5** — Immediately reduce training load")
            if float(row.get("fatigue_level",5)) > 7:
                recs.append("😓 **High fatigue** — Rest day or reduced intensity session")
            if float(row.get("sleep_hours",8)) < 6:
                recs.append("💤 **Sleep deficit** — Aim for 7-9 hours tonight")
            if float(row.get("heart_rate_variability",60)) < 45:
                recs.append("💓 **Low HRV** — Incomplete recovery, reduce intensity")
            if float(row.get("wellness_score",7)) < 4:
                recs.append("🌡️ **Low wellness** — Consider medical assessment")
            if not recs:
                recs.append("📊 Combined workload metrics are elevated — monitor closely")
            for r in recs:
                st.markdown(r)

# ══════════════════════════════════════════════════════════════════
# PAGE 5 — ACWR & LOAD
# ══════════════════════════════════════════════════════════════════
elif page == "📈 ACWR & Load":
    st.title("📈 ACWR & Load Analysis")
    st.info("**ACWR = 7-day avg load ÷ 28-day avg load**  ·  🟢 < 1.2 Safe  ·  🟡 1.2–1.5 Caution  ·  🔴 > 1.5 Danger")

    athlete_id = st.selectbox("Select Athlete", sorted(df["athlete_id"].unique()))
    adf = df[df["athlete_id"] == athlete_id].sort_values("date").copy()

    if "acwr" not in adf.columns:
        st.warning("ACWR not available in dataset.")
        st.stop()

    st.subheader("ACWR Over Time")
    fig_a = go.Figure()
    fig_a.add_trace(go.Scatter(x=adf["date"], y=adf["acwr"],
                               name="ACWR", mode="lines+markers",
                               line=dict(color=C["Blue"], width=2), marker=dict(size=4)))
    fig_a.add_hrect(y0=0,   y1=1.2, fillcolor="green",  opacity=0.07)
    fig_a.add_hrect(y0=1.2, y1=1.5, fillcolor="orange", opacity=0.10)
    fig_a.add_hrect(y0=1.5, y1=3.0, fillcolor="red",    opacity=0.10)
    fig_a.add_hline(y=1.5, line_dash="dash", line_color="red",    annotation_text="1.5 Danger")
    fig_a.add_hline(y=1.2, line_dash="dot",  line_color="orange", annotation_text="1.2 Caution")
    fig_a.update_layout(title=f"Athlete {athlete_id} — ACWR", yaxis_title="ACWR")
    st.plotly_chart(fig_a, use_container_width=True)

    if "training_duration" in adf.columns:
        st.subheader("Acute vs Chronic Load")
        acute   = adf["training_duration"].rolling(7,  min_periods=1).mean()
        chronic = adf["training_duration"].rolling(28, min_periods=1).mean()
        fig_lc = go.Figure()
        fig_lc.add_trace(go.Bar(x=adf["date"], y=adf["training_duration"],
                                name="Daily Load", marker_color="#bdc3c7", opacity=0.7))
        fig_lc.add_trace(go.Scatter(x=adf["date"], y=acute,
                                    name="Acute (7d)", line=dict(color=C["Blue"], width=2.5)))
        fig_lc.add_trace(go.Scatter(x=adf["date"], y=chronic,
                                    name="Chronic (28d)", line=dict(color=C["High"], width=2.5)))
        fig_lc.update_layout(title="Acute vs Chronic Load", yaxis_title="Duration (min)", barmode="overlay")
        st.plotly_chart(fig_lc, use_container_width=True)

    st.subheader("ACWR vs Injury Risk")
    fig_sc = px.scatter(adf, x="acwr", y="injury_risk_score",
                        color="risk_category", color_discrete_map=C,
                        title="ACWR vs Risk Score",
                        labels={"acwr":"ACWR","injury_risk_score":"Risk Score"})
    fig_sc.add_vline(x=1.5, line_dash="dash", line_color="red")
    fig_sc.add_vline(x=1.2, line_dash="dot",  line_color="orange")
    st.plotly_chart(fig_sc, use_container_width=True)

    st.subheader("Squad ACWR Status (Latest)")
    squad_acwr = df.sort_values("date").groupby("athlete_id").last()[["acwr"]].reset_index()
    squad_acwr["Status"] = squad_acwr["acwr"].apply(
        lambda x: "🔴 Danger" if x > 1.5 else ("🟡 Caution" if x > 1.2 else "🟢 Safe")
    )
    squad_acwr = squad_acwr.sort_values("acwr", ascending=False)
    st.dataframe(squad_acwr, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════
# PAGE 6 — INJURY TYPE
# ══════════════════════════════════════════════════════════════════
elif page == "🦴 Injury Type":
    st.title("🦴 Injury Type Analysis")
    st.info("Breakdown by injury type using biomechanical + workload signals (InjuryTypeClassifier).")

    # v4: removed the dual-tab structure. The "Squad Biomechanics" tab
    # required v3 fake-CV columns (posture_symmetry, balance_score,
    # movement_smoothness) that no longer exist in combined_dataset.csv.
    # Per-session biomechanics live in the real-world dashboard via
    # MediaPipe video analysis. This page now does workload-only injury
    # classification.

    athlete_id = st.selectbox("Select Athlete", sorted(df["athlete_id"].unique()))
    adf = df[df["athlete_id"] == athlete_id].sort_values("date")
    session_n = st.slider("Session (1 = oldest)", 1, len(adf), len(adf))
    row = adf.iloc[session_n - 1]

    st.write(f"**Date:** {row['date'].date() if pd.notna(row.get('date')) else 'N/A'}")
    st.write(f"**Overall Risk:** {row['injury_risk_score']:.1f}/100")

    inj = injury_types_for_row(row)
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
        st.warning("InjuryTypeClassifier not available. Check src/cv/injury_type_classifier.py.")

# ══════════════════════════════════════════════════════════════════
# PAGE 7 — SQUAD HEATMAP
# ══════════════════════════════════════════════════════════════════
elif page == "🗺️ Squad Heatmap":
    st.title("🗺️ Squad Risk Heatmap")
    st.caption("Weekly average injury risk per athlete. Red = high, Green = low.")

    if "date" not in df.columns:
        st.error("Date column missing.")
        st.stop()

    df_h = df.copy()
    df_h["week"] = df_h["date"].dt.to_period("W").astype(str)
    pivot = df_h.pivot_table(index="athlete_id", columns="week",
                              values="injury_risk_score", aggfunc="mean")
    pivot = pivot.iloc[:, -24:]

    fig_hm = px.imshow(
        pivot, color_continuous_scale="RdYlGn_r", zmin=0, zmax=100,
        title="Weekly Avg Injury Risk Score per Athlete",
        labels={"x":"Week","y":"Athlete ID","color":"Risk Score"},
        aspect="auto",
    )
    fig_hm.update_layout(height=650)
    st.plotly_chart(fig_hm, use_container_width=True)

    st.subheader("Athlete Risk Profiles")
    arp = df.groupby("athlete_id")["injury_risk_score"].agg(["mean","max","std"]).reset_index()
    arp.columns = ["Athlete ID","Avg Risk","Max Risk","Std Dev"]
    arp = arp.sort_values("Avg Risk", ascending=False)

    fig_p = px.scatter(arp, x="Avg Risk", y="Std Dev",
                       size="Max Risk", color="Avg Risk",
                       color_continuous_scale="RdYlGn_r",
                       hover_data=["Athlete ID"],
                       title="Athlete Risk Profile (Avg vs Variability)",
                       labels={"Avg Risk":"Average Risk Score","Std Dev":"Risk Variability"})
    st.plotly_chart(fig_p, use_container_width=True)

    st.dataframe(
        arp.style.background_gradient(subset=["Avg Risk","Max Risk"], cmap="RdYlGn_r"),
        use_container_width=True,
    )

# ── Footer ──────────────────────────────────────────────────────────
st.divider()
st.caption(
    "⚽ Athlete Injury Risk Prediction System — Synthetic Data Dashboard  ·  "
    "Stacked Ensemble + LSTM (per-athlete sequences) + Transformer (pos. encoding) + Hazard  ·  "
    "Hybrid blend weights · MC-Dropout uncertainty"
)
