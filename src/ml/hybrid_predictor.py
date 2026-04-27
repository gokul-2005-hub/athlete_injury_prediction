# src/ml/hybrid_predictor.py
#
# v4 — Two-channel hybrid (Option C refactor)
#
# ── ARCHITECTURE ─────────────────────────────────────────────────────────────
# Channel 1 (workload-ML, ~50 features):
#     workload_risk = 0.35*Stack + 0.30*LSTM + 0.20*Transformer + 0.15*Hazard
#
# Channel 2 (biomechanical-rules, computed from MediaPipe pose stats):
#     biomech_risk  = compute_biomech_risk(video_features) ∈ [0, 100]
#
# Final blend:
#     α_eff      = clamp(BIOMECH_VIDEO_WEIGHT * confidence, α_min, α_max)
#     final_risk = (1 - α_eff) * workload_risk + α_eff * biomech_risk
#
# When predict() is called WITHOUT video_features, behaviour is identical to
# the v3 single-channel hybrid (workload only). When called WITH a video
# features dict (from PoseEstimator.process_video() or the rule module),
# the biomech channel contributes up to 40% of the final score depending on
# detection confidence.
#
# ── BACKWARD COMPATIBILITY ──────────────────────────────────────────────────
# All existing callers of predict(df, mc_samples=N) keep working unchanged —
# they just won't get a biomech contribution. New callers pass
# video_features=... and (optionally) classifier_results=... .

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Optional, Dict, List

import tensorflow as tf
from tensorflow.keras.models import load_model

from .layers import SinusoidalPositionalEncoding

# ── Two-channel config (with safe import fallback) ─────────────────────────
try:
    from config import (
        BIOMECH_VIDEO_WEIGHT, BIOMECH_ALPHA_MIN, BIOMECH_ALPHA_MAX,
        BIOMECH_CLASSIFIER_BLEND,
    )
except Exception:
    # Fallbacks if config.py is unavailable (e.g. unit tests)
    BIOMECH_VIDEO_WEIGHT     = 0.40
    BIOMECH_ALPHA_MIN        = 0.00
    BIOMECH_ALPHA_MAX        = 0.40
    BIOMECH_CLASSIFIER_BLEND = 0.30

# Biomech risk module (rule-based)
try:
    from src.cv.biomech_risk_module import compute_biomech_risk, BiomechRiskResult
except Exception:
    try:
        from cv.biomech_risk_module import compute_biomech_risk, BiomechRiskResult
    except Exception:
        compute_biomech_risk = None
        BiomechRiskResult    = None

# ── Paths ─────────────────────────────────────────────────────────────────────

STACK_MODEL_PATH       = Path("models/ml/stacked_injury_model.pkl")
LSTM_MODEL_PATH        = Path("models/ml/lstm_sequence_model.keras")
LSTM_META_PATH         = Path("models/ml/lstm_injury_model_meta.pkl")
TRANSFORMER_MODEL_PATH = Path("models/ml/transformer_sequence_model.keras")
TRANSFORMER_META_PATH  = Path("models/ml/transformer_meta.pkl")
HAZARD_MODEL_PATH      = Path("models/ml/hazard_model.pkl")

_FALLBACK_WEIGHTS = {
    "stack": 0.35, "lstm": 0.30, "transformer": 0.20, "hazard": 0.15,
}


def set_biomech_weight(weight: float) -> None:
    """
    Override the global BIOMECH_VIDEO_WEIGHT at runtime.

    The dashboard's sidebar slider calls this so the next predict() call
    uses the new α target without restarting the app. Bounded to [0.0, 1.0].
    Note: this only affects this module's module-level constant — config.py's
    value is not mutated.
    """
    global BIOMECH_VIDEO_WEIGHT
    BIOMECH_VIDEO_WEIGHT = float(np.clip(weight, 0.0, 1.0))


def get_biomech_weight() -> float:
    """Read the current effective BIOMECH_VIDEO_WEIGHT."""
    return float(BIOMECH_VIDEO_WEIGHT)


class HybridPredictor:
    """
    Two-channel hybrid predictor for injury risk.

    Channel 1 — Workload ML
        Stacked Ensemble (XGB+LGB+Cat+ET → Ridge) + LSTM + Transformer + Hazard
        Trained on synthetic workload data; fixed blend weights.

    Channel 2 — Biomechanical Rules (NEW in v4)
        Rule-based score derived from MediaPipe pose statistics. Activated
        when video_features is passed to predict(). Confidence-scaled so a
        short or noisy clip contributes less than a clean 6+ second analysis.

    Blend
        final_risk = (1 - α_eff) * workload_risk + α_eff * biomech_risk
        α_eff      = clamp(BIOMECH_VIDEO_WEIGHT * confidence, α_min, α_max)

    MC-Dropout (mc_samples > 1) still applies only to LSTM and Transformer.
    """

    def __init__(self):
        print("Loading stacked ensemble...")
        self.stack_model = joblib.load(STACK_MODEL_PATH)

        # ── Workload-channel weights (fixed) ──────────────────────
        w = _FALLBACK_WEIGHTS
        total = sum(w.values())
        self._weights = np.array([
            w["stack"] / total, w["lstm"] / total,
            w["transformer"] / total, w["hazard"] / total,
        ])
        print(
            f"Workload-channel weights — "
            f"stack={self._weights[0]:.3f}, lstm={self._weights[1]:.3f}, "
            f"transformer={self._weights[2]:.3f}, hazard={self._weights[3]:.3f}"
        )
        print(
            f"Biomech-channel max alpha = {BIOMECH_VIDEO_WEIGHT:.2f}  "
            f"(scaled by confidence; range {BIOMECH_ALPHA_MIN:.2f}–{BIOMECH_ALPHA_MAX:.2f})"
        )

        n_meta = self.stack_model.get("n_meta_inputs", 3)
        print(f"Ensemble meta expects {n_meta} base-model inputs.")

        print("Loading LSTM sequence model...")
        self.lstm_model   = load_model(LSTM_MODEL_PATH)
        self.lstm_meta    = joblib.load(LSTM_META_PATH)
        self.lstm_seq_len = int(self.lstm_meta.get("sequence_length", 30))

        print("Loading Transformer sequence model...")
        self.transformer_model = load_model(
            TRANSFORMER_MODEL_PATH,
            custom_objects={"SinusoidalPositionalEncoding": SinusoidalPositionalEncoding},
        )
        self.transformer_meta    = joblib.load(TRANSFORMER_META_PATH)
        self.transformer_seq_len = int(self.transformer_meta.get("sequence_length", 30))

        print("Loading hazard (survival) model...")
        _raw_hazard = joblib.load(HAZARD_MODEL_PATH)
        if isinstance(_raw_hazard, dict) and "model" in _raw_hazard:
            self.hazard_model        = _raw_hazard["model"]
            # v6: percentile-rank normalisation reference (sorted ascending
            # array of training raw hazards). Falls back to the older
            # train_max_hazard value if loaded from a v4/v5 bundle, with a
            # warning telling the user to retrain.
            self._hazard_train_dist  = _raw_hazard.get("train_hazard_dist")
            self._hazard_max_ref     = _raw_hazard.get("train_max_hazard")
        else:
            self.hazard_model       = _raw_hazard
            self._hazard_train_dist = None
            self._hazard_max_ref    = None

        if self._hazard_train_dist is None and self._hazard_max_ref is None:
            print(
                "WARNING: hazard model bundle has no normalisation reference. "
                "Falling back to per-call max — predictions will vary by batch. "
                "Re-run train_hazard_model.py to fix."
            )
        elif self._hazard_train_dist is None:
            print(
                "WARNING: hazard bundle uses legacy max-normalisation, which "
                "made most sessions display 0. Re-run train_hazard_model.py "
                "to upgrade to percentile-rank scoring."
            )

        if compute_biomech_risk is None:
            print(
                "WARNING: biomech_risk_module unavailable — video features will "
                "not influence predictions even if passed to predict()."
            )

        print("All models loaded.")

    # ── Helper ────────────────────────────────────────────────────

    def _align(self, df: pd.DataFrame, features: list) -> pd.DataFrame:
        return df.reindex(columns=features, fill_value=0)

    # ── Stack prediction ──────────────────────────────────────────

    def predict_stack(self, df: pd.DataFrame) -> np.ndarray:
        features = self.stack_model["features"]
        X = self._align(df, features)
        n_meta = self.stack_model.get("n_meta_inputs", 3)

        if n_meta == 4 and "et" in self.stack_model:
            p = np.column_stack([
                self.stack_model["xgb"].predict(X),
                self.stack_model["lgb"].predict(X),
                self.stack_model["cat"].predict(X),
                self.stack_model["et"].predict(X),
            ])
        else:
            p = np.column_stack([
                self.stack_model["xgb"].predict(X),
                self.stack_model["lgb"].predict(X),
                self.stack_model["cat"].predict(X),
            ])
        return self.stack_model["meta"].predict(p)

    # ── LSTM prediction ───────────────────────────────────────────

    def predict_lstm(
        self,
        df: pd.DataFrame,
        mc_samples: int = 1,
        require_full_sequence: bool = True,
    ) -> Optional[tuple[float, float]]:
        features = self.lstm_meta["features"]
        scaler   = self.lstm_meta["scaler"]
        seq_len  = self.lstm_seq_len

        X = self._align(df, features)
        if len(X) < seq_len:
            if require_full_sequence:
                return None
            pad_len  = seq_len - len(X)
            pad_vals = np.tile(X.iloc[0].values, (pad_len, 1))
            X_padded = np.vstack([pad_vals, X.values])
            X_scaled = scaler.transform(pd.DataFrame(X_padded, columns=features))
        else:
            X_scaled = scaler.transform(X)

        seq = np.expand_dims(X_scaled[-seq_len:], axis=0)

        if mc_samples <= 1:
            pred = self.lstm_model(seq, training=False).numpy().flatten()
            return float(pred[-1]), 0.0

        preds = np.array([
            self.lstm_model(seq, training=True).numpy().flatten()[-1]
            for _ in range(mc_samples)
        ])
        return float(preds.mean()), float(preds.std())

    # ── Transformer prediction ────────────────────────────────────

    def predict_transformer(
        self,
        df: pd.DataFrame,
        mc_samples: int = 1,
        require_full_sequence: bool = True,
    ) -> Optional[tuple[float, float]]:
        features = self.transformer_meta["features"]
        scaler   = self.transformer_meta["scaler"]
        seq_len  = self.transformer_seq_len

        X = self._align(df, features)
        if len(X) < seq_len:
            if require_full_sequence:
                return None
            pad_len  = seq_len - len(X)
            pad_vals = np.tile(X.iloc[0].values, (pad_len, 1))
            X_padded = np.vstack([pad_vals, X.values])
            X_scaled = scaler.transform(pd.DataFrame(X_padded, columns=features))
        else:
            X_scaled = scaler.transform(X)

        seq = np.expand_dims(X_scaled[-seq_len:], axis=0)

        if mc_samples <= 1:
            pred = self.transformer_model(seq, training=False).numpy().flatten()
            return float(pred[-1]), 0.0

        preds = np.array([
            self.transformer_model(seq, training=True).numpy().flatten()[-1]
            for _ in range(mc_samples)
        ])
        return float(preds.mean()), float(preds.std())

    # ── Hazard prediction ─────────────────────────────────────────

    def predict_hazard(self, df: pd.DataFrame) -> np.ndarray:
        """
        Score raw cumulative-hazard outputs into a 0-100 scale.

        v6: percentile-rank normalisation. For each raw score, we look up
        its rank against the sorted training distribution and return that
        percentile. So a session whose raw hazard equals the training-set
        median displays as 50.0, the training-set 90th percentile displays
        as 90.0, etc. Properly distributed across 0-100.

        Falls back to (a) divide-by-train-max if a v4/v5 bundle is loaded,
        (b) per-call max as last resort. Both fallbacks display a warning
        at load time.
        """
        if hasattr(self.hazard_model, "feature_names_in_"):
            features = list(self.hazard_model.feature_names_in_)
        else:
            features = [c for c in self.stack_model["features"] if c != "injury_risk_score"]

        X   = self._align(df, features)
        raw = self.hazard_model.predict(X)

        # Primary path: percentile-rank against training distribution
        if self._hazard_train_dist is not None and len(self._hazard_train_dist) > 0:
            ranks = np.searchsorted(self._hazard_train_dist, raw, side="right")
            pct   = (ranks / len(self._hazard_train_dist)) * 100.0
            return np.clip(pct, 0, 100).astype(float)

        # Fallback A: legacy v4/v5 max normalisation
        if self._hazard_max_ref is not None and self._hazard_max_ref > 0:
            return np.clip(raw / self._hazard_max_ref * 100, 0, 100)

        # Fallback B: per-call max (oldest behaviour — batch-dependent)
        max_val = max(raw.max(), 1e-6)
        return np.clip(raw / max_val * 100, 0, 100)

    # ── Feature importance ────────────────────────────────────────

    def get_feature_importance(self) -> dict:
        try:
            fi = self.stack_model["xgb"].feature_importances_
            return dict(zip(self.stack_model["features"], fi))
        except Exception:
            return {}

    # ── Effective alpha helper ────────────────────────────────────
    @staticmethod
    def _effective_alpha(confidence: float) -> float:
        """Confidence-scaled biomech weight, clamped to configured range."""
        a = BIOMECH_VIDEO_WEIGHT * float(np.clip(confidence, 0.0, 1.0))
        return float(np.clip(a, BIOMECH_ALPHA_MIN, BIOMECH_ALPHA_MAX))

    # ── Main predict ──────────────────────────────────────────────

    def predict(
        self,
        df:                 pd.DataFrame,
        mc_samples:         int = 1,
        video_features:     Optional[Dict] = None,
        classifier_results: Optional[List] = None,
        biomech_sport:      Optional[str]  = None,
    ) -> dict:
        """
        Run all 4 workload components + optional biomech channel.

        Args:
            df:                 Engineered session-history DataFrame.
            mc_samples:         MC-Dropout passes (LSTM/Transformer only).
            video_features:     Optional dict from PoseEstimator.process_video()
                                — when supplied, the biomechanical-rules channel
                                contributes up to BIOMECH_VIDEO_WEIGHT of the
                                final score (scaled by detection confidence).
            classifier_results: Optional list of InjuryTypeRisk from
                                InjuryTypeClassifier — its max risk_score is
                                blended into biomech_risk.
            biomech_sport:      Sport label/key for sport-specific rule
                                thresholds ("Tennis", "Badminton", "Running",
                                "Squat / Strength", "Generic"). Default
                                "generic" — preserves v1 behaviour for callers
                                that don't pass sport.

        Returns dict with workload component scores, biomech_risk, blend
        weights actually applied, and final_risk_score.
        """
        # ─── Channel 1 — workload models ───────────────────────────────
        stack_arr  = self.predict_stack(df)
        lstm_res   = self.predict_lstm(df, mc_samples=mc_samples)
        transf_res = self.predict_transformer(df, mc_samples=mc_samples)
        hazard_arr = self.predict_hazard(df)

        stack_val  = float(np.clip(stack_arr[-1],  0, 100))
        hazard_val = float(np.clip(hazard_arr[-1], 0, 100))

        lstm_val = lstm_std = None
        if lstm_res is not None:
            lstm_val = float(np.clip(lstm_res[0], 0, 100))
            lstm_std = float(lstm_res[1])

        transf_val = transf_std = None
        if transf_res is not None:
            transf_val = float(np.clip(transf_res[0], 0, 100))
            transf_std = float(transf_res[1])

        # Weighted workload blend with graceful degradation
        w           = self._weights.copy()
        vals        = [stack_val, lstm_val, transf_val, hazard_val]
        active_mask = np.array([v is not None for v in vals], dtype=float)
        w_active    = w * active_mask
        if w_active.sum() < 1e-9:
            w_active = active_mask / max(active_mask.sum(), 1)
        else:
            w_active /= w_active.sum()

        active_vals    = np.array([v if v is not None else 0.0 for v in vals])
        workload_risk  = float(np.clip((w_active * active_vals).sum(), 0, 100))

        stds        = [s for s in [lstm_std, transf_std] if s is not None]
        uncertainty = float(np.mean(stds)) if stds else 0.0

        # ─── Channel 2 — biomechanical rules ──────────────────────────
        biomech_risk     = 0.0
        biomech_conf     = 0.0
        biomech_rules    = []
        biomech_components = {}
        biomech_classifier_max = None
        biomech_n_frames = 0
        biomech_sport_used = "generic"
        biomech_phase_summary = {}

        if video_features and compute_biomech_risk is not None:
            try:
                bres = compute_biomech_risk(
                    video_features,
                    classifier_results=classifier_results,
                    classifier_blend=BIOMECH_CLASSIFIER_BLEND,
                    sport=biomech_sport,
                )
                biomech_risk           = bres.biomech_risk
                biomech_conf           = bres.confidence
                biomech_rules          = [
                    {
                        "name":        r.name,
                        "severity":    r.severity,
                        "points":      r.points,
                        "value":       r.value,
                        "threshold":   r.threshold,
                        "description": r.description,
                        "phase":       getattr(r, "phase", "global"),
                        "fallback":    getattr(r, "fallback", False),
                    } for r in bres.triggered_rules
                ]
                biomech_components     = bres.components
                biomech_classifier_max = bres.classifier_max
                biomech_n_frames       = bres.n_frames
                biomech_sport_used     = getattr(bres, "sport", "generic")
                biomech_phase_summary  = getattr(bres, "phase_summary", {})
            except Exception as e:
                # Fail-safe: if biomech scoring blows up, fall back to
                # workload-only rather than crashing the whole prediction.
                print(f"[HybridPredictor] biomech scoring failed: {e}")

        # ─── Final two-channel blend ──────────────────────────────────
        alpha_eff = self._effective_alpha(biomech_conf)
        final_val = (1 - alpha_eff) * workload_risk + alpha_eff * biomech_risk
        final_val = float(np.clip(final_val, 0, 100))

        return {
            # Workload channel components
            "ensemble":          stack_val,
            "lstm":              lstm_val,
            "transformer":       transf_val,
            "hazard":            hazard_val,
            "lstm_std":          lstm_std  or 0.0,
            "transformer_std":   transf_std or 0.0,
            "uncertainty_band":  uncertainty,
            "workload_risk":     workload_risk,

            # Biomech channel
            "biomech_risk":      biomech_risk,
            "biomech_confidence": biomech_conf,
            "biomech_rules":     biomech_rules,
            "biomech_components": biomech_components,
            "biomech_classifier_max": biomech_classifier_max,
            "biomech_n_frames":  biomech_n_frames,
            "biomech_sport":     biomech_sport_used,
            "biomech_phase_summary": biomech_phase_summary,

            # Final hybrid output
            "alpha_video":       alpha_eff,
            "final_risk_score":  final_val,

            # Effective weights actually applied this call. Sum = 1.
            "effective_weights": {
                "stack":       float(w_active[0] * (1 - alpha_eff)),
                "lstm":        float(w_active[1] * (1 - alpha_eff)),
                "transformer": float(w_active[2] * (1 - alpha_eff)),
                "hazard":      float(w_active[3] * (1 - alpha_eff)),
                "biomech":     float(alpha_eff),
            },
        }
