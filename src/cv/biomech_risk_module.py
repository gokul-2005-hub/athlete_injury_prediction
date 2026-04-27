# src/cv/biomech_risk_module.py
#
# v2 — biomechanical risk channel for the hybrid predictor.
#
# CHANGES SINCE v1
# ────────────────
#
#   PHASE A — confidence formula (issue #7):
#     Confidence is now (frame_factor × detection_rate × mean_visibility),
#     with a hard cap at 0.30 for clips with fewer than 60 effective frames.
#     Old: confidence = clip(n_frames / 180). A 6-second blurry video where
#     half the frames missed a pose used to score 1.0; now it scores ~0.3.
#
#   PHASE B — sport-aware thresholds (issue #10):
#     Each rule's "low / med / high" thresholds now come from a per-sport
#     table. Tennis players legitimately produce 28° knee std from lunging
#     across rallies; that's healthy tennis, not an ACL signal. Running
#     keeps tighter thresholds because steady-state running shouldn't have
#     wide knee variability.
#
#     Sports table (ordered by typical range):
#       SQUAT  < RUNNING < GENERIC < TENNIS < BADMINTON
#
#     Pass sport="tennis" to compute_biomech_risk() to apply tennis
#     thresholds. Default is "generic" (unchanged from v1).
#
#   PHASE C — per-phase rule routing (issue #4):
#     If the input dict carries a "phases" block (produced by v4 of
#     PoseEstimator), each rule scores against the clinically appropriate
#     phase rather than the global mean/std:
#       knee_variability → landing-phase std (the actual ACL signal)
#       torso_lean       → running-phase mean (hamstring context)
#       balance          → single-support frames (running + landing)
#     If no phase data is present, the rule falls back to global stats
#     with a 0.6× weight multiplier and adds a transparent note.
#
# WHY RULE-BASED, NOT TRAINED?
# We have no labelled real video data. The previous "trained" approach
# laundered fatigue values through fake CV columns (knee_velocity,
# joint_variability) that were just noisy functions of HRV/sleep. v4
# eliminates those columns from the ML pipeline and replaces them with
# this transparent, sports-medicine-based rule channel.

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


# ──────────────────────────────────────────────────────────────────────
# Rule weights — calibrated so a clean video → ~0–15 pts, severely
# dysfunctional video → ~80–95 pts. The composite is clipped to [0, 100].
# ──────────────────────────────────────────────────────────────────────

RULE_WEIGHTS = {
    "knee_asymmetry":       {"low": 8.0,  "med": 16.0, "high": 22.0},
    "knee_variability":     {"low": 8.0,  "med": 14.0, "high": 20.0},
    "knee_symmetry":        {"low": 6.0,  "med": 12.0, "high": 18.0},
    "balance_stability":    {"low": 6.0,  "med": 12.0, "high": 18.0},
    "torso_lean":           {"low": 5.0,  "med": 9.0,  "high": 14.0},
    "movement_fluidity":    {"low": 4.0,  "med": 8.0,  "high": 10.0},
}

# When a rule falls back to global stats (because no per-phase data is
# available for its target phase), its contribution is reduced by this
# factor and the rule's description gets an explicit note.
PHASE_FALLBACK_WEIGHT = 0.6

# Minimum frames in the target phase before per-phase scoring is trusted.
# Below this, we fall back to global stats with the reduced weight.
MIN_PHASE_FRAMES = 8


# ──────────────────────────────────────────────────────────────────────
# Phase B — Sport-aware threshold tables
# ──────────────────────────────────────────────────────────────────────
# Each sport has its own thresholds for the 6 rules. "above" means the
# value crossing the threshold = bad (asymmetry, std, lean); "below" means
# value falling under the threshold = bad (symmetry, balance, fluidity).
#
# Calibration philosophy:
#   - Healthy training motion in that sport sits BELOW the "low"
#     threshold (no rule fires)
#   - "Low" = mild concern, worth noting
#   - "Med" = noticeable pattern, monitor
#   - "High" = strong injury risk signal at clinical level
#
# Numbers come from (a) published sports-medicine ranges where available,
# (b) reasonable extrapolation for sports without published norms, scaled
# from the steady-state running baseline.

SPORT_THRESHOLDS: Dict[str, Dict[str, Dict[str, float]]] = {
    # ── GENERIC — safe default for sports not in the list ─────────
    "generic": {
        "knee_asymmetry":    {"low": 8.0,  "med": 15.0, "high": 25.0},
        "knee_variability":  {"low": 10.0, "med": 18.0, "high": 28.0},
        "knee_symmetry":     {"low": 0.80, "med": 0.70, "high": 0.55},   # below
        "balance_stability": {"low": 0.75, "med": 0.60, "high": 0.50},   # below
        "torso_lean":        {"low": 0.15, "med": 0.22, "high": 0.30},
        "movement_fluidity": {"low": 0.75, "med": 0.60, "high": 0.45},   # below
    },

    # ── TENNIS — wide knee range from lunging, asymmetric serve ───
    # Tennis groundstrokes legitimately produce 25-35° knee std and
    # 0.20-0.30 sagittal lean. Tighter thresholds would flag every
    # competitive player. Knee asymmetry tolerance also raised because
    # forehand/backhand mechanics are inherently asymmetric.
    "tennis": {
        "knee_asymmetry":    {"low": 12.0, "med": 22.0, "high": 32.0},
        "knee_variability":  {"low": 18.0, "med": 28.0, "high": 38.0},
        "knee_symmetry":     {"low": 0.72, "med": 0.60, "high": 0.45},
        "balance_stability": {"low": 0.65, "med": 0.50, "high": 0.40},
        "torso_lean":        {"low": 0.22, "med": 0.32, "high": 0.42},
        "movement_fluidity": {"low": 0.65, "med": 0.50, "high": 0.35},
    },

    # ── BADMINTON — even wider because of explosive overhead smashes,
    # rapid lunges, and frequent jumps. Knee variability and torso lean
    # are particularly high in elite badminton without it being injurious.
    "badminton": {
        "knee_asymmetry":    {"low": 14.0, "med": 24.0, "high": 35.0},
        "knee_variability":  {"low": 20.0, "med": 30.0, "high": 42.0},
        "knee_symmetry":     {"low": 0.70, "med": 0.58, "high": 0.42},
        "balance_stability": {"low": 0.62, "med": 0.48, "high": 0.38},
        "torso_lean":        {"low": 0.24, "med": 0.34, "high": 0.45},
        "movement_fluidity": {"low": 0.62, "med": 0.48, "high": 0.32},
    },

    # ── RUNNING — steady-state, narrowest acceptable knee variability ─
    # Running gait should be consistent; high variability is a real
    # neuromuscular signal. Forward lean is normally 0.10-0.18; above
    # that suggests fatigue-driven form breakdown.
    "running": {
        "knee_asymmetry":    {"low": 5.0,  "med": 10.0, "high": 18.0},
        "knee_variability":  {"low": 6.0,  "med": 12.0, "high": 18.0},
        "knee_symmetry":     {"low": 0.85, "med": 0.78, "high": 0.65},
        "balance_stability": {"low": 0.78, "med": 0.65, "high": 0.55},
        "torso_lean":        {"low": 0.10, "med": 0.15, "high": 0.22},
        "movement_fluidity": {"low": 0.80, "med": 0.65, "high": 0.50},
    },

    # ── SQUAT / STRENGTH TRAINING — controlled motion, very tight ─
    # Strength training should be deliberate and bilaterally symmetric.
    # Any noticeable asymmetry or variability under load is a real
    # form-breakdown signal.
    "squat_strength": {
        "knee_asymmetry":    {"low": 4.0,  "med": 8.0,  "high": 14.0},
        "knee_variability":  {"low": 4.0,  "med": 8.0,  "high": 14.0},
        "knee_symmetry":     {"low": 0.88, "med": 0.80, "high": 0.70},
        "balance_stability": {"low": 0.82, "med": 0.70, "high": 0.60},
        "torso_lean":        {"low": 0.12, "med": 0.18, "high": 0.25},
        "movement_fluidity": {"low": 0.82, "med": 0.70, "high": 0.55},
    },
}

# Map dashboard-visible sport labels to the threshold-table keys.
SPORT_LABEL_TO_KEY = {
    "Tennis":                 "tennis",
    "Badminton":              "badminton",
    "Running":                "running",
    "Squat / Strength":       "squat_strength",
    "Generic":                "generic",
}


# ──────────────────────────────────────────────────────────────────────
# Dataclasses
# ──────────────────────────────────────────────────────────────────────

@dataclass
class TriggeredRule:
    """One rule that contributed points to the biomechanical risk score."""
    name:        str
    severity:    str       # "Low" | "Medium" | "High"
    points:      float
    value:       float
    threshold:   float
    description: str
    phase:       str = "global"   # which phase this was scored on
    fallback:    bool = False     # True ⇒ scored on global with reduced weight


@dataclass
class BiomechRiskResult:
    """Container for biomechanical risk output."""
    biomech_risk:    float
    confidence:      float
    triggered_rules: List[TriggeredRule] = field(default_factory=list)
    components:      Dict[str, float]    = field(default_factory=dict)
    classifier_max:  Optional[float]     = None
    n_frames:        int                 = 0
    sport:           str                 = "generic"
    phase_summary:   Dict[str, int]      = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _safe_float(d: Dict, key: str, default: float = 0.0) -> float:
    """Tolerant getter that handles None / NaN / strings gracefully."""
    try:
        v = d.get(key, default)
        if v is None:
            return default
        v = float(v)
        if not np.isfinite(v):
            return default
        return v
    except (TypeError, ValueError):
        return default


def _rule_threshold_severity(value: float, thresholds: Dict[str, float],
                             direction: str = "above") -> str:
    """
    Return severity bucket for a value relative to three thresholds.

    direction='above': value > threshold = bad (asymmetry, std, lean)
    direction='below': value < threshold = bad (symmetry, balance, fluidity)
    """
    low, med, high = thresholds["low"], thresholds["med"], thresholds["high"]
    if direction == "above":
        if value >= high:  return "high"
        if value >= med:   return "med"
        if value >= low:   return "low"
        return "none"
    else:
        if value <= high:  return "high"
        if value <= med:   return "med"
        if value <= low:   return "low"
        return "none"


def _phase_value(video_features: Dict, phase: str, key: str) -> Optional[float]:
    """
    Read a value from the per-phase block if it has enough frames.

    Phase C entry point — accepts both the nested `phases` dict produced
    by PoseEstimator v4 AND the legacy flat `{phase}_{key}` keys, so this
    works whether or not PoseEstimator was upgraded.
    """
    nested = video_features.get("phases") or {}
    pdata  = nested.get(phase) if isinstance(nested, dict) else None
    if pdata:
        if pdata.get("frame_count", 0) >= MIN_PHASE_FRAMES:
            v = pdata.get(key)
            if v is not None and np.isfinite(v):
                return float(v)
    # Fallback to legacy flat key
    flat = video_features.get(f"{phase}_{key}")
    if flat is not None:
        try:
            v = float(flat)
            if np.isfinite(v):
                # Only use if we know there are enough frames
                fc = video_features.get(f"{phase}_frame_count", 0)
                if fc >= MIN_PHASE_FRAMES:
                    return v
        except (TypeError, ValueError):
            pass
    return None


def _resolve_sport(sport: Optional[str]) -> str:
    """Map an arbitrary sport string (label or key) to a thresholds key."""
    if not sport:
        return "generic"
    s = str(sport).strip()
    if s in SPORT_THRESHOLDS:
        return s
    if s in SPORT_LABEL_TO_KEY:
        return SPORT_LABEL_TO_KEY[s]
    s_lower = s.lower()
    for key in SPORT_THRESHOLDS:
        if key in s_lower or s_lower in key:
            return key
    return "generic"


# ──────────────────────────────────────────────────────────────────────
# Main scoring function
# ──────────────────────────────────────────────────────────────────────

def compute_biomech_risk(
    video_features:     Dict,
    classifier_results: Optional[List] = None,
    classifier_blend:   float = 0.30,
    sport:              Optional[str] = None,
) -> BiomechRiskResult:
    """
    Compute a 0-100 biomechanical injury risk score from MediaPipe pose stats.

    Args:
        video_features: dict produced by PoseEstimator.process_video().
            v4 output includes a "phases" sub-dict for per-phase routing
            and "detection_rate" / "mean_visibility" for confidence.
        classifier_results: optional list of InjuryTypeRisk from
            InjuryTypeClassifier — its max risk_score is mixed in at
            weight classifier_blend.
        classifier_blend: 0-1, how much weight to give classifier_max.
        sport: which sport to use thresholds for. Accepts "tennis",
            "badminton", "running", "squat_strength", "generic", or the
            dashboard label form ("Tennis", etc.). Defaults to "generic".

    Returns:
        BiomechRiskResult.
    """
    sport_key   = _resolve_sport(sport)
    thresholds  = SPORT_THRESHOLDS[sport_key]

    # ── No video → zero contribution, zero confidence ────────────────
    if not video_features:
        return BiomechRiskResult(
            biomech_risk=0.0, confidence=0.0,
            triggered_rules=[], components={},
            classifier_max=None, n_frames=0, sport=sport_key,
        )

    triggered: List[TriggeredRule] = []
    components: Dict[str, float]   = {}

    # ── Phase availability checks ─────────────────────────────────
    landing_avail  = _phase_value(video_features, "landing",  "frame_count") is None and \
                     (video_features.get("phases", {}).get("landing", {}).get("frame_count", 0) >= MIN_PHASE_FRAMES)
    # Re-check more directly:
    nested = video_features.get("phases") or {}
    landing_frames  = (nested.get("landing", {}) or {}).get("frame_count",
                       video_features.get("landing_frame_count", 0))
    running_frames  = (nested.get("running", {}) or {}).get("frame_count",
                       video_features.get("running_frame_count", 0))

    has_landing = landing_frames >= MIN_PHASE_FRAMES
    has_running = running_frames >= MIN_PHASE_FRAMES

    phase_summary = {
        "landing":   int(landing_frames),
        "running":   int(running_frames),
        "squatting": int((nested.get("squatting", {}) or {}).get("frame_count",
                          video_features.get("squatting_frame_count", 0))),
        "jumping":   int((nested.get("jumping", {}) or {}).get("frame_count",
                          video_features.get("jumping_frame_count", 0))),
        "standing":  int((nested.get("standing", {}) or {}).get("frame_count",
                          video_features.get("standing_frame_count", 0))),
    }

    # Helper to get a rule's value with phase routing + fallback
    def _routed_value(target_phase: str, key: str,
                      global_fallback_key: str,
                      default: float):
        v_phase = _phase_value(video_features, target_phase, key)
        if v_phase is not None:
            return v_phase, target_phase, False
        v_global = _safe_float(video_features, global_fallback_key, default)
        return v_global, "global", True

    # ── Helpers to add a triggered rule ────────────────────────────
    def _add_rule(rule_key: str, name: str, severity: str, value: float,
                  thr_dict: Dict[str, float], phase: str, fallback: bool,
                  description_base: str):
        weight_mult = PHASE_FALLBACK_WEIGHT if fallback else 1.0
        pts = RULE_WEIGHTS[rule_key][severity] * weight_mult
        thr = thr_dict[severity]
        sev_label = "High" if severity == "high" else \
                    "Medium" if severity == "med" else "Low"
        descr = description_base
        if fallback:
            descr += " [Scored on global stats — insufficient phase frames.]"
        else:
            descr += f" [Scored on {phase}-phase frames]"
        triggered.append(TriggeredRule(
            name=name, severity=sev_label, points=pts, value=value,
            threshold=thr, description=descr, phase=phase, fallback=fallback,
        ))
        components[rule_key] = pts

    # ── Rule 1: Knee asymmetry (mean L/R diff) ─────────────────────
    # Routed through landing phase (ACL signal) when available.
    knee_l_mean, _, _      = _routed_value("landing", "knee_angle_left_mean",
                                           "knee_angle_left_mean",  160.0)
    knee_r_mean, p1, fb1   = _routed_value("landing", "knee_angle_right_mean",
                                           "knee_angle_right_mean", 160.0)
    asym = abs(knee_l_mean - knee_r_mean)
    sev  = _rule_threshold_severity(asym, thresholds["knee_asymmetry"], "above")
    if sev != "none":
        _add_rule(
            "knee_asymmetry",
            "Knee asymmetry",
            sev, asym, thresholds["knee_asymmetry"],
            phase=p1, fallback=fb1,
            description_base=(
                f"L/R knee angle differs by {asym:.1f}° "
                f"(threshold {thresholds['knee_asymmetry'][sev]:.0f}° for {sport_key}). "
                "Bilateral asymmetry under load is associated with non-contact "
                "ACL injury risk."
            ),
        )

    # ── Rule 2: Knee variability (std) ─────────────────────────────
    # PRIMARY ACL SIGNAL — routed through landing phase when present.
    knee_l_std, _, _   = _routed_value("landing", "knee_angle_left_std",
                                       "knee_angle_left_std", 5.0)
    knee_r_std, p2, fb2 = _routed_value("landing", "knee_angle_right_std",
                                        "knee_angle_right_std", 5.0)
    knee_var = max(knee_l_std, knee_r_std)
    sev = _rule_threshold_severity(knee_var, thresholds["knee_variability"], "above")
    if sev != "none":
        _add_rule(
            "knee_variability",
            "Knee instability",
            sev, knee_var, thresholds["knee_variability"],
            phase=p2, fallback=fb2,
            description_base=(
                f"Knee angle std = {knee_var:.1f}° "
                f"(threshold {thresholds['knee_variability'][sev]:.0f}° for {sport_key}). "
                "Erratic knee motion suggests reduced neuromuscular control."
            ),
        )

    # ── Rule 3: Knee symmetry (low = bad) ─────────────────────────
    # NOTE: this is the renamed posture_symmetry. Read knee_symmetry_mean
    # first, fall back to posture_symmetry_mean for old CSVs.
    knee_sym = _safe_float(video_features, "knee_symmetry_mean",
                _safe_float(video_features, "posture_symmetry_mean", 0.85))
    sev = _rule_threshold_severity(knee_sym, thresholds["knee_symmetry"], "below")
    if sev != "none":
        # Symmetry is global by nature (not phase-routed)
        _add_rule(
            "knee_symmetry",
            "Knee L/R symmetry",
            sev, knee_sym, thresholds["knee_symmetry"],
            phase="global", fallback=False,
            description_base=(
                f"Knee L/R symmetry = {knee_sym:.2f} "
                f"(target ≥ {thresholds['knee_symmetry'][sev]:.2f} for {sport_key}). "
                "Persistent left-right knee imbalance compensates load to one limb."
            ),
        )

    # ── Rule 4: Balance / hip drop ─────────────────────────────────
    # Routed through running OR landing (single-support frames). If
    # neither is available, fall back to global with reduced weight.
    bal_running = _phase_value(video_features, "running",
                                "balance_stability_mean")
    bal_landing = _phase_value(video_features, "landing",
                                "balance_stability_mean")
    if bal_running is not None or bal_landing is not None:
        # Average available single-support phases
        vals = [v for v in (bal_running, bal_landing) if v is not None]
        balance = float(np.mean(vals))
        bal_phase = "running+landing" if len(vals) > 1 else \
                    ("running" if bal_running is not None else "landing")
        bal_fb = False
    else:
        balance = _safe_float(video_features, "balance_stability_mean", 0.85)
        bal_phase, bal_fb = "global", True

    sev = _rule_threshold_severity(balance, thresholds["balance_stability"], "below")
    if sev != "none":
        _add_rule(
            "balance_stability",
            "Balance / hip stability",
            sev, balance, thresholds["balance_stability"],
            phase=bal_phase, fallback=bal_fb,
            description_base=(
                f"Balance stability = {balance:.2f} "
                f"(target ≥ {thresholds['balance_stability'][sev]:.2f} for {sport_key}). "
                "Hip drop pattern raises lateral knee + lumbar load."
            ),
        )

    # ── Rule 5: Torso lean ─────────────────────────────────────────
    # Hamstring strain context — running phase preferred.
    lean, p5, fb5 = _routed_value("running", "torso_lean_mean",
                                  "torso_lean_mean", 0.10)
    sev = _rule_threshold_severity(lean, thresholds["torso_lean"], "above")
    if sev != "none":
        _add_rule(
            "torso_lean",
            "Forward torso lean",
            sev, lean, thresholds["torso_lean"],
            phase=p5, fallback=fb5,
            description_base=(
                f"Torso lean = {lean:.2f} "
                f"(threshold {thresholds['torso_lean'][sev]:.2f} for {sport_key}). "
                "Excessive trunk lean during locomotion overloads hamstrings."
            ),
        )

    # ── Rule 6: Movement fluidity ──────────────────────────────────
    # Running-phase preferred (steady-state motion most reflective of
    # neuromuscular control).
    flu, p6, fb6 = _routed_value("running", "movement_fluidity_mean",
                                 "movement_fluidity_mean", 0.85)
    sev = _rule_threshold_severity(flu, thresholds["movement_fluidity"], "below")
    if sev != "none":
        _add_rule(
            "movement_fluidity",
            "Movement fluidity",
            sev, flu, thresholds["movement_fluidity"],
            phase=p6, fallback=fb6,
            description_base=(
                f"Movement fluidity = {flu:.2f} "
                f"(target ≥ {thresholds['movement_fluidity'][sev]:.2f} for {sport_key}). "
                "Jerky/hesitant motion suggests neuromuscular fatigue."
            ),
        )

    # ── Compose rule-based score ─────────────────────────────────
    rule_score = float(np.clip(sum(components.values()), 0.0, 100.0))

    # ── Optionally blend with InjuryTypeClassifier ────────────────
    classifier_max = None
    if classifier_results:
        try:
            classifier_max = max(
                float(getattr(r, "risk_score", 0)) for r in classifier_results
            )
        except (ValueError, TypeError):
            classifier_max = None

    if classifier_max is not None and classifier_blend > 0:
        b = float(np.clip(classifier_blend, 0.0, 1.0))
        biomech_risk = (1 - b) * rule_score + b * classifier_max
    else:
        biomech_risk = rule_score
    biomech_risk = float(np.clip(biomech_risk, 0.0, 100.0))

    # ── Confidence (Phase A #7) ───────────────────────────────────
    n_frames        = int(_safe_float(video_features, "frames_processed",
                          _safe_float(video_features, "n_frames_analysed", 0)))
    detection_rate  = _safe_float(video_features, "detection_rate", 1.0)
    mean_visibility = _safe_float(video_features, "mean_visibility", 1.0)

    frame_factor = float(np.clip(n_frames / 180.0, 0.0, 1.0))
    det_factor   = max(0.10, detection_rate)        # floor at 10%
    vis_factor   = max(0.10, mean_visibility)
    confidence   = float(np.clip(frame_factor * det_factor * vis_factor, 0.0, 1.0))

    # Phase A #8 — short clip cap. Below 60 effective frames
    # (~4 seconds at skip=2 / 30fps), confidence is capped at 0.30.
    if n_frames < 60:
        confidence = float(min(confidence, 0.30))

    return BiomechRiskResult(
        biomech_risk=biomech_risk,
        confidence=confidence,
        triggered_rules=sorted(triggered, key=lambda t: t.points, reverse=True),
        components=components,
        classifier_max=classifier_max,
        n_frames=n_frames,
        sport=sport_key,
        phase_summary=phase_summary,
    )


# ──────────────────────────────────────────────────────────────────────
# Convenience helpers
# ──────────────────────────────────────────────────────────────────────

def biomech_risk_level(score: float) -> str:
    if score >= 65: return "High"
    if score >= 35: return "Medium"
    return "Low"


def summarize_rules(result: BiomechRiskResult, max_rules: int = 6) -> str:
    if not result.triggered_rules:
        return "No biomechanical rules triggered."
    parts = [f"{r.name}: +{r.points:.0f}pts ({r.severity})"
             for r in result.triggered_rules[:max_rules]]
    return " · ".join(parts)
