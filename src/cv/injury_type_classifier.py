import numpy as np
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class InjuryTypeRisk:
    """
    Risk assessment for a specific injury type.

    name        : injury name
    risk_score  : 0–100
    risk_level  : Low / Medium / High
    key_signals : which features triggered this risk
    advice      : specific prevention recommendation
    """
    name:        str
    risk_score:  float
    risk_level:  str
    key_signals: List[str]
    advice:      str


class InjuryTypeClassifier:
    """
    Improvement 4 — Injury Type Risk Classification.

    Instead of a single injury_risk_score (0–100), this classifier
    breaks down risk into SPECIFIC injury types, each driven by
    different biomechanical and workload signals.

    Supported injury types
    ──────────────────────
    1. ACL (Anterior Cruciate Ligament)
       Signals: knee angle variability, landing mechanics, asymmetry

    2. Hamstring Strain
       Signals: high sprint load, knee extension variability, fatigue

    3. Ankle Instability
       Signals: balance instability, landing mechanics, running asymmetry

    4. Stress Fracture
       Signals: chronic overload (ACWR), training duration, low HRV

    5. Shoulder Overuse
       Signals: shoulder asymmetry, torso lean, arm swing variability

    Each injury type has its own rule-based scoring function
    combining wearable + video features.
    """

    def classify(
        self,
        workload_features: Dict,
        video_features:    Dict,
    ) -> List[InjuryTypeRisk]:
        """
        Compute risk scores for all supported injury types.

        Parameters
        ----------
        workload_features : dict from wearable data
            Keys: training_duration, heart_rate_variability,
                  sprint_count, sleep_hours, intensity_rating,
                  fatigue_level, wellness_score, previous_injuries
        video_features    : dict from PoseEstimator.process_video()
            Keys: knee_angle_left_std, posture_symmetry_mean,
                  landing_risk_index, body_symmetry_score_mean, etc.

        Returns
        -------
        List of InjuryTypeRisk sorted by risk_score descending.
        """
        results = [
            self._acl_risk(workload_features, video_features),
            self._hamstring_risk(workload_features, video_features),
            self._ankle_risk(workload_features, video_features),
            self._stress_fracture_risk(workload_features, video_features),
            self._shoulder_overuse_risk(workload_features, video_features),
        ]

        # Sort highest risk first
        results.sort(key=lambda x: x.risk_score, reverse=True)
        return results

    # ── Helper ─────────────────────────────────────────────────────

    @staticmethod
    def _level(score: float) -> str:
        if score >= 65:
            return "High"
        elif score >= 35:
            return "Medium"
        return "Low"

    @staticmethod
    def _get(d: Dict, key: str, default: float = 0.0) -> float:
        """Safe float getter."""
        try:
            return float(d.get(key, default) or default)
        except (TypeError, ValueError):
            return default

    # ── 1. ACL Risk ────────────────────────────────────────────────

    def _acl_risk(self, w: Dict, v: Dict) -> InjuryTypeRisk:
        """
        ACL injuries are strongly associated with:
        - High knee angle variability (especially during landing)
        - Low posture/body symmetry
        - High intensity with low HRV (neuromuscular fatigue)
        - Previous injuries
        """
        signals  = []
        score    = 0.0

        # Knee angle variability — primary ACL signal
        knee_std = max(
            self._get(v, "knee_angle_left_std"),
            self._get(v, "knee_angle_right_std"),
        )
        if knee_std > 25:
            score += 35
            signals.append(f"High knee angle variability ({knee_std:.1f}°)")
        elif knee_std > 15:
            score += 20
            signals.append(f"Moderate knee variability ({knee_std:.1f}°)")

        # Landing mechanics — phase-specific risk
        landing_risk = self._get(v, "landing_risk_index")
        if landing_risk > 40:
            score += 25
            signals.append(f"Poor landing mechanics (risk index {landing_risk:.1f})")
        elif landing_risk > 20:
            score += 12

        # Low body symmetry
        sym = self._get(v, "body_symmetry_score_mean", 1.0)
        if sym < 0.70:
            score += 20
            signals.append(f"Low body symmetry ({sym:.2f})")
        elif sym < 0.82:
            score += 10

        # Neuromuscular fatigue: high intensity + low HRV
        hrv       = self._get(w, "heart_rate_variability", 60)
        intensity = self._get(w, "intensity_rating", 5)
        if hrv < 45 and intensity > 7:
            score += 15
            signals.append(f"Neuromuscular fatigue (HRV {hrv:.0f}, intensity {intensity:.1f})")

        # Injury history
        if self._get(w, "previous_injuries") >= 1:
            score += 10
            signals.append("Previous injury history")

        score = float(np.clip(score, 0, 100))

        return InjuryTypeRisk(
            name="ACL (Knee Ligament)",
            risk_score=round(score, 1),
            risk_level=self._level(score),
            key_signals=signals or ["No significant ACL risk signals"],
            advice=(
                "Focus on landing mechanics drills, single-leg stability work, "
                "and neuromuscular control exercises."
                if score >= 35 else
                "Maintain current ACL prevention programme."
            ),
        )

    # ── 2. Hamstring Strain ────────────────────────────────────────

    def _hamstring_risk(self, w: Dict, v: Dict) -> InjuryTypeRisk:
        """
        Hamstring strains are associated with:
        - High sprint count (eccentric overload)
        - Knee extension variability during running phase
        - Accumulated fatigue + low sleep
        - High training duration
        """
        signals = []
        score   = 0.0

        # Sprint load — primary hamstring risk
        sprints = self._get(w, "sprint_count", 0)
        if sprints > 18:
            score += 30
            signals.append(f"Very high sprint count ({sprints:.0f})")
        elif sprints > 12:
            score += 18
            signals.append(f"High sprint count ({sprints:.0f})")

        # Running phase knee variability (eccentric loading signal)
        running_knee_std = self._get(v, "running_knee_angle_left_std", 0)
        if running_knee_std > 20:
            score += 25
            signals.append(f"High knee variability during running ({running_knee_std:.1f}°)")
        elif running_knee_std > 12:
            score += 12

        # Fatigue + sleep deficit
        fatigue = self._get(w, "fatigue_level", 5)
        sleep   = self._get(w, "sleep_hours", 8)
        if fatigue > 7 and sleep < 6.5:
            score += 20
            signals.append(f"High fatigue ({fatigue:.1f}/10) with sleep deficit ({sleep:.1f}h)")
        elif fatigue > 7:
            score += 10
            signals.append(f"High fatigue level ({fatigue:.1f}/10)")

        # Training duration overload
        duration = self._get(w, "training_duration", 90)
        if duration > 120:
            score += 15
            signals.append(f"Extended training duration ({duration:.0f} min)")

        # Movement fluidity decline (neuromuscular fatigue)
        fluidity = self._get(v, "movement_fluidity_mean", 1.0)
        if fluidity < 0.60:
            score += 15
            signals.append(f"Reduced movement fluidity ({fluidity:.2f})")

        score = float(np.clip(score, 0, 100))

        return InjuryTypeRisk(
            name="Hamstring Strain",
            risk_score=round(score, 1),
            risk_level=self._level(score),
            key_signals=signals or ["No significant hamstring risk signals"],
            advice=(
                "Reduce sprint volume, add Nordic hamstring curls, "
                "prioritise sleep and recovery sessions."
                if score >= 35 else
                "Sprint load is manageable. Maintain warm-up routine."
            ),
        )

    # ── 3. Ankle Instability ───────────────────────────────────────

    def _ankle_risk(self, w: Dict, v: Dict) -> InjuryTypeRisk:
        """
        Ankle instability is associated with:
        - Low balance stability
        - Poor landing mechanics
        - Running phase asymmetry
        - Previous ankle injuries
        """
        signals = []
        score   = 0.0

        # Balance stability — primary ankle signal
        balance = self._get(v, "balance_stability_mean", 1.0)
        if balance < 0.65:
            score += 35
            signals.append(f"Poor balance stability ({balance:.2f})")
        elif balance < 0.80:
            score += 18
            signals.append(f"Reduced balance stability ({balance:.2f})")

        # Landing risk
        landing_risk = self._get(v, "landing_risk_index", 0)
        if landing_risk > 35:
            score += 25
            signals.append(f"High-risk landing mechanics (index {landing_risk:.1f})")

        # Posture symmetry
        posture = self._get(v, "posture_symmetry_mean", 1.0)
        if posture < 0.72:
            score += 15
            signals.append(f"Low posture symmetry ({posture:.2f})")

        # Wellness (low wellness = compromised proprioception)
        wellness = self._get(w, "wellness_score", 8)
        if wellness < 4:
            score += 15
            signals.append(f"Low wellness score ({wellness:.1f}/10)")

        # Injury history
        if self._get(w, "previous_injuries") >= 1:
            score += 10
            signals.append("Prior lower limb injury")

        score = float(np.clip(score, 0, 100))

        return InjuryTypeRisk(
            name="Ankle Instability",
            risk_score=round(score, 1),
            risk_level=self._level(score),
            key_signals=signals or ["No significant ankle risk signals"],
            advice=(
                "Add single-leg balance training, proprioception drills, "
                "and consider ankle taping for high-intensity sessions."
                if score >= 35 else
                "Ankle stability is acceptable. Continue balance exercises."
            ),
        )

    # ── 4. Stress Fracture ─────────────────────────────────────────

    def _stress_fracture_risk(self, w: Dict, v: Dict) -> InjuryTypeRisk:
        """
        Stress fractures are associated with:
        - High chronic load (ACWR > 1.5)
        - Consecutive high-duration sessions
        - Low HRV (inadequate recovery)
        - Low wellness score
        """
        signals = []
        score   = 0.0

        # ACWR — primary stress fracture signal
        acwr = self._get(w, "acwr", 1.0)
        if acwr > 1.5:
            score += 40
            signals.append(f"Dangerous workload spike (ACWR {acwr:.2f} > 1.5)")
        elif acwr > 1.3:
            score += 22
            signals.append(f"Elevated ACWR ({acwr:.2f})")
        elif acwr > 1.1:
            score += 10

        # Chronic overload: high duration + high running distance
        duration = self._get(w, "training_duration", 90)
        distance = self._get(w, "running_distance", 8)
        if duration > 110 and distance > 12:
            score += 20
            signals.append(f"High training volume ({duration:.0f}min / {distance:.1f}km)")

        # Recovery deficit
        hrv  = self._get(w, "heart_rate_variability", 60)
        if hrv < 40:
            score += 20
            signals.append(f"Very low HRV ({hrv:.0f}) — inadequate recovery")
        elif hrv < 50:
            score += 10

        sleep = self._get(w, "sleep_hours", 8)
        if sleep < 5.5:
            score += 15
            signals.append(f"Significant sleep deficit ({sleep:.1f}h)")

        score = float(np.clip(score, 0, 100))

        return InjuryTypeRisk(
            name="Stress Fracture",
            risk_score=round(score, 1),
            risk_level=self._level(score),
            key_signals=signals or ["No significant stress fracture risk signals"],
            advice=(
                "Reduce training volume immediately. Prioritise sleep and nutrition. "
                "Consider bone density screening if ACWR remains elevated."
                if score >= 35 else
                "Workload is within safe range. Monitor ACWR weekly."
            ),
        )

    # ── 5. Shoulder Overuse ────────────────────────────────────────

    def _shoulder_overuse_risk(self, w: Dict, v: Dict) -> InjuryTypeRisk:
        """
        Shoulder overuse injuries are associated with:
        - Shoulder asymmetry (compensatory mechanics)
        - Torso lean (altered upper body kinematics)
        - Elbow angle variability (erratic arm swing)
        - High intensity with poor recovery
        """
        signals = []
        score   = 0.0

        # Shoulder asymmetry — primary signal
        sh_sym = self._get(v, "shoulder_symmetry_mean", 1.0)
        if sh_sym < 0.65:
            score += 35
            signals.append(f"Significant shoulder asymmetry ({sh_sym:.2f})")
        elif sh_sym < 0.78:
            score += 18
            signals.append(f"Moderate shoulder asymmetry ({sh_sym:.2f})")

        # Torso lean — altered shoulder loading
        torso = self._get(v, "torso_lean_mean", 0)
        if torso > 0.30:
            score += 20
            signals.append(f"Excessive torso lean ({torso:.2f})")
        elif torso > 0.18:
            score += 10

        # Elbow angle variability (arm swing fatigue)
        elbow_std = max(
            self._get(v, "elbow_angle_left_std", 0),
            self._get(v, "elbow_angle_right_std", 0),
        )
        if elbow_std > 22:
            score += 20
            signals.append(f"High arm swing variability ({elbow_std:.1f}°)")

        # Fatigue driving compensatory mechanics
        fatigue   = self._get(w, "fatigue_level", 5)
        intensity = self._get(w, "intensity_rating", 5)
        if fatigue > 7.5 and intensity > 6:
            score += 15
            signals.append(f"High intensity ({intensity:.1f}) under fatigue ({fatigue:.1f})")

        score = float(np.clip(score, 0, 100))

        return InjuryTypeRisk(
            name="Shoulder Overuse",
            risk_score=round(score, 1),
            risk_level=self._level(score),
            key_signals=signals or ["No significant shoulder risk signals"],
            advice=(
                "Check throwing/overhead mechanics. Add rotator cuff strengthening "
                "and reduce overhead load until symmetry improves."
                if score >= 35 else
                "Shoulder mechanics look acceptable. Maintain posture exercises."
            ),
        )

    # ── Summary helper ─────────────────────────────────────────────

    @staticmethod
    def get_top_risk(results: List[InjuryTypeRisk], n: int = 2) -> List[InjuryTypeRisk]:
        """Return top N injury risks."""
        return results[:n]

    @staticmethod
    def to_dict(results: List[InjuryTypeRisk]) -> List[dict]:
        """Serialise results to JSON-safe list of dicts."""
        return [
            {
                "injury_type": r.name,
                "risk_score":  r.risk_score,
                "risk_level":  r.risk_level,
                "key_signals": r.key_signals,
                "advice":      r.advice,
            }
            for r in results
        ]
