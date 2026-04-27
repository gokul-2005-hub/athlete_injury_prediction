import numpy as np
from enum import Enum


class MovementPhase(str, Enum):
    """
    Discrete movement phases detected from pose landmarks.
    Each phase has different biomechanical risk profiles.
    """
    STANDING   = "standing"    # upright, low velocity
    RUNNING    = "running"     # alternating leg drive, forward lean
    SQUATTING  = "squatting"   # deep knee flexion
    JUMPING    = "jumping"     # upward vertical velocity, extension
    LANDING    = "landing"     # ⚠️ HIGH RISK — rapid deceleration
    UNKNOWN    = "unknown"     # insufficient data


class MovementPhaseDetector:
    """
    Rule-based movement phase classifier using MediaPipe pose landmarks.

    Why rule-based instead of ML?
    - No training data needed
    - Interpretable (coaches can understand the rules)
    - Fast enough for real-time video processing
    - Accurate enough for biomechanical phase segmentation

    Detection logic per phase
    ─────────────────────────
    LANDING   : knee angle dropping rapidly (> 15°/frame) AND knees bent < 150°
    JUMPING   : hip_y velocity strongly upward AND knees near extension (> 160°)
    SQUATTING : both knees < 130° AND relatively still (low velocity)
    RUNNING   : moderate knee flexion (130–165°) AND alternating leg asymmetry
    STANDING  : both knees > 165° AND low overall body velocity
    """

    # Thresholds (tunable)
    KNEE_EXTENDED      = 165.0   # degrees — standing / jumping
    KNEE_FLEXED_DEEP   = 130.0   # degrees — squat
    KNEE_FLEXED_LAND   = 150.0   # degrees — landing threshold
    KNEE_DROP_RATE     = 15.0    # degrees/frame — rapid flexion = landing
    HIP_VELOCITY_UP    = -0.015  # normalised y (MediaPipe y increases downward)
    VELOCITY_STILL     = 0.008   # below this = not running

    def __init__(self):
        self._prev_pts   = None
        self._prev_knees = None   # (knee_l, knee_r) from previous frame

    def reset(self):
        """Call between videos to clear state."""
        self._prev_pts   = None
        self._prev_knees = None

    @staticmethod
    def _angle(a, b, c) -> float:
        a = np.array(a); b = np.array(b); c = np.array(c)
        ba = a - b; bc = c - b
        denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-9
        return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))))

    def detect(self, lm) -> MovementPhase:
        """
        Classify a single frame's movement phase.

        Parameters
        ----------
        lm : list of MediaPipe landmark objects

        Returns
        -------
        MovementPhase enum value
        """
        try:
            pts = [(p.x, p.y) for p in lm]

            # Landmark indices
            L_HIP, R_HIP     = 23, 24
            L_KNEE, R_KNEE   = 25, 26
            L_ANKLE, R_ANKLE = 27, 28

            # Current knee angles
            knee_l = self._angle(pts[L_HIP],  pts[L_KNEE],  pts[L_ANKLE])
            knee_r = self._angle(pts[R_HIP],  pts[R_KNEE],  pts[R_ANKLE])
            knee_avg = (knee_l + knee_r) / 2.0

            # Hip centre vertical position
            hip_y = (pts[L_HIP][1] + pts[R_HIP][1]) / 2.0

            # Overall body velocity (mean landmark displacement)
            if self._prev_pts is not None:
                displacements = [
                    np.linalg.norm(np.array(pts[i]) - np.array(self._prev_pts[i]))
                    for i in range(min(len(pts), len(self._prev_pts)))
                ]
                body_velocity = float(np.mean(displacements))
                hip_velocity  = pts[L_HIP][1] - self._prev_pts[L_HIP][1]
            else:
                body_velocity = 0.0
                hip_velocity  = 0.0

            # Knee angle drop rate (positive = flexing)
            if self._prev_knees is not None:
                prev_knee_avg = (self._prev_knees[0] + self._prev_knees[1]) / 2.0
                knee_drop_rate = prev_knee_avg - knee_avg
            else:
                knee_drop_rate = 0.0

            # Left-right knee asymmetry (running signature)
            knee_asymmetry = abs(knee_l - knee_r)

            # ── Update state ──────────────────────────────────────
            self._prev_pts   = pts
            self._prev_knees = (knee_l, knee_r)

            # ── Classification rules (priority order) ─────────────

            # 1. LANDING — highest priority (most injury-critical)
            #    Rapid knee flexion while knees are already bent
            if knee_drop_rate > self.KNEE_DROP_RATE and knee_avg < self.KNEE_FLEXED_LAND:
                return MovementPhase.LANDING

            # 2. JUMPING — upward hip movement + extended knees
            if hip_velocity < self.HIP_VELOCITY_UP and knee_avg > self.KNEE_EXTENDED:
                return MovementPhase.JUMPING

            # 3. SQUATTING — deep knee flexion, body relatively still
            if knee_avg < self.KNEE_FLEXED_DEEP and body_velocity < self.VELOCITY_STILL * 3:
                return MovementPhase.SQUATTING

            # 4. RUNNING — moderate flexion + alternating asymmetry + moving
            if (self.KNEE_FLEXED_DEEP < knee_avg < self.KNEE_EXTENDED
                    and knee_asymmetry > 10.0
                    and body_velocity > self.VELOCITY_STILL):
                return MovementPhase.RUNNING

            # 5. STANDING — extended knees, low velocity
            if knee_avg > self.KNEE_EXTENDED and body_velocity < self.VELOCITY_STILL:
                return MovementPhase.STANDING

            # Default: RUNNING if moving, STANDING if still
            return MovementPhase.RUNNING if body_velocity > self.VELOCITY_STILL else MovementPhase.STANDING

        except Exception:
            return MovementPhase.UNKNOWN


class PhaseAggregator:
    """
    Collects per-frame (phase, features) pairs and computes
    phase-specific aggregate statistics.

    Output features (per phase that has enough frames):
        {phase}_knee_angle_mean
        {phase}_knee_angle_std
        {phase}_body_symmetry_mean
        {phase}_count          ← how many frames in this phase
        {phase}_risk_index     ← composite risk score for this phase
    """

    # Minimum frames to consider a phase valid
    MIN_FRAMES = 3

    # Phase-specific risk weights
    # Landing biomechanics matter most for injury
    PHASE_RISK_WEIGHTS = {
        MovementPhase.LANDING:   2.0,   # highest risk
        MovementPhase.JUMPING:   1.5,
        MovementPhase.RUNNING:   1.2,
        MovementPhase.SQUATTING: 1.0,
        MovementPhase.STANDING:  0.5,
        MovementPhase.UNKNOWN:   0.0,
    }

    def __init__(self):
        # phase → list of feature dicts
        self._phase_data: dict[MovementPhase, list[dict]] = {
            p: [] for p in MovementPhase
        }

    def add_frame(self, phase: MovementPhase, features: dict):
        """Record a frame's features under its detected phase."""
        self._phase_data[phase].append(features)

    def get_phase_counts(self) -> dict[str, int]:
        """Return how many frames were detected per phase."""
        return {p.value: len(frames) for p, frames in self._phase_data.items()}

    def get_dominant_phase(self) -> str:
        """Return the phase with the most frames."""
        counts = {p: len(f) for p, f in self._phase_data.items()
                  if p != MovementPhase.UNKNOWN}
        if not counts:
            return MovementPhase.UNKNOWN.value
        return max(counts, key=counts.get).value

    def aggregate(self) -> dict:
        """
        Compute phase-specific aggregate features.
        Only includes phases with >= MIN_FRAMES frames.

        Returns flat dict ready to merge into the ML feature row.
        """
        out = {}
        out["dominant_phase"] = self.get_dominant_phase()

        phase_counts = self.get_phase_counts()
        out["phase_counts"] = phase_counts

        # Per-phase biomechanical statistics
        for phase, frames in self._phase_data.items():
            if len(frames) < self.MIN_FRAMES:
                continue

            pname = phase.value
            out[f"{pname}_frame_count"] = len(frames)

            # Aggregate each numeric feature
            keys = frames[0].keys()
            for k in keys:
                vals = [f[k] for f in frames if k in f and isinstance(f[k], (int, float))]
                if vals:
                    out[f"{pname}_{k}_mean"] = float(np.mean(vals))
                    out[f"{pname}_{k}_std"]  = float(np.std(vals))

            # Phase risk index:
            # Low symmetry + high knee variability during landing = high risk
            sym_vals = [f.get("body_symmetry_score", 1.0) for f in frames]
            kn_vals  = [f.get("knee_angle_left", 160) for f in frames]
            weight   = self.PHASE_RISK_WEIGHTS.get(phase, 1.0)

            risk_index = weight * (
                (1.0 - float(np.mean(sym_vals))) * 50.0
                + float(np.std(kn_vals)) * 0.5
            )
            out[f"{pname}_risk_index"] = float(np.clip(risk_index, 0, 100))

        return out
