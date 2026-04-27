# src/cv/pose_estimator.py
#
# v4 — biomechanical extraction with:
#
#   PHASE A FIXES (accuracy / honesty)
#   ──────────────────────────────────
#   #1  Camera-distance normalisation: balance & shoulder symmetry are
#       divided by torso length (the way torso_lean already was). Same
#       person at any distance now produces the same number.
#   #2  Visibility weighting: every per-frame feature is collected with
#       its mean MediaPipe-visibility weight. Frames with occluded knees
#       contribute less to mean/std than fully-visible frames.
#   #3  Jitter smoothing: a 5-frame moving average is applied to raw
#       landmark positions BEFORE computing per-frame angles. Removes
#       MediaPipe's ~2-3° single-frame jitter without flattening real
#       motion.
#   #4  Honest naming: posture_symmetry → knee_symmetry (it always was
#       knee L/R symmetry; the misleading name caused a downstream
#       double-count). The legacy posture_symmetry_mean/std keys are
#       kept as aliases for one release so old history CSVs still load.
#   #5  Visibility / detection rate / mean visibility tracking — exposed
#       so biomech_risk_module can compute a real confidence figure.
#
#   PHASE C (per-phase aggregation)
#   ────────────────────────────────
#   The MovementPhaseDetector + PhaseAggregator already classify each
#   frame as landing / running / jumping / squatting / standing.
#   v4 ROUTES per-phase aggregates through to the output dict so
#   biomech_risk_module can score each rule against the clinically
#   correct phase:
#     * knee variability → landing-phase std (the actual ACL signal)
#     * torso lean       → running-phase mean (hamstring context)
#     * balance stability→ single-support frames (running + landing)

import cv2
import mediapipe as mp
import numpy as np
from collections import deque

from .movement_phase_detector import MovementPhaseDetector, PhaseAggregator


class PoseEstimator:
    """Full-body pose estimation with movement-phase segmentation."""

    # ── MediaPipe landmark indices ─────────────────────────────────
    L_HIP,      R_HIP      = 23, 24
    L_KNEE,     R_KNEE     = 25, 26
    L_ANKLE,    R_ANKLE    = 27, 28
    L_SHOULDER, R_SHOULDER = 11, 12
    L_ELBOW,    R_ELBOW    = 13, 14
    L_WRIST,    R_WRIST    = 15, 16

    # Landmarks whose visibility we care about for weighting (lower body
    # joints dominate injury-relevant features). Face / hand landmarks are
    # ignored on purpose — face wobble has nothing to do with movement
    # quality.
    KEY_LANDMARKS = [
        L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE,
        L_SHOULDER, R_SHOULDER,
    ]

    # 5-frame moving-average window on raw landmark x/y. Removes MediaPipe
    # single-frame jitter (~2–3°) without flattening real motion.
    SMOOTH_WINDOW = 5

    def __init__(self, confidence_threshold: float = 0.5):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=confidence_threshold,
            min_tracking_confidence=confidence_threshold,
        )

    # ── Geometry helpers ───────────────────────────────────────────

    @staticmethod
    def _angle(a, b, c) -> float:
        a = np.array(a); b = np.array(b); c = np.array(c)
        ba = a - b; bc = c - b
        denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-9
        return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / denom, -1, 1))))

    @staticmethod
    def _dist(a, b) -> float:
        return float(np.linalg.norm(np.array(a) - np.array(b)))

    @staticmethod
    def _smooth_landmarks(buffer: deque) -> list:
        """Phase A #3: 5-frame moving average over the raw landmark coords."""
        if len(buffer) == 1:
            return list(buffer[0])
        n_lm = len(buffer[0])
        out = []
        for i in range(n_lm):
            xs = [frame[i][0] for frame in buffer]
            ys = [frame[i][1] for frame in buffer]
            out.append((float(np.mean(xs)), float(np.mean(ys))))
        return out

    # ── Per-frame feature extraction ───────────────────────────────

    def _extract_features(self, smoothed_pts, lm) -> dict | None:
        """
        Compute all per-frame biomechanical features.

        Args:
            smoothed_pts: list of (x, y) tuples after the 5-frame avg.
            lm:           original MediaPipe landmark list (used only for
                          .visibility on the key landmarks).

        Returns:
            Feature dict + a "_visibility" weight key for downstream
            visibility-weighted aggregation.
        """
        try:
            pts = smoothed_pts

            # ── Lower body ─────────────────────────────────────────
            knee_l = self._angle(pts[self.L_HIP],  pts[self.L_KNEE],  pts[self.L_ANKLE])
            knee_r = self._angle(pts[self.R_HIP],  pts[self.R_KNEE],  pts[self.R_ANKLE])

            knee_max  = max(knee_l, knee_r, 1e-9)
            knee_sym  = float(np.clip(1.0 - abs(knee_l - knee_r) / knee_max, 0, 1))

            # ── Torso length normaliser (Phase A #1) ───────────────
            mid_hip_x = (pts[self.L_HIP][0] + pts[self.R_HIP][0]) / 2
            mid_hip_y = (pts[self.L_HIP][1] + pts[self.R_HIP][1]) / 2
            mid_sh_x  = (pts[self.L_SHOULDER][0] + pts[self.R_SHOULDER][0]) / 2
            mid_sh_y  = (pts[self.L_SHOULDER][1] + pts[self.R_SHOULDER][1]) / 2
            torso_len = max(
                self._dist((mid_hip_x, mid_hip_y), (mid_sh_x, mid_sh_y)),
                0.05,
            )

            # ── Camera-distance-normalised symmetries ──────────────
            hip_y_diff   = abs(pts[self.L_HIP][1] - pts[self.R_HIP][1])
            balance_stab = float(np.clip(1.0 - (hip_y_diff / torso_len), 0, 1))

            sh_y_diff    = abs(pts[self.L_SHOULDER][1] - pts[self.R_SHOULDER][1])
            # × 1.6 keeps the new shoulder_sym scale roughly comparable to
            # the legacy `× 8` so dashboards displaying historical numbers
            # don't shift wildly. Both are now distance-invariant.
            shoulder_sym = float(np.clip(
                1.0 - (sh_y_diff / torso_len) * 1.6, 0, 1
            ))

            # ── Upper body ─────────────────────────────────────────
            elbow_l = self._angle(pts[self.L_SHOULDER], pts[self.L_ELBOW], pts[self.L_WRIST])
            elbow_r = self._angle(pts[self.R_SHOULDER], pts[self.R_ELBOW], pts[self.R_WRIST])

            torso_lean = float(abs(mid_sh_x - mid_hip_x) / torso_len)

            # ── Full-body symmetry composite (Phase A #4) ─────────
            # Removed knee_sym from this composite — it was double-counted
            # with the knee_asymmetry rule. Body symmetry now blends only
            # shoulder + elbow symmetry. Knee symmetry stays as its own
            # dedicated channel (knee_symmetry).
            elbow_max = max(elbow_l, elbow_r, 1e-9)
            elbow_sym = 1.0 - abs(elbow_l - elbow_r) / elbow_max
            body_sym  = float(np.clip((shoulder_sym + elbow_sym) / 2.0, 0, 1))

            # ── Visibility weighting (Phase A #2) ──────────────────
            try:
                vis = float(np.mean([
                    lm[i].visibility for i in self.KEY_LANDMARKS
                ]))
            except Exception:
                vis = 1.0

            return {
                # Lower body
                "knee_angle_left":      knee_l,
                "knee_angle_right":     knee_r,
                "knee_symmetry":        knee_sym,
                "balance_stability":    balance_stab,
                # Upper body
                "shoulder_symmetry":    shoulder_sym,
                "elbow_angle_left":     elbow_l,
                "elbow_angle_right":    elbow_r,
                "torso_lean":           torso_lean,
                # Full body composite (no longer includes knee sym)
                "body_symmetry_score":  body_sym,
                # Frame-level visibility weight (popped later)
                "_visibility":          vis,
            }
        except Exception:
            return None

    # ── Video processing ───────────────────────────────────────────

    def process_video(self, video_path: str, skip_frames: int = 2) -> dict | None:
        """
        Process a full video. Returns a dict containing:

          1. Global aggregate features (visibility-weighted mean + std)
          2. Per-phase aggregate features (PhaseC)
          3. Phase distribution (% time in each phase)
          4. Confidence stats (n_frames, detection_rate, mean_visibility)
          5. Movement Quality Score (MQS, 0–100)

        Returns None on read failure or if no frames yielded a pose.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        phase_detector   = MovementPhaseDetector()
        phase_aggregator = PhaseAggregator()

        global_rows: list[dict]        = []
        global_visibility: list[float] = []
        smoothing_buffer: deque        = deque(maxlen=self.SMOOTH_WINDOW)
        prev_smoothed                  = None

        frame_idx       = 0
        total_frames    = 0   # frames where we attempted detection
        detected_frames = 0   # frames where MediaPipe returned landmarks

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % skip_frames == 0:
                total_frames += 1
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = self.pose.process(rgb)

                if res.pose_landmarks:
                    detected_frames += 1
                    lm  = res.pose_landmarks.landmark
                    pts = [(p.x, p.y) for p in lm]

                    smoothing_buffer.append(pts)
                    smoothed = self._smooth_landmarks(smoothing_buffer)

                    feats = self._extract_features(smoothed, lm)
                    if feats:
                        # Movement fluidity from SMOOTHED positions
                        if prev_smoothed is not None:
                            displ = [
                                self._dist(smoothed[i], prev_smoothed[i])
                                for i in self.KEY_LANDMARKS
                            ]
                            feats["movement_fluidity"] = float(
                                np.clip(1.0 - np.mean(displ) * 20.0, 0, 1)
                            )
                        else:
                            feats["movement_fluidity"] = 1.0
                        prev_smoothed = smoothed

                        vis = feats.pop("_visibility", 1.0)
                        global_visibility.append(vis)
                        global_rows.append(feats)

                        # Phase detection on the original landmarks (so
                        # detector velocity heuristics see real magnitudes)
                        phase = phase_detector.detect(lm)
                        phase_aggregator.add_frame(phase, feats)

            frame_idx += 1

        cap.release()

        if not global_rows:
            return None

        # ── 1. Global aggregates with visibility weighting ────────
        out: dict = {}
        feature_keys = [
            "knee_angle_left", "knee_angle_right", "knee_symmetry",
            "balance_stability", "shoulder_symmetry",
            "elbow_angle_left", "elbow_angle_right", "torso_lean",
            "body_symmetry_score", "movement_fluidity",
        ]
        weights = np.asarray(global_visibility, dtype=float)
        if weights.sum() < 1e-6:
            weights = np.ones_like(weights)

        for k in feature_keys:
            vals = np.asarray(
                [r[k] for r in global_rows if k in r], dtype=float
            )
            if len(vals) == 0:
                continue
            w = weights[: len(vals)]
            mean = float(np.average(vals, weights=w))
            var  = float(np.average((vals - mean) ** 2, weights=w))
            std  = float(np.sqrt(max(var, 0.0)))
            out[f"{k}_mean"] = mean
            out[f"{k}_std"]  = std

        # Backward-compatible aliases for posture_symmetry
        if "knee_symmetry_mean" in out:
            out["posture_symmetry_mean"] = out["knee_symmetry_mean"]
            out["posture_symmetry_std"]  = out["knee_symmetry_std"]

        # ── 2. Per-phase aggregates (Phase C) ─────────────────────
        phase_features = phase_aggregator.aggregate()
        out.update(phase_features)   # keep flat keys for back-compat

        nested_phases: dict[str, dict] = {}
        for k, v in phase_features.items():
            for phase_name in (
                "landing", "running", "jumping", "squatting", "standing"
            ):
                prefix = f"{phase_name}_"
                if k.startswith(prefix):
                    suffix = k[len(prefix):]
                    if (suffix.endswith("_mean") or suffix.endswith("_std")
                            or suffix == "frame_count" or suffix == "risk_index"):
                        nested_phases.setdefault(phase_name, {})[suffix] = v
                    break
        out["phases"]         = nested_phases
        out["dominant_phase"] = phase_features.get("dominant_phase", "unknown")
        out["phase_counts"]   = phase_features.get("phase_counts", {})

        # ── 3. Phase distribution (% time) ────────────────────────
        total = max(sum(phase_aggregator.get_phase_counts().values()), 1)
        for phase_name, count in phase_aggregator.get_phase_counts().items():
            out[f"phase_pct_{phase_name}"] = round(count / total, 3)

        # ── 4. Confidence stats (Phase A #5 — exposed) ────────────
        out["frames_processed"]  = len(global_rows)
        out["n_frames_analysed"] = len(global_rows)
        out["detection_rate"]    = float(detected_frames / max(total_frames, 1))
        out["mean_visibility"]   = float(np.mean(weights)) if len(weights) else 0.0

        # ── 5. MQS ────────────────────────────────────────────────
        out["movement_quality_score"] = self._compute_mqs(out)

        return out

    def get_feature_names(self) -> list[str]:
        base = [
            "knee_angle_left", "knee_angle_right", "knee_symmetry",
            "balance_stability", "shoulder_symmetry",
            "elbow_angle_left", "elbow_angle_right",
            "torso_lean", "body_symmetry_score", "movement_fluidity",
        ]
        return [f"{f}_{s}" for f in base for s in ("mean", "std")]

    @staticmethod
    def _compute_mqs(features: dict) -> float:
        """
        Movement Quality Score (0–100). Same composition as v3 but reads
        the renamed keys (knee_symmetry, with posture_symmetry alias for
        backcompat).
        """
        def safe_get(key, default=0.0):
            try:
                v = features.get(key, default)
                if v is None:
                    return default
                v = float(v)
                if not np.isfinite(v):
                    return default
                return v
            except (TypeError, ValueError):
                return default

        knee_l_std    = safe_get("knee_angle_left_std",  5)
        knee_r_std    = safe_get("knee_angle_right_std", 5)
        knee_stability = float(np.clip(100 - (knee_l_std + knee_r_std) * 1.8, 0, 100))

        knee_sym_score = safe_get("knee_symmetry_mean",
                                  safe_get("posture_symmetry_mean", 0.8)) * 100
        balance_score  = safe_get("balance_stability_mean", 0.8) * 100
        fluidity_score = safe_get("movement_fluidity_mean", 0.8) * 100
        shoulder_score = safe_get("shoulder_symmetry_mean", 0.8) * 100

        torso_lean  = safe_get("torso_lean_mean", 0.1)
        torso_score = float(np.clip(100 - torso_lean * 150, 0, 100))

        mqs = (
            knee_stability * 0.25 +
            knee_sym_score * 0.20 +
            balance_score  * 0.20 +
            fluidity_score * 0.20 +
            shoulder_score * 0.10 +
            torso_score    * 0.05
        )
        return round(float(np.clip(mqs, 0, 100)), 1)

    @staticmethod
    def mqs_grade(score: float) -> str:
        if score >= 90: return "Excellent"
        if score >= 70: return "Good"
        if score >= 50: return "Needs Improvement"
        return "Injury Risk Zone"
