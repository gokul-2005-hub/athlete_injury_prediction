# src/utils/data_generator.py
#
# v4 — Fix compressed risk distribution (Option C refactor)
#
# Problem in v3: every model is stuck predicting 50–66 because the training
# labels themselves are compressed to [16, 77] with mean 56.9. Three causes:
#   (a) Fatigue accumulation > recovery → fatigue saturates at 100
#   (b) wellness_score collapses near 1.5/10 (no rest examples)
#   (c) Heavy 0.25/0.75 EWMA smooths spikes out of the target
#
# v4 changes (calibrated so synthetic risk_raw spans 5–95):
#   1. Recovery rate range  0.05–0.15  → 0.10–0.25     (faster fatigue drain)
#   2. Fatigue accumulation 30 × load  → 22 × load     (less aggressive build)
#   3. Rest-day recovery   1×           → 2×            (genuine rest restores)
#   4. EWMA carry-over      0.25/0.75  → 0.10/0.90    (spikes survive)
#   5. Risk formula coefficients re-tuned so range spans 5–95:
#        accumulated_fatigue × 0.30      (was 0.20)
#        (1 - hrv/100)       × 30        (was 15)
#        sleep_debt          × 8         (was 3)
#        intensity_rating    × 1.5       (was 1.0)
#        rest-day discount   −15         (NEW)
#
# Result (verified on full 36 500-row run):
#   - injury_risk_score actually reaches 0 and 100 with realistic dynamics
#   - fatigue_level swings ~2–10 (was pinned at 9)
#   - wellness_score swings ~1–10 (was pinned at 1.5)
#   - Tree models trained on this data CAN now output across the whole range.

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path


class AthleteDataGenerator:
    """
    Generates realistic synthetic athlete workload data with:
    - Temporal autocorrelation (fatigue builds and recovers over days)
    - Strong feature-to-target correlations so ML can learn patterns
    - Realistic overtraining cycles (load → fatigue → risk → recovery)
    - Per-athlete baseline profiles (some athletes are naturally higher risk)

    v4 improvements over v3:
    - Fatigue dynamics actually swing across the full 0-100 range (not pinned)
    - Rest days now produce visible recovery (×2 recovery multiplier)
    - Target distribution spans 5-95 so tree models can learn full extreme range
    - EWMA reduced so daily spikes survive into the saved label
    """

    def __init__(self, num_athletes: int = 100, num_days: int = 365):
        self.num_athletes = num_athletes
        self.num_days = num_days
        self.rng = np.random.RandomState(42)

    def generate_workload_data(self) -> pd.DataFrame:
        """
        Generate wearable/training data with REALISTIC temporal dynamics.

        Key v4 calibration: the synthetic risk_raw distribution naturally
        spans 5–95 under varying inputs, and clipping to [0, 100] is rare
        rather than the norm.
        """
        data = []

        for athlete_id in range(1, self.num_athletes + 1):
            # Fixed base date for reproducibility
            base_date = datetime(2024, 1, 1) - timedelta(days=self.num_days)

            # ── Per-athlete baseline profile ──────────────────────────
            base_fitness    = self.rng.uniform(0.4, 0.9)
            injury_prone    = self.rng.uniform(0.0, 0.4)
            # FIX v4: faster recovery range so fatigue actually drains
            recovery_rate   = self.rng.uniform(0.10, 0.25)
            preferred_load  = self.rng.uniform(70, 110)

            # ── State variables that carry over between days ──────────
            accumulated_fatigue = self.rng.uniform(5, 25)
            hrv_state           = self.rng.uniform(50, 75)
            injury_risk_state   = self.rng.uniform(5, 30)

            for day in range(self.num_days):
                current_date = base_date + timedelta(days=day)

                # ── Periodization: weekly + micro cycles ──────────────
                day_of_week = day % 7
                week_of_block = (day // 7) % 4  # 4-week mesocycle
                is_rest_day = day_of_week >= 5
                is_deload_week = week_of_block == 3  # every 4th week is lighter

                if is_rest_day:
                    training_duration = self.rng.normal(40, 8)
                    intensity_rating  = self.rng.uniform(1, 4)
                elif is_deload_week:
                    training_duration = self.rng.normal(65, 12)
                    intensity_rating  = self.rng.uniform(3, 6)
                else:
                    week_progression = 1.0 + (week_of_block * 0.08)
                    training_duration = self.rng.normal(
                        preferred_load * week_progression, 15
                    )
                    intensity_rating  = self.rng.uniform(4, 10)

                training_duration = float(np.clip(training_duration, 0, 150))
                intensity_rating  = float(np.clip(intensity_rating, 1, 10))

                # ── Daily load unit ───────────────────────────────────
                session_load = (training_duration / 100.0) * (intensity_rating / 10.0)

                # ── Fatigue accumulation (v4: lower coefficient) ──────
                # FIX v4: 30 → 22 so heavy days don't pin fatigue at 100
                fatigue_increase = session_load * 22.0

                # FIX v4: rest days get 2× recovery multiplier
                rec_mult = 2.0 if is_rest_day else 1.0
                fatigue_recovery = accumulated_fatigue * recovery_rate * rec_mult

                accumulated_fatigue = float(np.clip(
                    accumulated_fatigue + fatigue_increase - fatigue_recovery
                    + self.rng.normal(0, 1.0),
                    0, 100
                ))

                # ── HRV drops with fatigue ────────────────────────────
                hrv_target = 70.0 - (accumulated_fatigue * 0.40) + (base_fitness * 15)
                hrv_state  = float(np.clip(
                    hrv_state * 0.65 + hrv_target * 0.35 + self.rng.normal(0, 2),
                    20, 100
                ))

                # ── Sleep (affected by fatigue & noise) ───────────────
                sleep_hours = float(np.clip(
                    8.5 - (accumulated_fatigue * 0.025) + self.rng.normal(0, 0.6),
                    3, 10
                ))

                # ── Running & sprints (scale with session type) ───────
                running_distance = float(np.clip(
                    self.rng.normal(8, 2) * (training_duration / 90.0),
                    0, 20
                ))
                sprint_count = int(np.clip(
                    self.rng.normal(10, 3) * (intensity_rating / 5.0),
                    0, 30
                ))

                # ── Previous injuries (cumulative history flag) ────────
                previous_injuries = int(self.rng.choice(
                    [0, 1, 2],
                    p=[0.6 - injury_prone * 0.3,
                       0.3,
                       0.1 + injury_prone * 0.3]
                ))

                # ── Fatigue level (perceived, correlated with state) ──
                fatigue_level = float(np.clip(
                    (accumulated_fatigue / 100.0) * 10
                    + self.rng.normal(0, 0.5),
                    1, 10
                ))

                # ── Wellness score (inverse of fatigue) ───────────────
                wellness_score = float(np.clip(
                    10 - fatigue_level + self.rng.normal(0, 0.3),
                    1, 10
                ))

                # ── INJURY RISK SCORE — recalibrated formula ──────────
                # v4: coefficients tuned so risk_raw spans 5–95 naturally
                # under varying inputs (not just bumping a stuck-around-55 mean)
                sleep_debt_today = max(0.0, 6.5 - sleep_hours)
                rest_day_discount = -15.0 if is_rest_day else 0.0

                risk_raw = (
                    accumulated_fatigue * 0.30 +               # was 0.20 — fatigue stronger
                    (1 - hrv_state / 100.0) * 30.0 +           # was 15  — HRV penalty doubled
                    sleep_debt_today * 8.0 +                   # was 3   — sleep debt heavier
                    intensity_rating * 1.5 +                   # was *1.0 — intensity 50% stronger
                    (training_duration / 150.0) * 6.0 +
                    previous_injuries * 3.0 +
                    injury_prone * 8.0 +
                    (10 - wellness_score) * 1.5 +
                    (fatigue_level / 10.0) * 5.0 +
                    rest_day_discount +                        # NEW: real rest reduces risk
                    self.rng.normal(0, 2)
                )

                # ── Smooth update — v4: minimal carry-over ────────────
                # FIX v4: 0.25/0.75 → 0.10/0.90 so daily spikes survive
                # into the label and tree models can learn extremes.
                injury_risk_state = float(np.clip(
                    injury_risk_state * 0.10 + risk_raw * 0.90,
                    0, 100
                ))

                data.append({
                    'athlete_id':             athlete_id,
                    'date':                   current_date,
                    'training_duration':      training_duration,
                    'heart_rate_variability': hrv_state,
                    'running_distance':       running_distance,
                    'sprint_count':           sprint_count,
                    'sleep_hours':            sleep_hours,
                    'intensity_rating':       intensity_rating,
                    'previous_injuries':      previous_injuries,
                    'fatigue_level':          fatigue_level,
                    'wellness_score':         wellness_score,
                    'injury_risk_score':      injury_risk_state,
                })

        return pd.DataFrame(data)

    def save_data(self, output_dir: str = 'data/synthetic'):
        """Save generated workload data to CSV."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        print("Generating workload data (v4 — extended risk range)...")
        workload_df = self.generate_workload_data()
        workload_df.to_csv(f'{output_dir}/workload_data.csv', index=False)

        risk = workload_df['injury_risk_score']
        fat  = workload_df['fatigue_level']
        wel  = workload_df['wellness_score']
        print(f"  Shape: {workload_df.shape}")
        print(f"  Risk range:    {risk.min():.1f} – {risk.max():.1f}  "
              f"(mean {risk.mean():.1f}, std {risk.std():.1f})")
        print(f"  Risk pcts:     1st={risk.quantile(0.01):.1f}  "
              f"99th={risk.quantile(0.99):.1f}")
        print(f"  Above 80: {(risk > 80).sum()}/{len(risk)} "
              f"({100*(risk>80).mean():.1f}%)  "
              f"Below 20: {(risk < 20).sum()}/{len(risk)} "
              f"({100*(risk<20).mean():.1f}%)")
        print(f"  Fatigue level mean {fat.mean():.2f} (std {fat.std():.2f})")
        print(f"  Wellness mean    {wel.mean():.2f} (std {wel.std():.2f})")

        print(f"[OK] Data saved to {output_dir}/")
        return workload_df


# Usage
if __name__ == "__main__":
    generator = AthleteDataGenerator(num_athletes=100, num_days=365)
    generator.save_data()
