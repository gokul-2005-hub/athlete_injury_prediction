# Migration Guide — v4 (Option C, Two-Channel Hybrid)

This document walks through applying the v4 refactor on top of your existing
v3 project. Read it once before copying files in.

## What v4 changes

| Concern | v3 (broken) | v4 (this refactor) |
|---|---|---|
| Risk score range in synthetic data | 15.9 – 76.7 | 0 – 100 (mean ~44) |
| Fatigue level | Pinned at 9.0/10 | Varies 1–10 (mean ~4.4) |
| Wellness | Pinned at 1.5/10 | Varies 1–10 (mean ~5.6) |
| CV features in ML training | Yes (fake — laundered fatigue) | No (removed) |
| Effect of uploading a video | ~0% (mapped to noise) | Up to 40% (rule-based channel) |
| InjuryTypeClassifier | Display-only, irrelevant | Display + 30% blend into biomech_risk |
| Feature count | ~62 | ~50 |
| MQS persistence across reloads | NaN bug | Fixed |
| Model architecture | 1 channel × 4 voices | 2 channels × 5 voices |

## Files to replace

Drop these from the refactor zip directly over your project:

```
config.py
src/utils/data_generator.py
src/utils/build_combined_dataset.py
src/cv/biomech_risk_module.py          ← NEW FILE
src/ml/hybrid_predictor.py
src/realtime/feature_builder.py
dashboard/realworld_app.py
dashboard/synthetic_app.py
run.txt
```

## Run order (clean rebuild)

```bash
# 1. Regenerate synthetic data with new dynamics
python -m src.utils.data_generator

# 2. Rebuild combined dataset (no CV columns)
python -m src.utils.build_combined_dataset

# 3. Retrain ALL models — they must learn the new label distribution
python -m src.ml.model_trainer
python -m src.ml.ensemble_model
python -m src.ml.lstm_trainer
python -m src.ml.transformer_trainer
python -m src.ml.train_hazard_model

# 4. Wipe v3 test athletes (their saved injury_risk_score values came from
#    the broken v3 models)
rm -f data/history/athlete_*.csv      # macOS/Linux
# del /q data\history\athlete_*.csv   # Windows

# 5. Launch dashboards
streamlit run dashboard/realworld_app.py
streamlit run dashboard/synthetic_app.py
```

Total wall-clock time: ~15–20 minutes on a typical laptop CPU.

## How to verify each phase worked

### After step 1 (`data_generator`)
The script prints a summary. You should see something like:

```
Risk range:    0.0 – 100.0  (mean 44.0, std 17.9)
Risk pcts:     1st=7.6  99th=86.5
Above 80: ~950/36500 (2.6%)  Below 20: ~4080/36500 (11.2%)
Fatigue level mean 4.43 (std 1.92)
Wellness mean    5.58 (std 1.92)
```

If you still see `Risk range: 15 – 77` you're running the old generator.

### After step 2 (`build_combined_dataset`)
Check the column count:
```python
import pandas as pd
df = pd.read_csv("data/combined_dataset.csv")
print("Cols:", len(df.columns))
# Should be ~50 (was 62). The 9 fake CV columns
# (posture_symmetry, balance_score, knee_velocity, ...) should NOT appear.
print([c for c in df.columns if "velocity" in c or c == "joint_variability"])
# Should be []
```

### After step 3 (model retraining)
Each trainer prints test metrics. R² should be in roughly the same range as
v3 (≈ 0.6–0.7), but the predictions can now span 0–100 — load the dashboard
and try the extreme-input case from your screenshot:

> 299 min, HRV 15, sleep 3h, fatigue 10, intensity 10, 60 sprints

Workload-only result should now sit in the **80–95 range** instead of 60–66.

### After step 5 (dashboard)
The Real-World Dashboard's Prediction Results page should now show **three
gauges in a row**:

```
[💪 Workload]   [🔀 Final Hybrid]   [🎥 Biomech]
```

Without a video, the right gauge shows a placeholder ("Upload a training
video..."). With a video, it shows the rule-based score, and below the
gauges a "Biomechanical Risk Breakdown" panel lists the rules that
triggered, with severity, points, observed value, and threshold.

## Quick sanity test: video makes a difference

Create one athlete, run the same workload twice — once without a video,
once with a video showing poor mechanics. Final risk scores should differ
by 15–35 points, with the biomech-included version higher.

If you see identical results both times, the predictor isn't receiving
`video_features=...`. Check that:

1. `dashboard/realworld_app.py`'s `build_features_and_predict(...)` is
   called with `video_features=video_features_for_predict` (it is in v4
   line ~785).
2. `src/ml/hybrid_predictor.py` `predict()` accepts a `video_features`
   keyword (it does in v4).
3. `src/cv/biomech_risk_module.py` exists and imports cleanly:
   ```python
   from src.cv.biomech_risk_module import compute_biomech_risk
   ```

## Tuning knobs (in `config.py`)

```python
BIOMECH_VIDEO_WEIGHT = 0.40   # max α — raise to 0.50 for video-heavy use
BIOMECH_ALPHA_MIN    = 0.00   # floor (0 = video must earn its weight)
BIOMECH_ALPHA_MAX    = 0.40   # ceiling (cap)
BIOMECH_CLASSIFIER_BLEND = 0.30  # weight given to InjuryTypeClassifier max
```

If you want video to count for more, raise `BIOMECH_VIDEO_WEIGHT` and
`BIOMECH_ALPHA_MAX` together (they should match unless you want a hard cap
that's tighter than the headline weight).

## Backward compatibility

* `predictor.predict(df)` still works — when called without
  `video_features=`, the biomech channel returns 0 and α_eff = 0, so the
  result is workload-only (identical to v3 behaviour, modulo the new
  training data).
* `MultiModalPredictor` in `src/realtime/multimodal_predictor.py` was
  already deprecated in v3 and is unchanged.
* All saved model artefacts are replaced by the retraining step. You do
  NOT need to keep v3 `.pkl` / `.keras` files.
* `data/history/*.csv` files saved by v3 contain an `injury_risk_score`
  column produced by the v3 model — those numbers are stale. Delete them
  and start fresh as instructed in step 4 above.

## Known follow-ups (not in this refactor)

* **Feature importance display** in the synthetic-data dashboard still uses
  the old workload-only feature list — works fine, but the importance bars
  will only show ~50 features now (no CV columns).
* The "Biomechanics Radar" chart on the Athlete Dashboard page still
  reads `posture_symmetry_mean`, `balance_stability_mean`, etc. from the
  saved session row. Those values come from the video upload (still
  present, just not used as ML features). Nothing to fix.
* `sequence_dataset.py` and `feature_engineering.py` are unchanged — they
  don't reference CV columns directly, so the existing logic still works
  on the smaller feature set.
