# v4 Merged Bundle — Installation Guide

This bundle contains the v4 two-channel refactor from `athlete_injury_prediction_v4_refactor.zip`
PLUS the missing fixes that the zip didn't include. Together they fix every
issue identified in the audit, including:

- the NaN crash that was forcing all predictions to 50.0 (zip already had this)
- batch-dependent hazard normalisation (now fixed)
- the inert sidebar slider (now fixed — wired to `set_biomech_weight()`)
- raw video columns silently sneaking into the ML predictor (now explicitly stripped)
- stale `lstm_trainer.py` architecture comments (now match the code)

## File mapping (where each file goes)

Drop each file into the matching path inside your existing project folder.
Windows: `C:\Users\LOQ\Documents\VS code\athlete_injury_prediction\`

| Patched file                | Replace in your project                              |
|-----------------------------|------------------------------------------------------|
| `config.py`                 | `config.py`                                          |
| `data_generator.py`         | `src/utils/data_generator.py`                        |
| `build_combined_dataset.py` | `src/utils/build_combined_dataset.py`                |
| `biomech_risk_module.py`    | `src/cv/biomech_risk_module.py`  *(NEW FILE)*        |
| `feature_builder.py`        | `src/realtime/feature_builder.py`                    |
| `hybrid_predictor.py`       | `src/ml/hybrid_predictor.py`                         |
| `hazard_model.py`           | `src/ml/hazard_model.py`                             |
| `train_hazard_model.py`     | `src/ml/train_hazard_model.py`                       |
| `lstm_trainer.py`           | `src/ml/lstm_trainer.py`                             |
| `realworld_app.py`          | `dashboard/realworld_app.py`                         |
| `synthetic_app.py`          | `dashboard/synthetic_app.py`                         |
| `run.txt`                   | `run.txt`                                            |

## Step-by-step installation

### Step 0 — Stop running dashboards

Press `Ctrl+C` in any terminal running Streamlit. You can't replace files
while Streamlit has them open.

### Step 1 — Back up your project

Right-click your `athlete_injury_prediction` folder → Copy → Paste. Rename
the copy to `athlete_injury_prediction_BACKUP`. Don't skip this.

### Step 2 — Replace files

Copy each patched file from this bundle into the path shown above. When
Windows asks about overwriting, click **Replace**.

`biomech_risk_module.py` is a NEW FILE (didn't exist before) — just drop it
into `src/cv/`.

### Step 3 — Open a terminal in the project root

```cmd
cd "C:\Users\LOQ\Documents\VS code\athlete_injury_prediction"
.\venv\Scripts\activate
```

You should see `(venv)` at the start of your prompt.

### Step 4 — Wipe stale broken-period data

The athletes you tested under v3 have `injury_risk_score = 50.0` saved on
disk for every session. No code fix can update those retroactively. Easiest
approach is wipe and start fresh:

```cmd
del /q data\history\athlete_*.csv
del data\realworld\athletes_meta.json
```

(macOS/Linux: `rm -f data/history/athlete_*.csv data/realworld/athletes_meta.json`)

### Step 5 — Run the v4 retraining pipeline

This is mandatory because the new data generator produces a different
target distribution (0–100 spread instead of 16–77 compressed). Every model
needs to retrain on the new labels.

Run these commands in this exact order:

```cmd
python -m src.utils.data_generator
python -m src.utils.build_combined_dataset
python -m src.ml.model_trainer
python -m src.ml.ensemble_model
python -m src.ml.lstm_trainer
python -m src.ml.transformer_trainer
python -m src.ml.train_hazard_model
```

Total time: ~10–15 minutes on a typical laptop.

### Step 6 — Verify the data generator output

After `python -m src.utils.data_generator` finishes, you should see lines
like these in the terminal:

```
Risk range: 0.0 – 100.0
Risk mean:  ~44 (std ~18)
Rows above 80: ~2.6%
Rows below 20: ~11%
Fatigue mean: ~4.4/10  (was 9.0/10 in v3 — saturated)
Wellness mean: ~5.6/10 (was 1.46/10 in v3 — collapsed)
```

If you still see "Risk range: 16 – 77" or fatigue mean near 9.0, the new
`data_generator.py` didn't replace correctly. Redo Step 2 for that file.

### Step 7 — Verify the hazard model retraining

After `python -m src.ml.train_hazard_model`, you should see a line:

```
Train max raw hazard (saved as normalisation reference): 24.84
```

If you see this, the new `train_max_hazard` reference is now baked into
your `hazard_model.pkl`. The dashboard will no longer print the
"WARNING: hazard model bundle has no `train_max_hazard` reference" message
at startup.

### Step 8 — Launch the dashboard

```cmd
streamlit run dashboard/realworld_app.py
```

Wait for the browser tab to open. In the terminal you should see (in order):

- `Loading stacked ensemble...`
- `Loading LSTM...`
- `Loading transformer...`
- `Loading hazard (survival) model...`
- `All models loaded.`

No `WARNING: hazard model bundle has no train_max_hazard` message.

### Step 9 — Smoke-test predictions

Create a fresh test athlete and add a session with extreme inputs:

| Field             | Value |
|-------------------|-------|
| Training duration | 299 min |
| HRV               | 15 |
| Sleep             | 3 h |
| Fatigue           | 10 |
| Sprint count      | 60 |
| Wellness          | 1 |
| Intensity         | 10 |

**Expected (v4 working correctly):**

- Final risk: 80–95 (NOT pinned at 50, NOT pinned at 66)
- Stacked Ensemble + Hazard show real numbers (not "N/A")
- LSTM + Transformer show "N/A — Need more sessions" (this is correct;
  the cold-start guard refuses to make a temporal prediction with <30
  sessions)
- "Blend (this prediction)" caption shows ~70% Ensemble / ~30% Hazard

Then repeat with a rest-day session (training 30 min, fatigue 2,
wellness 9, sleep 9, intensity 1):

**Expected:** Final risk 5–25.

If you see the full range from low-risk to high-risk responding to
real input changes, the v4 refactor is working as designed.

### Step 10 — Test the biomech channel (optional, requires a video)

1. Upload a training video to a new session.
2. Wait for "Analysed N frames via MediaPipe" message.
3. Save the session.
4. On the Prediction Results page you should see:
   - Three gauges side by side: Workload Risk, Biomech Risk, Final Hybrid
   - A "🎥 Biomechanical Risk — Rule Breakdown" panel listing which rules
     fired (e.g. "Knee Variability — High — +25.0 pts")
   - The Final Hybrid gauge differs from the Workload gauge by an amount
     proportional to the biomech score

Try the sidebar slider 🎚️ Biomech (video) weight α — moving it changes
the influence of video on the next prediction.

## Sanity checks

After Step 8 the dashboard should:

- Never print "Predictor error: Input X contains NaN"
- Never pin every athlete at exactly 50.0
- Never print "WARNING: hazard model bundle has no train_max_hazard"
- Show real numbers in the 5-Model Breakdown for Stacked Ensemble + Hazard
  even at session 1
- Show "N/A" honestly for LSTM/Transformer until 30 sessions exist
- Vary predictions across the 0–100 range based on input quality

## What's still NOT addressed in this bundle

These are documented design-level concerns, not crashes:

- **Synthetic→real domain shift**: All ML models trained on simulated
  workload data. Real wearables produce statistically similar but not
  identical features. Predictions are advisory, not calibrated.
- **Biomechanical extraction sport-context**: The biomech rules calibrate
  thresholds against running/squatting motion. Tennis and other sports
  produce wide normal joint ranges that the rules currently flag. Step 2
  of our roadmap addresses this.
- **2D pose only**: MediaPipe `pose_landmarks` are used (2D image coords).
  Knee valgus and sagittal forward lean (the actual ACL/hamstring
  predictors) require 3D `pose_world_landmarks`. Future work.
- **`injury_risk_score` self-loop**: The dashboard saves the model's
  predicted score as `injury_risk_score` in the history CSV; the next
  session's `injury_risk_trend_7d` then uses those predicted values as if
  they were ground truth. Documented in `realworld_app.py:save_session()`.
  Cannot be fixed without real injury labels.

## Troubleshooting

**"Predictor error: Input X contains NaN"**

`feature_builder.py` and/or `realworld_app.py` didn't replace correctly.
In `feature_builder.py`, search for `return_full_history` — it should
appear in the function signature.

**Predictions still pinned at 50.0**

You forgot to retrain after replacing `data_generator.py`. Redo Step 5
in full. The old model files in `models/ml/` were trained on v3's
compressed labels and cannot extrapolate.

**"WARNING: hazard model bundle has no train_max_hazard" printed at
startup**

You replaced `hybrid_predictor.py` but didn't retrain `train_hazard_model.py`.
Run `python -m src.ml.train_hazard_model` again.

**Sidebar slider shows but doesn't affect predictions**

`hybrid_predictor.py` didn't replace correctly. Search for
`set_biomech_weight` — it should appear after `_FALLBACK_WEIGHTS`.

**Streamlit shows old version even after replacing files**

Stop Streamlit (`Ctrl+C`), restart it, hard-refresh the browser
(`Ctrl+Shift+R`). Streamlit caches aggressively.

That's all of Step 1. After confirming this works, we can proceed to
Step 2 (the biomechanical extraction improvements — Bundle A/B/C/D from
the audit).
