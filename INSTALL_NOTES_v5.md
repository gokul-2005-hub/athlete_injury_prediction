# v5 — Step 2 Bundle (Bundles A + B + C)

This bundle layers three sets of biomechanical-extraction improvements on
top of the v4 Step 1 install. **Step 1 must be installed and working
first** — these files assume the v4 two-channel architecture and the
NaN-crash / hazard normalisation fixes are in place.

## What's in this bundle

| Bundle | What it does | Where |
|---|---|---|
| **A** — Surgical fixes | Camera-distance normalisation, visibility weighting, jitter smoothing, honest naming, real confidence formula | `pose_estimator.py`, `biomech_risk_module.py`, `realworld_app.py` |
| **B** — Sport-aware thresholds | 5 sports (Tennis / Badminton / Running / Squat / Generic) with their own healthy ranges. Tennis no longer flags lunging as injury risk | `biomech_risk_module.py`, `realworld_app.py` (new dropdown) |
| **C** — Per-phase routing | Knee variability now scored against landing-phase frames (the real ACL signal). Torso lean against running. Balance against single-support frames. With graceful fallback to global stats when phase data is thin | `pose_estimator.py`, `biomech_risk_module.py`, `realworld_app.py` |

## What you'll see change

1. **Same tennis video** as before now scores ~17 biomech risk under
   "Tennis" sport context, vs ~40 under "Running" context. The dashboard
   stops flagging legitimate tennis lunging as ACL risk.

2. **Camera-distance bias gone.** Same person at 2 m vs 5 m from camera
   now produces the same balance / shoulder symmetry numbers.

3. **MediaPipe jitter (~2-3°) no longer inflates knee_std.** The 5-frame
   moving average smooths out single-frame noise without flattening real
   motion.

4. **Confidence is honest.** A 6-second blurry clip with 45% pose
   detection used to score 1.00 confidence; now scores 0.18.

5. **Phase C is the biggest clinical improvement.** A video where the
   athlete's landings have 38° knee std but global average is 18° now
   correctly scores HIGH risk (the ACL signal isn't averaged away across
   running and standing frames).

6. **New "Phase" column** in the biomech rules table shows whether each
   rule scored against landing / running / global. Rules that fell back
   to global because their target phase had < 8 frames are flagged
   `[FB]` with reduced weight.

## File mapping

| Patched file | Drop into |
|---|---|
| `pose_estimator.py`        | `src/cv/pose_estimator.py` |
| `biomech_risk_module.py`   | `src/cv/biomech_risk_module.py` |
| `hybrid_predictor.py`      | `src/ml/hybrid_predictor.py` |
| `realworld_app.py`         | `dashboard/realworld_app.py` |

## Step-by-step install

### Step 0 — Stop running dashboards

`Ctrl+C` in the Streamlit terminal. Close any VS Code tabs that have
the four files above open.

### Step 1 — Verify v4 Step 1 is installed

Open a terminal in the project root:

```cmd
cd "C:\Users\LOQ\Documents\VS code\athlete_injury_prediction"
.\venv\Scripts\activate
```

Quick check that v4 is in place — open `dashboard/realworld_app.py`
and search (`Ctrl+F`) for `return_full_history`. If you find it, v4
is installed. If not, install v4 Step 1 first (see `INSTALL_NOTES_v4.md`).

### Step 2 — Back up the four files you're about to replace

```cmd
copy src\cv\pose_estimator.py        src\cv\pose_estimator.py.v4backup
copy src\cv\biomech_risk_module.py   src\cv\biomech_risk_module.py.v4backup
copy src\ml\hybrid_predictor.py      src\ml\hybrid_predictor.py.v4backup
copy dashboard\realworld_app.py      dashboard\realworld_app.py.v4backup
```

### Step 3 — Replace files

Drop each patched file into its corresponding location. Confirm replace
when Windows asks.

### Step 4 — Verify each file replaced correctly

Search each file for a string that only exists in the patched version:

| File | Search for |
|---|---|
| `pose_estimator.py` | `SMOOTH_WINDOW = 5` |
| `biomech_risk_module.py` | `SPORT_THRESHOLDS` |
| `hybrid_predictor.py` | `biomech_sport:` |
| `realworld_app.py` | `biomech_sport_select` |

If any search comes up empty, that file didn't replace correctly. Redo
Step 3 for it.

### Step 5 — Restart the dashboard

```cmd
streamlit run dashboard/realworld_app.py
```

**No retraining needed.** Bundles A/B/C are pure video-pipeline changes
— they don't touch the ML models on disk.

### Step 6 — Test the new sport selector

1. Go to **New Session** in the dashboard.
2. Scroll down to the video upload area.
3. You should see a new dropdown: **"Sport context for biomech analysis"**
   with 5 options: Tennis / Badminton / Running / Squat / Strength /
   Generic.
4. The dropdown is pre-filled from the active athlete's sport (or
   "Generic" if their sport isn't in the list).
5. Upload a video, save the session.
6. On the **Prediction Results** page, scroll to the biomech panel:
   - You should see a banner showing **Sport context: Tennis** (or
     whichever you picked) and **Phase distribution** counts.
   - The rules table has a new **Phase** column showing landing /
     running / global per rule.

### Step 7 — Validate the tennis fix

Take the same video that previously got flagged for "Knee instability —
High" under the old code. Try it twice:

| Sport context selected | Expected biomech risk |
|---|---|
| Running | Higher (~30-50 if knee_std is 25-30°) |
| Tennis | Substantially lower (~10-20 for the same input) |

If you see a clear difference between the two, sport-aware thresholds
are working.

## What's NOT in this bundle (left for later if needed)

- **Bundle D — 3D landmarks.** Switching MediaPipe from `pose_landmarks`
  (2D) to `pose_world_landmarks` (3D) would enable true knee valgus
  detection, which is THE non-contact ACL signal. About a 1-day refactor
  that breaks every existing biomech threshold and forces re-tuning.
  Worth doing if you want clinical-grade ACL detection specifically.
- **Athlete-level threshold scaling** (age / level adjustments). Could
  be added on top of Bundle B if the 5-sport granularity isn't enough.
- **Camera-angle detection** (front view vs side view). Currently the
  metrics assume a roughly perpendicular view; an oblique camera angle
  introduces systematic bias.

## Troubleshooting

**Sport dropdown doesn't appear** — `realworld_app.py` didn't replace.
Search for `biomech_sport_select` to confirm.

**Same biomech risk under different sports** — `biomech_risk_module.py`
or `hybrid_predictor.py` didn't replace. Search for `SPORT_THRESHOLDS`
in `biomech_risk_module.py`.

**"Phase" column missing from rules table** — `realworld_app.py` didn't
replace. The rules dict needs the `phase` and `fallback` keys to render.

**Streamlit shows old version** — `Ctrl+C` to stop, restart streamlit,
hard-refresh browser (`Ctrl+Shift+R`).

**Sports list doesn't match expectation** — open `biomech_risk_module.py`
and look for `SPORT_LABEL_TO_KEY`. The 5 entries are fixed in code; if
you want to add Cricket / Volleyball / etc., you'd add a row to that
dict and a matching threshold table.

## What to send back

After install + Step 6 + Step 7, ideally a screenshot of the new biomech
panel showing:
- The "Sport context: …" banner
- The phase distribution
- The Phase column in the rules table
- Different biomech_risk values when you switch the sport dropdown

That confirms all three bundles landed correctly.
