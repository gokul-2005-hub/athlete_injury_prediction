# src/ml/sequence_dataset.py
#
# FIX: Critical cross-athlete contamination bug.
#
# Previous code called create_sequences() on the full date-sorted dataframe.
# Because all 50 athletes share the same calendar dates (microseconds apart),
# sort_values("date") interleaves them:
#   row 0  → athlete_1,  day 1
#   row 1  → athlete_2,  day 1   ← different athlete
#   row 29 → athlete_30, day 1
#
# A 30-row window therefore contained 30 DIFFERENT athletes on the same day
# — cross-sectional, not temporal.  The LSTM/Transformer were learning
# cross-athlete correlations, not within-athlete temporal patterns.
#
# Fix: group by athlete_id before slicing sequences, and never let a window
# cross an athlete boundary.

import numpy as np
import pandas as pd
from typing import Optional


def create_sequences(
    df: pd.DataFrame,
    sequence_length: int = 30,
    athlete_col: Optional[str] = "athlete_id",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a tabular DataFrame into (X_seq, y_seq) arrays suitable for
    LSTM / Transformer training.

    If `athlete_col` is present the DataFrame is grouped by athlete so that
    no sequence ever spans two different athletes.  This is the critical fix
    vs the previous implementation which created sequences across the full
    date-sorted frame, mixing rows from different athletes.

    Args:
        df:              Must contain 'injury_risk_score_next' as the target
                         column.  Sort by ["athlete_id", "date"] before calling.
        sequence_length: Number of past time-steps per sequence window.
        athlete_col:     Column name for athlete grouping.  Pass None to skip
                         grouping (single-athlete datasets / unit tests).

    Returns:
        X: float32 array of shape (N, sequence_length, n_features)
        y: float32 array of shape (N,)
    """

    # FIX: removed meaningless `c is not None` check; added explicit exclusion
    # of non-feature columns that could sneak in during real-time inference
    _exclude = {"date", "injury_risk_score_next", "injury_risk_score",
                "session_notes", "has_video_biomechanics", "risk_category"}
    if athlete_col:
        _exclude.add(athlete_col)
    feature_cols = [
        c for c in df.columns
        if c not in _exclude and df[c].dtype in ("float64", "float32", "int64", "int32")
    ]

    sequences: list[np.ndarray] = []
    targets:   list[float]      = []

    if athlete_col and athlete_col in df.columns:
        # ── Per-athlete windowing ────────────────────────────────────
        # Each athlete contributes (n_sessions - sequence_length) sequences.
        # No window ever crosses an athlete boundary.
        groups = df.groupby(athlete_col, sort=False)

        for _, group in groups:
            group = group.reset_index(drop=True)

            X_a = group[feature_cols].values.astype(np.float32)
            y_a = group["injury_risk_score_next"].values.astype(np.float32)

            n = len(X_a)
            if n <= sequence_length:
                # Too few sessions for even one window — skip this athlete.
                continue

            for i in range(n - sequence_length):
                sequences.append(X_a[i : i + sequence_length])
                targets.append(y_a[i + sequence_length])

    else:
        # ── Fallback: single athlete / no grouping ───────────────────
        X = df[feature_cols].values.astype(np.float32)
        y = df["injury_risk_score_next"].values.astype(np.float32)

        for i in range(len(X) - sequence_length):
            sequences.append(X[i : i + sequence_length])
            targets.append(y[i + sequence_length])

    if not sequences:
        raise ValueError(
            f"No sequences were created.  Check that each athlete has more "
            f"than {sequence_length} sessions and that 'injury_risk_score_next' "
            f"is present in the DataFrame."
        )

    X_out = np.stack(sequences, axis=0)          # (N, seq_len, n_features)
    y_out = np.array(targets, dtype=np.float32)  # (N,)

    return X_out, y_out
