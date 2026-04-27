from pathlib import Path
from typing import Dict, Optional
import pandas as pd
from datetime import datetime, timezone


class AthleteHistoryStore:
    """
    Disk-backed per-athlete history manager.
    Each athlete has their own CSV file:
    data/history/athlete_<id>.csv
    """

    def __init__(self, base_dir: Optional[Path] = None):
        # Project root (…/athlete_injury_prediction)
        self.base_dir = base_dir or Path(__file__).resolve().parents[2]
        self.history_dir = self.base_dir / "data" / "history"
        self.history_dir.mkdir(parents=True, exist_ok=True)

    def _athlete_path(self, athlete_id: int) -> Path:
        """
        Return path to athlete CSV file.
        Example: data/history/athlete_1.csv
        """
        return self.history_dir / f"athlete_{athlete_id}.csv"

    def load_history(self, athlete_id: int) -> pd.DataFrame:
        """
        Load an athlete's history from disk.
        If it doesn't exist, return an empty DataFrame.
        """
        path = self._athlete_path(athlete_id)

        if not path.exists():
            return pd.DataFrame()

        df = pd.read_csv(path, parse_dates=["date"])
        df = df.sort_values("date").reset_index(drop=True)
        return df

    def append_session(
        self,
        athlete_id: int,
        session_data: Dict
    ) -> pd.DataFrame:
        """
        Append a new session to the athlete's history.

        session_data must contain all raw + engineered fields
        AND a 'date' field (datetime or ISO string).
        """

        path = self._athlete_path(athlete_id)

        # Ensure date exists
        if "date" not in session_data:
            session_data["date"] = datetime.now(timezone.utc)

        # Convert to DataFrame (single row)
        new_row = pd.DataFrame([session_data])
        new_row["date"] = pd.to_datetime(new_row["date"])

        # Load existing history
        history = self.load_history(athlete_id)

        # Append
        updated = pd.concat([history, new_row], ignore_index=True)

        # Sort by date (safety)
        updated = updated.sort_values("date").reset_index(drop=True)

        # Save back to disk
        updated.to_csv(path, index=False)

        return updated
