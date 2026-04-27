# src/ml/feature_engineering.py

import pandas as pd
import numpy as np


class FeatureEngineer:

    def __init__(self, df):
        self.df = df.copy()

    # --------------------------------------------------
    # ACWR
    # --------------------------------------------------

    def add_acwr(self):

        acute = (
            self.df
            .groupby("athlete_id")["training_duration"]
            .transform(lambda x: x.rolling(7, min_periods=1).mean())
        )

        chronic = (
            self.df
            .groupby("athlete_id")["training_duration"]
            .transform(lambda x: x.rolling(28, min_periods=1).mean())
        )

        self.df["acwr"] = acute / (chronic + 1e-6)

    # --------------------------------------------------
    # EWMA
    # --------------------------------------------------

    def add_ewma_features(self):

        spans = [7, 14, 28]

        for span in spans:

            self.df[f"training_duration_ewma_{span}"] = (
                self.df
                .groupby("athlete_id")["training_duration"]
                .transform(lambda x: x.ewm(span=span).mean())
            )

    # --------------------------------------------------
    # VOLATILITY
    # --------------------------------------------------

    def add_volatility_features(self):

        cols = [
            "training_duration",
            "sleep_hours",
            "fatigue_level"
        ]

        for col in cols:

            self.df[f"{col}_std_7"] = (
                self.df
                .groupby("athlete_id")[col]
                .transform(lambda x: x.rolling(7, min_periods=1).std().fillna(0))
            )

    # --------------------------------------------------
    # ACCELERATION
    # --------------------------------------------------

    def add_acceleration_features(self):

        self.df["training_duration_diff1"] = (
            self.df
            .groupby("athlete_id")["training_duration"]
            .diff(1)
            .fillna(0)
        )

        self.df["training_duration_diff2"] = (
            self.df
            .groupby("athlete_id")["training_duration"]
            .diff(2)
            .fillna(0)
        )

    # --------------------------------------------------
    # INTERACTIONS
    # --------------------------------------------------

    def add_interaction_features(self):

        self.df["training_fatigue_interaction"] = (
            self.df["training_duration"] * self.df["fatigue_level"]
        )

        self.df["training_sleep_interaction"] = (
            self.df["training_duration"] * self.df["sleep_hours"]
        )

        self.df["hrv_fatigue_interaction"] = (
            self.df["heart_rate_variability"] * self.df["fatigue_level"]
        )

        # SAFE ACWR interaction
        if "acwr" in self.df.columns:

            self.df["acwr_fatigue_interaction"] = (
                self.df["acwr"] * self.df["fatigue_level"]
            )

    # --------------------------------------------------
    # FATIGUE INDEX
    # --------------------------------------------------

    def add_fatigue_index(self):

        self.df["sleep_debt"] = 8 - self.df["sleep_hours"]

        if "acwr" in self.df.columns:

            self.df["fatigue_index"] = (
                0.5 * self.df["acwr"]
                + 0.3 * self.df["fatigue_level"]
                + 0.2 * self.df["sleep_debt"]
            )

    # --------------------------------------------------
    # BASELINE FEATURES
    # --------------------------------------------------

    def add_baseline_features(self):

        cols = [
            "heart_rate_variability",
            "sleep_hours",
            "fatigue_level"
        ]

        for col in cols:

            mean = (
                self.df
                .groupby("athlete_id")[col]
                .transform(lambda x: x.expanding().mean())
            )

            std = (
                self.df
                .groupby("athlete_id")[col]
                .transform(lambda x: x.expanding().std())
            )

            self.df[f"{col}_baseline_dev"] = self.df[col] - mean
            self.df[f"{col}_baseline_z"] = (self.df[col] - mean) / (std + 1e-6)

    # --------------------------------------------------
    # MAIN PIPELINE
    # --------------------------------------------------

    def run(self):

        # IMPORTANT ORDER
        self.add_acwr()
        self.add_ewma_features()
        self.add_volatility_features()
        self.add_acceleration_features()
        self.add_interaction_features()
        self.add_fatigue_index()
        self.add_baseline_features()

        return self.df