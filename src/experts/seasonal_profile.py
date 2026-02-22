"""Seasonal profile Expert – day-of-week x hour-of-day mean profile."""

import numpy as np
import pandas as pd

from src.experts.base import BaseExpert


class STLSeasonalMean(BaseExpert):
    """Predict using the mean value for each (day-of-week, hour) slot.

    During :meth:`fit`, a 7x24 profile matrix is computed from *history*.
    :meth:`predict_next` looks up the corresponding slot for *tstamp*.
    """

    def __init__(self) -> None:
        self._profile: np.ndarray | None = None  # shape (7, 24)

    @property
    def name(self) -> str:
        return "STLSeasonalMean"

    def fit(self, history: pd.Series, **kwargs) -> None:
        """Compute the 7x24 day-of-week x hour mean profile."""
        if history.empty:
            self._profile = None
            return

        profile = np.full((7, 24), np.nan)
        dow = history.index.dayofweek  # 0=Mon … 6=Sun
        hour = history.index.hour

        for d in range(7):
            for h in range(24):
                mask = (dow == d) & (hour == h)
                vals = history[mask]
                if len(vals) > 0:
                    mean_val = vals.mean()
                    profile[d, h] = mean_val if not np.isnan(mean_val) else 0.0

        # Replace any remaining NaN slots with 0.0
        profile = np.nan_to_num(profile, nan=0.0)
        self._profile = profile

    def predict_next(
        self,
        history: pd.Series,
        tstamp: pd.Timestamp,
        exog: dict | None = None,
    ) -> float:
        """Return the profile value for *tstamp*'s day-of-week and hour."""
        if self._profile is None:
            return 0.0
        val = float(self._profile[tstamp.dayofweek, tstamp.hour])
        return val if not np.isnan(val) else 0.0
