"""Naive family of Experts: simple baselines that require no training."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.experts.base import BaseExpert


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_last(history: pd.Series) -> float:
    """Return the last non-NaN value in *history*, or 0.0 as ultimate fallback."""
    if history is None or len(history) == 0:
        return 0.0
    last = history.iloc[-1]
    if math.isnan(last):
        non_nan = history.dropna()
        return float(non_nan.iloc[-1]) if len(non_nan) > 0 else 0.0
    return float(last)


# ---------------------------------------------------------------------------
# LastValue
# ---------------------------------------------------------------------------

class LastValue(BaseExpert):
    """Predict the most recent observed value."""

    @property
    def name(self) -> str:
        return "LastValue"

    def fit(self, history: pd.Series, **kwargs) -> None:  # noqa: D401
        pass  # nothing to learn

    def predict_next(
        self,
        history: pd.Series,
        tstamp: pd.Timestamp,
        exog: dict | None = None,
    ) -> float:
        return _safe_last(history)


# ---------------------------------------------------------------------------
# SeasonalNaive
# ---------------------------------------------------------------------------

class SeasonalNaive(BaseExpert):
    """Return the value observed *season_length* steps ago.

    Parameters
    ----------
    season_length : int
        Number of time steps defining one seasonal cycle (e.g. 24 for daily
        seasonality on hourly data, 168 for weekly).
    """

    def __init__(self, season_length: int = 24) -> None:
        self._season_length = season_length

    @property
    def name(self) -> str:
        return f"SeasonalNaive_{self._season_length}"

    def fit(self, history: pd.Series, **kwargs) -> None:
        pass

    def predict_next(
        self,
        history: pd.Series,
        tstamp: pd.Timestamp,
        exog: dict | None = None,
    ) -> float:
        if history is not None and len(history) >= self._season_length:
            val = history.iloc[-self._season_length]
            if not math.isnan(val):
                return float(val)
        # Fallback: not enough history — use last value
        return _safe_last(history)


# ---------------------------------------------------------------------------
# Drift
# ---------------------------------------------------------------------------

class Drift(BaseExpert):
    """Linear-trend extrapolation over a rolling window.

    Fits a straight line to the last *window* observations and projects
    one step ahead.

    Parameters
    ----------
    window : int
        Number of recent observations used for the trend estimate.
    """

    def __init__(self, window: int = 24) -> None:
        self._window = window

    @property
    def name(self) -> str:
        return f"Drift_{self._window}"

    def fit(self, history: pd.Series, **kwargs) -> None:
        pass

    def predict_next(
        self,
        history: pd.Series,
        tstamp: pd.Timestamp,
        exog: dict | None = None,
    ) -> float:
        if history is None or len(history) == 0:
            return 0.0

        tail = history.iloc[-self._window:]
        clean = tail.dropna()

        if len(clean) < 2:
            return _safe_last(history)

        # Simple linear regression: y on 0..n-1, predict at n
        n = len(clean)
        x = np.arange(n, dtype=np.float64)
        y = clean.values.astype(np.float64)
        slope = (np.dot(x, y) - n * x.mean() * y.mean()) / (
            np.dot(x, x) - n * x.mean() ** 2
        )
        intercept = y.mean() - slope * x.mean()
        prediction = intercept + slope * n  # one step beyond last index

        if math.isnan(prediction) or math.isinf(prediction):
            return _safe_last(history)

        return float(prediction)
