"""Moving-average family of Experts."""

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
# SMA — Simple Moving Average
# ---------------------------------------------------------------------------

class SMA(BaseExpert):
    """Predict the mean of the last *window* observations.

    Parameters
    ----------
    window : int
        Number of recent observations to average.
    """

    def __init__(self, window: int = 24) -> None:
        self._window = window

    @property
    def name(self) -> str:
        return f"SMA_{self._window}"

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

        if len(clean) == 0:
            return _safe_last(history)

        result = float(np.mean(clean.values))

        if math.isnan(result):
            return _safe_last(history)

        return result


# ---------------------------------------------------------------------------
# Median — Moving Median
# ---------------------------------------------------------------------------

class Median(BaseExpert):
    """Predict the median of the last *window* observations.

    Parameters
    ----------
    window : int
        Number of recent observations to take the median of.
    """

    def __init__(self, window: int = 24) -> None:
        self._window = window

    @property
    def name(self) -> str:
        return f"Median_{self._window}"

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

        if len(clean) == 0:
            return _safe_last(history)

        result = float(np.median(clean.values))

        if math.isnan(result):
            return _safe_last(history)

        return result
