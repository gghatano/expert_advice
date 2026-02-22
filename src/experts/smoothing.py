"""Exponential smoothing family of Experts."""

from __future__ import annotations

import math

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
# EMA — Exponential Moving Average
# ---------------------------------------------------------------------------

class EMA(BaseExpert):
    """Predict using Exponential Moving Average.

    The EMA is computed recursively:
        ema_0 = y_0
        ema_t = alpha * y_t + (1 - alpha) * ema_{t-1}

    The latest EMA value is returned as the forecast.

    Parameters
    ----------
    alpha : float
        Smoothing factor in (0, 1]. Higher values give more weight to
        recent observations.
    max_lookback : int
        Maximum number of recent observations used to compute EMA.
        Limits computation cost for very long histories.
    """

    def __init__(self, alpha: float = 0.1, max_lookback: int = 500) -> None:
        self._alpha = alpha
        self._max_lookback = max_lookback

    @property
    def name(self) -> str:
        return f"EMA_{self._alpha}"

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

        # Use at most max_lookback recent values for efficiency
        tail = history.iloc[-self._max_lookback:]
        clean = tail.dropna()

        if len(clean) == 0:
            return _safe_last(history)

        # Recursive EMA computation
        values = clean.values
        ema = float(values[0])
        alpha = self._alpha
        for i in range(1, len(values)):
            ema = alpha * float(values[i]) + (1 - alpha) * ema

        if math.isnan(ema) or math.isinf(ema):
            return _safe_last(history)

        return ema
