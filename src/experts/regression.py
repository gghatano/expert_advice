"""Regression-based Experts: Ridge, Huber, and KNN with lag features."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.neighbors import KNeighborsRegressor

from src.experts.base import BaseExpert

# ---------------------------------------------------------------------------
# Feature construction helper
# ---------------------------------------------------------------------------

_LAGS = [1, 2, 3, 24, 168]


def _build_features(history: pd.Series, tstamp: pd.Timestamp) -> np.ndarray:
    """Build a 1-D feature vector from lag values and calendar features.

    Features (in order):
        lag_1, lag_2, lag_3, lag_24, lag_168, hour, day_of_week
    Missing lags (history too short) are filled with 0.
    """
    feats: list[float] = []
    n = len(history)
    for lag in _LAGS:
        if n >= lag:
            val = float(history.iloc[-lag])
            feats.append(0.0 if math.isnan(val) else val)
        else:
            feats.append(0.0)
    feats.append(float(tstamp.hour))
    feats.append(float(tstamp.dayofweek))
    return np.array(feats, dtype=np.float64)


def _build_training_data(
    history: pd.Series,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Build X, y matrices from *history* for supervised training.

    Each sample i uses lags relative to position i and the timestamp at
    position i as features, with the value at position i as the target.

    Returns None when there are not enough observations to form any sample.
    """
    n = len(history)
    min_required = max(_LAGS) + 1  # need at least 169 obs for a full feature set
    # We can still train with fewer obs, but need at least lag-1 + 1 = 2
    if n < 2:
        return None

    start = min(max(_LAGS), n - 1)  # first index where we can build a sample
    # Actually, we allow partial lags (filled with 0), so start from index 1
    start = 1

    X_rows: list[np.ndarray] = []
    y_vals: list[float] = []

    for i in range(start, n):
        sub = history.iloc[:i]
        ts = history.index[i]
        feat = _build_features(sub, ts)
        target = float(history.iloc[i])
        if math.isnan(target):
            continue
        X_rows.append(feat)
        y_vals.append(target)

    if len(X_rows) == 0:
        return None

    return np.vstack(X_rows), np.array(y_vals, dtype=np.float64)


def _safe_last(history: pd.Series) -> float:
    """Return the last non-NaN value, or 0.0 as ultimate fallback."""
    if history is None or len(history) == 0:
        return 0.0
    last = history.iloc[-1]
    if math.isnan(last):
        non_nan = history.dropna()
        return float(non_nan.iloc[-1]) if len(non_nan) > 0 else 0.0
    return float(last)


# ---------------------------------------------------------------------------
# RidgeLag
# ---------------------------------------------------------------------------


class RidgeLag(BaseExpert):
    """Ridge regression expert with lag + calendar features.

    Parameters
    ----------
    alpha : float
        Regularisation strength passed to ``sklearn.linear_model.Ridge``.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self._alpha = alpha
        self._model: Ridge | None = None

    @property
    def name(self) -> str:
        return f"RidgeLag_{self._alpha}"

    def fit(self, history: pd.Series, **kwargs) -> None:
        data = _build_training_data(history)
        if data is None:
            self._model = None
            return
        X, y = data
        model = Ridge(alpha=self._alpha)
        model.fit(X, y)
        self._model = model

    def predict_next(
        self,
        history: pd.Series,
        tstamp: pd.Timestamp,
        exog: dict | None = None,
    ) -> float:
        if self._model is None or history is None or len(history) == 0:
            return _safe_last(history)
        feat = _build_features(history, tstamp).reshape(1, -1)
        pred = float(self._model.predict(feat)[0])
        if math.isnan(pred) or math.isinf(pred):
            return _safe_last(history)
        return pred


# ---------------------------------------------------------------------------
# HuberRegressorLag
# ---------------------------------------------------------------------------


class HuberRegressorLag(BaseExpert):
    """Huber-loss regression expert — robust to outliers.

    Uses the same lag + calendar feature set as :class:`RidgeLag`.
    """

    def __init__(self) -> None:
        self._model: HuberRegressor | None = None

    @property
    def name(self) -> str:
        return "HuberRegressorLag"

    def fit(self, history: pd.Series, **kwargs) -> None:
        data = _build_training_data(history)
        if data is None:
            self._model = None
            return
        X, y = data
        model = HuberRegressor()
        model.fit(X, y)
        self._model = model

    def predict_next(
        self,
        history: pd.Series,
        tstamp: pd.Timestamp,
        exog: dict | None = None,
    ) -> float:
        if self._model is None or history is None or len(history) == 0:
            return _safe_last(history)
        feat = _build_features(history, tstamp).reshape(1, -1)
        pred = float(self._model.predict(feat)[0])
        if math.isnan(pred) or math.isinf(pred):
            return _safe_last(history)
        return pred


# ---------------------------------------------------------------------------
# KNNLag
# ---------------------------------------------------------------------------


class KNNLag(BaseExpert):
    """K-nearest-neighbours regression with lag + calendar features.

    Parameters
    ----------
    k : int
        Number of neighbours.
    max_samples : int
        Only the most recent *max_samples* observations are used for
        building the training set, keeping inference lightweight.
    """

    def __init__(self, k: int = 5, max_samples: int = 500) -> None:
        self._k = k
        self._max_samples = max_samples
        self._model: KNeighborsRegressor | None = None

    @property
    def name(self) -> str:
        return f"KNNLag_{self._k}"

    def fit(self, history: pd.Series, **kwargs) -> None:
        # Trim to most recent max_samples for lightweight training
        if len(history) > self._max_samples:
            history = history.iloc[-self._max_samples :]

        data = _build_training_data(history)
        if data is None:
            self._model = None
            return
        X, y = data
        # k cannot exceed the number of training samples
        effective_k = min(self._k, len(X))
        if effective_k < 1:
            self._model = None
            return
        model = KNeighborsRegressor(n_neighbors=effective_k)
        model.fit(X, y)
        self._model = model

    def predict_next(
        self,
        history: pd.Series,
        tstamp: pd.Timestamp,
        exog: dict | None = None,
    ) -> float:
        if self._model is None or history is None or len(history) == 0:
            return _safe_last(history)
        feat = _build_features(history, tstamp).reshape(1, -1)
        pred = float(self._model.predict(feat)[0])
        if math.isnan(pred) or math.isinf(pred):
            return _safe_last(history)
        return pred
