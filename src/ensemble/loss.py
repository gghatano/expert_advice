"""Loss functions for expert evaluation."""

from __future__ import annotations

import math


def mae_loss(y_true: float, y_pred: float) -> float:
    """Mean Absolute Error (single observation).

    Returns
    -------
    float
        ``|y_true - y_pred|``
    """
    return abs(y_true - y_pred)


def smape_loss(y_true: float, y_pred: float) -> float:
    """Symmetric Mean Absolute Percentage Error (single observation).

    Defined as ``|y_true - y_pred| / ((|y_true| + |y_pred|) / 2 + eps)``.
    Returns a value in [0, 2]. A small epsilon avoids division by zero
    when both values are zero.

    Returns
    -------
    float
        sMAPE for a single observation.
    """
    numerator = abs(y_true - y_pred)
    denominator = (abs(y_true) + abs(y_pred)) / 2.0 + 1e-8
    return numerator / denominator


def rmse_loss(y_true: float, y_pred: float) -> float:
    """Root Mean Squared Error (single observation).

    For a single time point this is equivalent to MAE, but is provided
    for interface consistency.

    Returns
    -------
    float
        ``sqrt((y_true - y_pred)**2) == |y_true - y_pred|``
    """
    return math.sqrt((y_true - y_pred) ** 2)
