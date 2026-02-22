"""Loss scaling utilities.

These helpers normalise raw loss values so that experts evaluated on
time series of different scales can be compared fairly.
"""

from __future__ import annotations


def by_train_mae(loss: float, train_mae: float) -> float:
    """Scale *loss* by dividing by the series' training MAE.

    Parameters
    ----------
    loss : float
        Raw loss value (non-negative).
    train_mae : float
        Mean Absolute Error computed on the training set for the same
        series.  Must be positive.

    Returns
    -------
    float
        ``loss / train_mae``
    """
    if train_mae <= 0:
        raise ValueError("train_mae must be positive")
    return loss / train_mae


def relative(loss: float, y_true: float) -> float:
    """Scale *loss* relative to the true value.

    ``loss / (|y_true| + eps)`` where ``eps = 1e-6`` guards against
    division by zero.

    Parameters
    ----------
    loss : float
        Raw loss value (non-negative).
    y_true : float
        Actual observed value.

    Returns
    -------
    float
        Relative loss.
    """
    return loss / (abs(y_true) + 1e-6)
