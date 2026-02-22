"""Base class for all Experts."""

from abc import ABC, abstractmethod

import pandas as pd


class BaseExpert(ABC):
    """Abstract base class that every Expert must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return a unique, human-readable identifier including parameters."""
        ...

    @abstractmethod
    def fit(self, history: pd.Series, **kwargs) -> None:
        """Optionally learn from *history* (a time-indexed Series of floats).

        Naive experts may ignore this; statistical / ML experts will use it.
        """
        ...

    @abstractmethod
    def predict_next(
        self,
        history: pd.Series,
        tstamp: pd.Timestamp,
        exog: dict | None = None,
    ) -> float:
        """Return a point forecast for the next time step.

        Parameters
        ----------
        history : pd.Series
            Observed values up to (but not including) *tstamp*.
        tstamp : pd.Timestamp
            The timestamp we are predicting for.
        exog : dict | None
            Optional exogenous features (e.g. temperature).

        Returns
        -------
        float
            The predicted value.  Must never be NaN.
        """
        ...
