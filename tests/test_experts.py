"""Tests for Expert base class and Naive family."""

import math

import numpy as np
import pandas as pd
import pytest

from src.experts.base import BaseExpert
from src.experts.naive import Drift, LastValue, SeasonalNaive


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def hourly_history() -> pd.Series:
    """168 hours (1 week) of synthetic load data."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2024-01-01", periods=168, freq="h")
    # gentle daily pattern + noise
    hour = np.arange(168) % 24
    values = 100 + 20 * np.sin(2 * np.pi * hour / 24) + rng.normal(0, 2, 168)
    return pd.Series(values, index=idx)


@pytest.fixture()
def short_history() -> pd.Series:
    """Only 3 observations — used to test fallback behaviour."""
    idx = pd.date_range("2024-06-01", periods=3, freq="h")
    return pd.Series([10.0, 20.0, 30.0], index=idx)


@pytest.fixture()
def empty_history() -> pd.Series:
    return pd.Series(dtype=float)


EXPERTS_WITH_DEFAULTS: list[BaseExpert] = [
    LastValue(),
    SeasonalNaive(24),
    SeasonalNaive(168),
    Drift(24),
    Drift(168),
]


# ---------------------------------------------------------------------------
# Interface tests
# ---------------------------------------------------------------------------

class TestInterface:
    """Every Expert must satisfy the BaseExpert contract."""

    @pytest.mark.parametrize("expert", EXPERTS_WITH_DEFAULTS, ids=lambda e: e.name)
    def test_is_base_expert(self, expert: BaseExpert):
        assert isinstance(expert, BaseExpert)

    @pytest.mark.parametrize("expert", EXPERTS_WITH_DEFAULTS, ids=lambda e: e.name)
    def test_predict_returns_float(self, expert: BaseExpert, hourly_history: pd.Series):
        ts = hourly_history.index[-1] + pd.Timedelta(hours=1)
        result = expert.predict_next(hourly_history, ts)
        assert isinstance(result, float)

    @pytest.mark.parametrize("expert", EXPERTS_WITH_DEFAULTS, ids=lambda e: e.name)
    def test_predict_not_nan(self, expert: BaseExpert, hourly_history: pd.Series):
        ts = hourly_history.index[-1] + pd.Timedelta(hours=1)
        result = expert.predict_next(hourly_history, ts)
        assert not math.isnan(result)

    @pytest.mark.parametrize("expert", EXPERTS_WITH_DEFAULTS, ids=lambda e: e.name)
    def test_name_is_str(self, expert: BaseExpert):
        assert isinstance(expert.name, str)
        assert len(expert.name) > 0


# ---------------------------------------------------------------------------
# Fallback / edge-case tests
# ---------------------------------------------------------------------------

class TestFallback:
    """Experts must never return NaN, even with insufficient data."""

    @pytest.mark.parametrize("expert", EXPERTS_WITH_DEFAULTS, ids=lambda e: e.name)
    def test_short_history(self, expert: BaseExpert, short_history: pd.Series):
        ts = short_history.index[-1] + pd.Timedelta(hours=1)
        result = expert.predict_next(short_history, ts)
        assert isinstance(result, float)
        assert not math.isnan(result)

    @pytest.mark.parametrize("expert", EXPERTS_WITH_DEFAULTS, ids=lambda e: e.name)
    def test_empty_history(self, expert: BaseExpert, empty_history: pd.Series):
        ts = pd.Timestamp("2024-01-01")
        result = expert.predict_next(empty_history, ts)
        assert isinstance(result, float)
        assert not math.isnan(result)


# ---------------------------------------------------------------------------
# Semantic / sanity tests
# ---------------------------------------------------------------------------

class TestLastValue:
    def test_returns_last(self, hourly_history: pd.Series):
        expert = LastValue()
        ts = hourly_history.index[-1] + pd.Timedelta(hours=1)
        assert expert.predict_next(hourly_history, ts) == float(hourly_history.iloc[-1])


class TestSeasonalNaive:
    def test_returns_value_from_season_ago(self, hourly_history: pd.Series):
        expert = SeasonalNaive(24)
        ts = hourly_history.index[-1] + pd.Timedelta(hours=1)
        expected = float(hourly_history.iloc[-24])
        assert expert.predict_next(hourly_history, ts) == expected

    def test_different_season_lengths(self):
        a = SeasonalNaive(24)
        b = SeasonalNaive(48)
        assert a.name != b.name


class TestDrift:
    def test_linear_trend(self):
        """Perfect linear data should yield exact extrapolation."""
        idx = pd.date_range("2024-01-01", periods=48, freq="h")
        linear = pd.Series(np.arange(48, dtype=float), index=idx)
        expert = Drift(window=48)
        ts = idx[-1] + pd.Timedelta(hours=1)
        result = expert.predict_next(linear, ts)
        assert abs(result - 48.0) < 1e-6

    def test_different_windows(self):
        a = Drift(24)
        b = Drift(168)
        assert a.name != b.name


# ---------------------------------------------------------------------------
# Multiple instances
# ---------------------------------------------------------------------------

class TestMultipleInstances:
    def test_unique_names(self):
        experts = [
            LastValue(),
            SeasonalNaive(24),
            SeasonalNaive(48),
            SeasonalNaive(168),
            Drift(24),
            Drift(168),
        ]
        names = [e.name for e in experts]
        assert len(names) == len(set(names)), f"Duplicate names: {names}"
