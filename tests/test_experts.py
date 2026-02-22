"""Tests for Expert base class and Naive family."""

import math

import numpy as np
import pandas as pd
import pytest

from src.experts.base import BaseExpert
from src.experts.moving_avg import SMA, Median
from src.experts.naive import Drift, LastValue, SeasonalNaive
from src.experts.regression import HuberRegressorLag, KNNLag, RidgeLag
from src.experts.seasonal_profile import STLSeasonalMean
from src.experts.smoothing import EMA


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
    SMA(24),
    SMA(168),
    Median(24),
    Median(168),
    EMA(0.1),
    EMA(0.3),
    STLSeasonalMean(),
    RidgeLag(0.1),
    RidgeLag(1.0),
    HuberRegressorLag(),
    KNNLag(5),
    KNNLag(3),
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
# Moving Average / Smoothing tests
# ---------------------------------------------------------------------------

class TestSMA:
    def test_returns_mean_of_window(self):
        """SMA of constant series should return that constant."""
        idx = pd.date_range("2024-01-01", periods=10, freq="h")
        series = pd.Series([5.0] * 10, index=idx)
        expert = SMA(window=5)
        ts = idx[-1] + pd.Timedelta(hours=1)
        assert expert.predict_next(series, ts) == 5.0

    def test_known_average(self):
        idx = pd.date_range("2024-01-01", periods=4, freq="h")
        series = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx)
        expert = SMA(window=4)
        ts = idx[-1] + pd.Timedelta(hours=1)
        assert expert.predict_next(series, ts) == 2.5

    def test_window_smaller_than_history(self):
        idx = pd.date_range("2024-01-01", periods=6, freq="h")
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], index=idx)
        expert = SMA(window=3)
        ts = idx[-1] + pd.Timedelta(hours=1)
        # Mean of last 3: (4+5+6)/3 = 5.0
        assert expert.predict_next(series, ts) == 5.0

    def test_different_windows(self):
        a = SMA(24)
        b = SMA(48)
        assert a.name != b.name


class TestMedian:
    def test_returns_median_of_window(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="h")
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx)
        expert = Median(window=5)
        ts = idx[-1] + pd.Timedelta(hours=1)
        assert expert.predict_next(series, ts) == 3.0

    def test_even_window(self):
        idx = pd.date_range("2024-01-01", periods=4, freq="h")
        series = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx)
        expert = Median(window=4)
        ts = idx[-1] + pd.Timedelta(hours=1)
        assert expert.predict_next(series, ts) == 2.5

    def test_different_windows(self):
        a = Median(24)
        b = Median(48)
        assert a.name != b.name


class TestEMA:
    def test_constant_series(self):
        """EMA of constant series should return that constant."""
        idx = pd.date_range("2024-01-01", periods=20, freq="h")
        series = pd.Series([7.0] * 20, index=idx)
        expert = EMA(alpha=0.3)
        ts = idx[-1] + pd.Timedelta(hours=1)
        assert abs(expert.predict_next(series, ts) - 7.0) < 1e-9

    def test_high_alpha_tracks_recent(self):
        """With alpha close to 1.0, EMA should be close to the last value."""
        idx = pd.date_range("2024-01-01", periods=100, freq="h")
        rng = np.random.default_rng(99)
        values = rng.normal(50, 5, 100)
        series = pd.Series(values, index=idx)
        expert = EMA(alpha=0.99)
        ts = idx[-1] + pd.Timedelta(hours=1)
        result = expert.predict_next(series, ts)
        assert abs(result - float(values[-1])) < 1.0

    def test_different_alphas(self):
        a = EMA(0.1)
        b = EMA(0.3)
        assert a.name != b.name

    def test_single_observation(self):
        idx = pd.date_range("2024-01-01", periods=1, freq="h")
        series = pd.Series([42.0], index=idx)
        expert = EMA(alpha=0.5)
        ts = idx[-1] + pd.Timedelta(hours=1)
        assert expert.predict_next(series, ts) == 42.0


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
            SMA(24),
            SMA(48),
            Median(24),
            Median(48),
            EMA(0.1),
            EMA(0.3),
            RidgeLag(0.1),
            RidgeLag(1.0),
            HuberRegressorLag(),
            KNNLag(5),
            KNNLag(3),
        ]
        names = [e.name for e in experts]
        assert len(names) == len(set(names)), f"Duplicate names: {names}"


# ---------------------------------------------------------------------------
# Regression Experts
# ---------------------------------------------------------------------------

REGRESSION_EXPERTS: list[BaseExpert] = [
    RidgeLag(0.1),
    RidgeLag(1.0),
    HuberRegressorLag(),
    KNNLag(5),
    KNNLag(3),
]


class TestRegressionInterface:
    """Regression experts must satisfy the BaseExpert contract."""

    @pytest.mark.parametrize("expert", REGRESSION_EXPERTS, ids=lambda e: e.name)
    def test_is_base_expert(self, expert: BaseExpert):
        assert isinstance(expert, BaseExpert)

    @pytest.mark.parametrize("expert", REGRESSION_EXPERTS, ids=lambda e: e.name)
    def test_name_is_str(self, expert: BaseExpert):
        assert isinstance(expert.name, str)
        assert len(expert.name) > 0


class TestRegressionFitPredict:
    """Fit on hourly data, then predict — basic round-trip."""

    @pytest.mark.parametrize("expert", REGRESSION_EXPERTS, ids=lambda e: e.name)
    def test_fit_then_predict(self, expert: BaseExpert, hourly_history: pd.Series):
        expert.fit(hourly_history)
        ts = hourly_history.index[-1] + pd.Timedelta(hours=1)
        result = expert.predict_next(hourly_history, ts)
        assert isinstance(result, float)
        assert not math.isnan(result)

    @pytest.mark.parametrize("expert", REGRESSION_EXPERTS, ids=lambda e: e.name)
    def test_prediction_in_reasonable_range(self, expert: BaseExpert, hourly_history: pd.Series):
        """After fitting, prediction should be in a plausible range (not wildly off)."""
        expert.fit(hourly_history)
        ts = hourly_history.index[-1] + pd.Timedelta(hours=1)
        result = expert.predict_next(hourly_history, ts)
        # Data is ~100 +/- 20, so prediction should be within [0, 300]
        assert 0 < result < 300


class TestRegressionFallback:
    """Regression experts must handle insufficient data gracefully."""

    @pytest.mark.parametrize("expert", REGRESSION_EXPERTS, ids=lambda e: e.name)
    def test_predict_without_fit(self, expert: BaseExpert, hourly_history: pd.Series):
        """Predicting without prior fit should not crash."""
        ts = hourly_history.index[-1] + pd.Timedelta(hours=1)
        result = expert.predict_next(hourly_history, ts)
        assert isinstance(result, float)
        assert not math.isnan(result)

    @pytest.mark.parametrize("expert", REGRESSION_EXPERTS, ids=lambda e: e.name)
    def test_short_history_fallback(self, expert: BaseExpert, short_history: pd.Series):
        """With only 3 data points, fit + predict should still return a valid float."""
        expert.fit(short_history)
        ts = short_history.index[-1] + pd.Timedelta(hours=1)
        result = expert.predict_next(short_history, ts)
        assert isinstance(result, float)
        assert not math.isnan(result)

    @pytest.mark.parametrize("expert", REGRESSION_EXPERTS, ids=lambda e: e.name)
    def test_empty_history_fallback(self, expert: BaseExpert, empty_history: pd.Series):
        expert.fit(empty_history)
        ts = pd.Timestamp("2024-01-01")
        result = expert.predict_next(empty_history, ts)
        assert isinstance(result, float)
        assert not math.isnan(result)


# ---------------------------------------------------------------------------
# STLSeasonalMean tests
# ---------------------------------------------------------------------------

class TestSTLSeasonalMean:
    """Tests for STLSeasonalMean seasonal profile expert."""

    def test_is_base_expert(self):
        expert = STLSeasonalMean()
        assert isinstance(expert, BaseExpert)

    def test_name(self):
        expert = STLSeasonalMean()
        assert expert.name == "STLSeasonalMean"

    def test_unfitted_returns_zero(self):
        """Before fit(), predict_next should return 0.0 as fallback."""
        expert = STLSeasonalMean()
        ts = pd.Timestamp("2024-03-15 10:00:00")  # Friday 10:00
        result = expert.predict_next(pd.Series(dtype=float), ts)
        assert result == 0.0

    def test_fit_empty_history_then_predict(self):
        """fit with empty history should still produce 0.0 fallback."""
        expert = STLSeasonalMean()
        expert.fit(pd.Series(dtype=float))
        ts = pd.Timestamp("2024-03-15 10:00:00")
        result = expert.predict_next(pd.Series(dtype=float), ts)
        assert result == 0.0

    def test_predict_matches_dow_hour(self, hourly_history: pd.Series):
        """After fit, predict_next should return the mean for the matching dow x hour."""
        expert = STLSeasonalMean()
        expert.fit(hourly_history)

        # Pick a specific timestamp and verify against manual calculation
        ts = pd.Timestamp("2024-01-08 05:00:00")  # Monday, hour 5
        dow, hour = ts.dayofweek, ts.hour

        # Manually compute expected mean for this dow x hour
        mask = (hourly_history.index.dayofweek == dow) & (hourly_history.index.hour == hour)
        expected = float(hourly_history[mask].mean())

        result = expert.predict_next(hourly_history, ts)
        assert abs(result - expected) < 1e-10

    def test_predict_different_slots_differ(self):
        """Different dow x hour slots should generally produce different values."""
        # Build 2 weeks of data with a clear daily pattern
        rng = np.random.default_rng(123)
        idx = pd.date_range("2024-01-01", periods=336, freq="h")  # 2 weeks
        hour = idx.hour
        values = 50.0 + 30.0 * np.sin(2 * np.pi * hour / 24) + rng.normal(0, 1, 336)
        history = pd.Series(values, index=idx)

        expert = STLSeasonalMean()
        expert.fit(history)

        # Predict for hour=6 vs hour=18 on same day -- should differ
        ts_morning = pd.Timestamp("2024-01-15 06:00:00")
        ts_evening = pd.Timestamp("2024-01-15 18:00:00")
        assert expert.predict_next(history, ts_morning) != expert.predict_next(history, ts_evening)

    def test_predict_returns_float_after_fit(self, hourly_history: pd.Series):
        expert = STLSeasonalMean()
        expert.fit(hourly_history)
        ts = hourly_history.index[-1] + pd.Timedelta(hours=1)
        result = expert.predict_next(hourly_history, ts)
        assert isinstance(result, float)
        assert not math.isnan(result)
