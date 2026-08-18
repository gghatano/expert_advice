"""ベクトル化 Expert が逐次版と一致することを検証する.

``src.experts.vectorized`` は速度のために書き直した実装なので、既存の
``src.experts.*`` (仕様の正) と同じ値を返すことをここで担保する。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.experts.moving_avg import SMA, Median
from src.experts.naive import Drift, LastValue, SeasonalNaive
from src.experts.smoothing import EMA
from src.experts.vectorized import build_expert_matrix


@pytest.fixture
def series() -> pd.Series:
    """周期性 + トレンド + ノイズを持つ 300 点の時系列."""
    idx = pd.date_range("2020-01-01", periods=300, freq="h")
    rng = np.random.RandomState(42)
    t = np.arange(300)
    y = 100 + 20 * np.sin(2 * np.pi * t / 24) + 0.05 * t + rng.normal(0, 3, 300)
    return pd.Series(y, index=idx)


def _sequential(expert, s: pd.Series, start: int) -> np.ndarray:
    """逐次 API で t=start.. の 1 ステップ先予測を並べる."""
    out = []
    for t in range(start, len(s)):
        out.append(expert.predict_next(s.iloc[:t], s.index[t]))
    return np.asarray(out, dtype=np.float64)


def _vector_column(s: pd.Series, name: str, start: int) -> np.ndarray:
    names, P = build_expert_matrix(
        s.to_numpy(), index=s.index, period=24, week=168, preset="light30"
    )
    return P[start:, names.index(name)]


class TestVectorizedMatchesSequential:
    @pytest.mark.parametrize(
        "expert, name",
        [
            (LastValue(), "LastValue"),
            (SeasonalNaive(season_length=24), "SeasonalNaive_24"),
            (SeasonalNaive(season_length=48), "SeasonalNaive_48"),
            (SMA(window=24), "SMA_24"),
            (SMA(window=48), "SMA_48"),
            (Median(window=24), "Median_24"),
            (EMA(alpha=0.3), "EMA_0.3"),
            (EMA(alpha=0.05), "EMA_0.05"),
            (Drift(window=24), "Drift_24"),
            (Drift(window=48), "Drift_48"),
        ],
    )
    def test_matches(self, series: pd.Series, expert, name: str) -> None:
        # 履歴が十分たまった後 (168 点以降) を比較対象にする。
        start = 168
        seq = _sequential(expert, series, start)
        vec = _vector_column(series, name, start)
        np.testing.assert_allclose(vec, seq, rtol=1e-9, atol=1e-9)


class TestBuildExpertMatrix:
    def test_shape_and_names(self, series: pd.Series) -> None:
        names, P = build_expert_matrix(series.to_numpy(), index=series.index)
        assert P.shape == (len(series), len(names))
        assert len(set(names)) == len(names)

    def test_no_nan_or_inf(self, series: pd.Series) -> None:
        _, P = build_expert_matrix(series.to_numpy(), index=series.index)
        assert np.all(np.isfinite(P))

    def test_no_leakage(self) -> None:
        """未来の値を変えても過去時点の予測は変わらない (リークなし)."""
        rng = np.random.RandomState(1)
        y = rng.normal(100, 10, 400)
        cut = 300
        _, P_full = build_expert_matrix(y, period=24, week=168, train_end=200)
        y2 = y.copy()
        y2[cut:] += 500.0  # 未来だけ壊す
        _, P_mod = build_expert_matrix(y2, period=24, week=168, train_end=200)
        np.testing.assert_allclose(P_full[:cut], P_mod[:cut], rtol=1e-9, atol=1e-9)

    def test_horizon_1_is_the_default(self, series: pd.Series) -> None:
        _, a = build_expert_matrix(series.to_numpy(), index=series.index)
        _, b = build_expert_matrix(series.to_numpy(), index=series.index, horizon=1)
        np.testing.assert_allclose(a, b)

    @pytest.mark.parametrize("h", [1, 6, 24])
    def test_no_leakage_at_horizon(self, h: int) -> None:
        """P[t] は y[:t-h+1] にしか依存しない."""
        rng = np.random.RandomState(3)
        y = rng.normal(100, 10, 600)
        m = 400
        kw = dict(period=24, week=168, train_end=200, horizon=h)
        _, P = build_expert_matrix(y, **kw)
        y2 = y.copy()
        y2[m:] += 500.0
        _, P2 = build_expert_matrix(y2, **kw)
        np.testing.assert_allclose(P[: m + h], P2[: m + h], rtol=1e-9, atol=1e-9)

    def test_horizon_24_uses_a_day_old_observation(self) -> None:
        """h=24 の LastValue / SeasonalNaive_24 はともに y[t-24]."""
        y = np.arange(300, dtype=float)
        names, P = build_expert_matrix(y, period=24, week=168, horizon=24)
        np.testing.assert_allclose(P[100, names.index("LastValue")], y[76])
        np.testing.assert_allclose(P[100, names.index("SeasonalNaive_24")], y[76])

    def test_rejects_bad_horizon(self, series: pd.Series) -> None:
        with pytest.raises(ValueError):
            build_expert_matrix(series.to_numpy(), horizon=0)

    def test_core8_preset(self, series: pd.Series) -> None:
        names, P = build_expert_matrix(
            series.to_numpy(), index=series.index, preset="core8"
        )
        assert len(names) == 8
        assert P.shape[1] == 8

    def test_works_without_index(self) -> None:
        rng = np.random.RandomState(2)
        y = rng.normal(50, 5, 500)
        names, P = build_expert_matrix(y, index=None, period=24, week=168)
        assert np.all(np.isfinite(P))
        assert P.shape == (500, len(names))

    def test_rejects_bad_input(self) -> None:
        with pytest.raises(ValueError):
            build_expert_matrix(np.zeros((10, 2)))
        with pytest.raises(ValueError):
            build_expert_matrix(np.zeros(2))
        with pytest.raises(ValueError):
            build_expert_matrix(np.zeros(100), preset="nope")

    def test_ridge_beats_naive_on_predictable_series(self, series: pd.Series) -> None:
        """学習系 Expert が実際に機能していることの健全性チェック."""
        names, P = build_expert_matrix(
            series.to_numpy(), index=series.index, train_end=150
        )
        y = series.to_numpy()
        err = np.abs(y[168:, None] - P[168:])
        ridge = err[:, names.index("RidgeLag_1.0")].mean()
        last = err[:, names.index("LastValue")].mean()
        assert ridge < last
