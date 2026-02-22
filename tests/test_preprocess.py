"""前処理パイプラインのテスト."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import (
    clip_outliers,
    fill_missing,
    preprocess,
    resample_hourly,
)
from src.data.split import TimeSeriesSplit, split_temporal


# ---------- fixtures ----------


@pytest.fixture()
def df_15min() -> pd.DataFrame:
    """15 分間隔のテスト用 DataFrame (4 時間 = 16 行, 3 系列)."""
    idx = pd.date_range("2013-01-01", periods=16, freq="15min")
    data = {
        "A": np.arange(1, 17, dtype=float),
        "B": np.arange(101, 117, dtype=float),
        "C": np.arange(201, 217, dtype=float),
    }
    return pd.DataFrame(data, index=idx)


@pytest.fixture()
def df_hourly_with_nan() -> pd.DataFrame:
    """1 時間間隔で欠損を含むテスト用 DataFrame."""
    idx = pd.date_range("2013-01-01", periods=48, freq="1h")
    rng = np.random.default_rng(42)
    data = pd.DataFrame(
        rng.random((48, 3)) * 100,
        index=idx,
        columns=["A", "B", "C"],
    )
    # 欠損を挿入
    data.iloc[5:8, 0] = np.nan  # A: 3 連続欠損 (ffill で埋まる)
    data.iloc[10:40, 1] = np.nan  # B: 30 連続欠損 (ffill limit=24 超え → 0)
    return data


@pytest.fixture()
def df_full_year() -> pd.DataFrame:
    """2011-2014 の 1 時間間隔テスト用 DataFrame."""
    idx = pd.date_range("2011-01-01", "2014-12-31 23:00:00", freq="1h")
    rng = np.random.default_rng(0)
    data = pd.DataFrame(
        rng.random((len(idx), 2)) * 100,
        index=idx,
        columns=["X", "Y"],
    )
    return data


# ---------- 集約処理のテスト ----------


class TestResampleHourly:
    def test_sum_aggregation(self, df_15min: pd.DataFrame) -> None:
        result = resample_hourly(df_15min, method="sum")
        # 4 時間分 → 4 行
        assert len(result) == 4
        # 最初の 1 時間: A=[1,2,3,4] → sum=10
        assert result.iloc[0]["A"] == pytest.approx(10.0)
        # B=[101,102,103,104] → sum=410
        assert result.iloc[0]["B"] == pytest.approx(410.0)

    def test_mean_aggregation(self, df_15min: pd.DataFrame) -> None:
        result = resample_hourly(df_15min, method="mean")
        assert len(result) == 4
        # A=[1,2,3,4] → mean=2.5
        assert result.iloc[0]["A"] == pytest.approx(2.5)

    def test_invalid_method(self, df_15min: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="method"):
            resample_hourly(df_15min, method="median")  # type: ignore[arg-type]


# ---------- 欠損補完のテスト ----------


class TestFillMissing:
    def test_ffill_within_limit(self, df_hourly_with_nan: pd.DataFrame) -> None:
        result = fill_missing(df_hourly_with_nan)
        # A の欠損 (3 連続) は ffill で全て埋まる
        assert not result["A"].isna().any()
        # ffill で埋まった部分は直前の値と同じ
        assert result.iloc[5]["A"] == df_hourly_with_nan.iloc[4]["A"]
        assert result.iloc[7]["A"] == df_hourly_with_nan.iloc[4]["A"]

    def test_ffill_exceeds_limit_fills_zero(
        self, df_hourly_with_nan: pd.DataFrame
    ) -> None:
        result = fill_missing(df_hourly_with_nan)
        # B の欠損 30 連続: 最初の 24 個は ffill、残り 6 個は 0
        assert not result["B"].isna().any()
        # index=34 は ffill limit 超え → 0
        assert result.iloc[34]["B"] == 0.0

    def test_no_nan_remains(self, df_hourly_with_nan: pd.DataFrame) -> None:
        result = fill_missing(df_hourly_with_nan)
        assert result.isna().sum().sum() == 0

    def test_reproducibility(self, df_hourly_with_nan: pd.DataFrame) -> None:
        r1 = fill_missing(df_hourly_with_nan.copy())
        r2 = fill_missing(df_hourly_with_nan.copy())
        pd.testing.assert_frame_equal(r1, r2)


# ---------- 外れ値クリップのテスト ----------


class TestClipOutliers:
    def test_values_within_range(self) -> None:
        idx = pd.date_range("2013-01-01", periods=1000, freq="1h")
        rng = np.random.default_rng(99)
        df = pd.DataFrame(
            rng.normal(50, 20, (1000, 2)),
            index=idx,
            columns=["A", "B"],
        )
        result = clip_outliers(df, lower_pct=1.0, upper_pct=99.0)
        for col in result.columns:
            lo = df[col].quantile(0.01)
            hi = df[col].quantile(0.99)
            assert result[col].min() >= lo - 1e-10
            assert result[col].max() <= hi + 1e-10


# ---------- 分割のテスト ----------


class TestSplitTemporal:
    def test_no_leak(self, df_full_year: pd.DataFrame) -> None:
        s = split_temporal(df_full_year)
        assert isinstance(s, TimeSeriesSplit)
        # Train の最後 < Valid の最初
        assert s.train.index.max() < s.valid.index.min()
        # Valid の最後 < Test の最初
        assert s.valid.index.max() < s.test.index.min()

    def test_no_overlap(self, df_full_year: pd.DataFrame) -> None:
        s = split_temporal(df_full_year)
        train_idx = set(s.train.index)
        valid_idx = set(s.valid.index)
        test_idx = set(s.test.index)
        assert train_idx.isdisjoint(valid_idx)
        assert valid_idx.isdisjoint(test_idx)
        assert train_idx.isdisjoint(test_idx)

    def test_covers_full_range(self, df_full_year: pd.DataFrame) -> None:
        s = split_temporal(df_full_year)
        total = len(s.train) + len(s.valid) + len(s.test)
        assert total == len(df_full_year)

    def test_train_date_range(self, df_full_year: pd.DataFrame) -> None:
        s = split_temporal(df_full_year)
        assert s.train.index.min() == pd.Timestamp("2011-01-01 00:00:00")
        assert s.train.index.max() == pd.Timestamp("2013-12-31 23:00:00")

    def test_valid_date_range(self, df_full_year: pd.DataFrame) -> None:
        s = split_temporal(df_full_year)
        assert s.valid.index.min() == pd.Timestamp("2014-01-01 00:00:00")
        assert s.valid.index.max() == pd.Timestamp("2014-06-30 23:00:00")

    def test_test_date_range(self, df_full_year: pd.DataFrame) -> None:
        s = split_temporal(df_full_year)
        assert s.test.index.min() == pd.Timestamp("2014-07-01 00:00:00")
        assert s.test.index.max() == pd.Timestamp("2014-12-31 23:00:00")


# ---------- パイプライン統合テスト ----------


class TestPreprocessPipeline:
    def test_pipeline_basic(self, df_15min: pd.DataFrame) -> None:
        result = preprocess(df_15min)
        assert len(result) == 4
        assert not result.isna().any().any()

    def test_pipeline_with_clip(self, df_15min: pd.DataFrame) -> None:
        result = preprocess(df_15min, clip=True)
        assert len(result) == 4
