"""前処理パイプラインのテスト."""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.load_uci import _detect_format, _resolve_csv_path, load_electricity
from src.data.preprocess import (
    clip_outliers,
    drop_high_missing,
    fill_missing,
    load_processed,
    preprocess,
    resample_hourly,
    save_processed,
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

    def test_pipeline_with_drop_missing(self) -> None:
        """drop_missing=True で欠損率の高い系列が除外される."""
        idx = pd.date_range("2013-01-01", periods=16, freq="15min")
        data = pd.DataFrame(
            {
                "good": np.arange(16, dtype=float),
                "bad": [np.nan] * 10 + list(range(6)),  # 欠損率 62.5%
            },
            index=idx,
        )
        result = preprocess(data, drop_missing=True, missing_threshold=0.05)
        # "bad" は除外され "good" のみ残る
        assert "good" in result.columns
        assert "bad" not in result.columns


# ---------- load_uci のテスト ----------


class TestLoadElectricity:
    """load_uci.py の各機能のテスト."""

    def _make_csv(
        self,
        tmp_path: Path,
        content: str,
        filename: str = "data.csv",
    ) -> Path:
        """ヘルパー: tmp_path にCSVファイルを書き出して返す."""
        p = tmp_path / filename
        p.write_text(textwrap.dedent(content), encoding="utf-8")
        return p

    def test_comma_separated_csv(self, tmp_path: Path) -> None:
        """カンマ区切りCSVの読み込み."""
        csv_path = self._make_csv(
            tmp_path,
            """\
            datetime,A,B
            2013-01-01 00:00:00,1.0,2.0
            2013-01-01 00:15:00,3.0,4.0
            """,
        )
        df = load_electricity(path=csv_path)
        assert df.shape == (2, 2)
        assert list(df.columns) == ["A", "B"]
        assert df.iloc[0]["A"] == pytest.approx(1.0)

    def test_semicolon_separated_csv(self, tmp_path: Path) -> None:
        """セミコロン区切りCSVの読み込み."""
        csv_path = self._make_csv(
            tmp_path,
            """\
            datetime;A;B
            01/01/2013 00:00:00;1,5;2,5
            01/01/2013 00:15:00;3,5;4,5
            """,
        )
        df = load_electricity(path=csv_path)
        assert df.shape == (2, 2)
        # 小数点カンマ: 1,5 → 1.5
        assert df.iloc[0]["A"] == pytest.approx(1.5)
        assert df.iloc[0]["B"] == pytest.approx(2.5)

    def test_european_decimal_detection(self, tmp_path: Path) -> None:
        """ヨーロッパ形式（セミコロン区切り・小数点カンマ）の自動検出."""
        csv_path = self._make_csv(
            tmp_path,
            """\
            datetime;X;Y
            01/01/2013 00:00:00;10,25;20,75
            01/01/2013 00:15:00;30,50;40,00
            """,
        )
        sep, decimal = _detect_format(csv_path)
        assert sep == ";"
        assert decimal == ","

    def test_comma_format_detection(self, tmp_path: Path) -> None:
        """カンマ区切り形式の自動検出."""
        csv_path = self._make_csv(
            tmp_path,
            """\
            datetime,X,Y
            2013-01-01 00:00:00,10.25,20.75
            """,
        )
        sep, decimal = _detect_format(csv_path)
        assert sep == ","
        assert decimal == "."

    def test_trailing_semicolon_empty_column_removed(self, tmp_path: Path) -> None:
        """末尾セミコロンによる空カラムが除去される."""
        csv_path = self._make_csv(
            tmp_path,
            """\
            datetime;A;B;
            01/01/2013 00:00:00;1,0;2,0;
            01/01/2013 00:15:00;3,0;4,0;
            """,
        )
        df = load_electricity(path=csv_path)
        # 空カラム "" は除去されている
        assert "" not in df.columns
        assert len(df.columns) == 2

    def test_unnamed_column_removed(self, tmp_path: Path) -> None:
        """Unnamed カラムが除去される."""
        # pandas が自動生成する "Unnamed: N" カラムをシミュレート
        csv_path = self._make_csv(
            tmp_path,
            """\
            datetime,A,Unnamed: 2
            2013-01-01 00:00:00,1.0,99.0
            2013-01-01 00:15:00,3.0,99.0
            """,
        )
        df = load_electricity(path=csv_path)
        assert not any(c.startswith("Unnamed") for c in df.columns)
        assert "A" in df.columns

    def test_directory_auto_detect(self, tmp_path: Path) -> None:
        """ディレクトリ指定でCSVを自動検出する."""
        self._make_csv(
            tmp_path,
            """\
            datetime,A,B
            2013-01-01 00:00:00,1.0,2.0
            2013-01-01 00:15:00,3.0,4.0
            """,
            filename="electricity.csv",
        )
        df = load_electricity(raw_dir=tmp_path)
        assert df.shape == (2, 2)

    def test_file_not_found_direct_path(self, tmp_path: Path) -> None:
        """存在しないファイルパスを指定した場合 FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_electricity(path=tmp_path / "nonexistent.csv")

    def test_file_not_found_empty_directory(self, tmp_path: Path) -> None:
        """CSVがないディレクトリを指定した場合 FileNotFoundError."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            load_electricity(raw_dir=empty_dir)


# ---------- drop_high_missing / Parquet キャッシュのテスト ----------


class TestDropHighMissing:
    def test_drops_high_missing_columns(self) -> None:
        """欠損率が閾値を超える系列が除外される."""
        idx = pd.date_range("2013-01-01", periods=100, freq="1h")
        df = pd.DataFrame(
            {
                "good": np.arange(100, dtype=float),
                "bad": [np.nan] * 20 + list(range(80)),  # 欠損率 20%
            },
            index=idx,
        )
        result = drop_high_missing(df, threshold=0.10)
        assert "good" in result.columns
        assert "bad" not in result.columns

    def test_keeps_low_missing_columns(self) -> None:
        """欠損率が閾値以下の系列は残る."""
        idx = pd.date_range("2013-01-01", periods=100, freq="1h")
        df = pd.DataFrame(
            {
                "A": [np.nan] * 3 + list(range(97)),  # 欠損率 3%
                "B": np.arange(100, dtype=float),      # 欠損率 0%
            },
            index=idx,
        )
        result = drop_high_missing(df, threshold=0.05)
        assert "A" in result.columns
        assert "B" in result.columns


class TestParquetCache:
    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        """save_processed → load_processed で同じデータが復元される."""
        idx = pd.date_range("2013-01-01", periods=24, freq="1h")
        df = pd.DataFrame(
            {"A": np.arange(24, dtype=float), "B": np.arange(100, 124, dtype=float)},
            index=idx,
        )
        df.index.name = "datetime"

        saved_path = save_processed(df, name="test_cache.parquet", processed_dir=tmp_path)
        assert saved_path.exists()

        loaded = load_processed(name="test_cache.parquet", processed_dir=tmp_path)
        pd.testing.assert_frame_equal(df, loaded, check_freq=False)

    def test_load_processed_file_not_found(self, tmp_path: Path) -> None:
        """キャッシュが存在しない場合 FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_processed(name="no_such_file.parquet", processed_dir=tmp_path)
