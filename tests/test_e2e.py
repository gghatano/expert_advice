"""E2Eテスト・CLI引数境界値テスト.

合成データを使い、全パイプライン（前処理→分割→Expert生成→逐次予測→レポート）を通す。
実データ (UCI Electricity) は不要。
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import preprocess
from src.data.split import split_temporal
from src.ensemble.hedge import Hedge
from src.ensemble.meta_eta import MetaEtaHedge
from src.run_experiment import (
    _compute_train_mae_naive,
    _create_ensemble,
    _generate_run_id,
    _load_and_preprocess,
    _parse_etas,
    _select_series,
    _set_seed,
    build_parser,
    main,
)


# ---------------------------------------------------------------------------
# 合成データのヘルパー
# ---------------------------------------------------------------------------


def _make_synthetic_data() -> pd.DataFrame:
    """2011-2015年の15分間隔、5系列の合成データを生成.

    各系列は sin(hour) + noise のようなパターン。
    テスト用なので期間を短縮し、split_temporal のデフォルト境界に合わせる:
      train: 2011-01-01 ~ 2013-12-31
      valid: 2014-01-01 ~ 2014-06-30
      test:  2014-07-01 ~ 2014-12-31
    15分間隔で生成し、preprocess で1時間に集約される前提。
    """
    # 15分間隔のDatetimeIndex (2011-01-01 ~ 2014-12-31)
    idx = pd.date_range("2011-01-01", "2014-12-31 23:45:00", freq="15min")
    rng = np.random.RandomState(12345)

    data = {}
    for i in range(5):
        hour = idx.hour + idx.minute / 60.0
        # 基本パターン: sin(hour) ベースに系列ごとオフセット
        base = 100.0 + 50.0 * np.sin(2 * np.pi * hour / 24.0) + i * 20.0
        noise = rng.normal(0, 5.0, size=len(idx))
        data[f"series_{i:03d}"] = np.maximum(base + noise, 0.0)

    df = pd.DataFrame(data, index=idx)
    df.index.name = "datetime"
    return df


def _make_preprocessed_data() -> pd.DataFrame:
    """preprocess済み (1時間間隔) の合成データを返す."""
    raw = _make_synthetic_data()
    return preprocess(raw, resample_method="sum", clip=False)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def synthetic_hourly() -> pd.DataFrame:
    """preprocess済みの合成データを返すfixture."""
    return _make_preprocessed_data()


@pytest.fixture()
def tmp_project(tmp_path: Path) -> Path:
    """tmp_path をプロジェクトルートとして使うfixture.

    reports/ ディレクトリは main() が自動的に作成する。
    """
    return tmp_path


# ---------------------------------------------------------------------------
# 2. E2Eテスト: 全パイプライン通し
# ---------------------------------------------------------------------------


class TestE2EPipeline:
    """合成データで前処理→分割→Expert生成→逐次予測→レポート生成を通す."""

    def _run_main(
        self,
        tmp_project: Path,
        extra_args: list[str] | None = None,
    ) -> Path:
        """main() を実行し、生成されたレポートディレクトリを返す.

        _load_and_preprocess をモックして合成データを返すことで、
        実データ不要で全パイプラインを実行する。
        """
        hourly_df = _make_preprocessed_data()

        argv = [
            "--data-path", "dummy",
            "--experts", "light30",
            "--eta-mode", "fixed",
            "--etas", "0.1",
            "--series-sample", "2",
            "--seed", "42",
        ]
        if extra_args:
            argv.extend(extra_args)

        with (
            patch("src.run_experiment._load_and_preprocess", return_value=hourly_df),
            patch("src.run_experiment._PROJECT_ROOT", tmp_project),
        ):
            main(argv)

        # reports/ 配下の最新ディレクトリを返す
        report_dirs = sorted((tmp_project / "reports").iterdir())
        assert len(report_dirs) >= 1, "レポートディレクトリが生成されていません"
        return report_dirs[-1]

    def test_full_pipeline_produces_all_artifacts(self, tmp_project: Path) -> None:
        """全パイプラインを通し、必要な成果物がすべて生成されることを確認."""
        report_dir = self._run_main(tmp_project)

        # 必須ファイルの存在確認
        expected_files = [
            "experiment_config.json",
            "predictions.parquet",
            "summary.md",
            "metrics.json",
            "experts_rank.csv",
        ]
        for fname in expected_files:
            fpath = report_dir / fname
            assert fpath.exists(), f"{fname} が存在しません: {report_dir}"

        # experiment_config.json の内容確認
        with open(report_dir / "experiment_config.json") as f:
            config = json.load(f)
        assert config["experts"] == "light30"
        assert config["seed"] == 42

        # predictions.parquet の内容確認
        pred_df = pd.read_parquet(report_dir / "predictions.parquet")
        assert len(pred_df) > 0, "predictions.parquet が空です"
        required_cols = [
            "timestamp", "y_true", "y_pred", "y_pred_equal",
            "y_pred_best", "loss_raw", "series", "phase",
        ]
        for col in required_cols:
            assert col in pred_df.columns, f"列 {col} が predictions.parquet にありません"

        # phase が valid と test の両方を含むこと
        phases = set(pred_df["phase"].unique())
        assert "valid" in phases
        assert "test" in phases

        # metrics.json の内容確認
        with open(report_dir / "metrics.json") as f:
            metrics = json.load(f)
        assert "test" in metrics or "valid" in metrics

        # experts_rank.csv の内容確認
        rank_df = pd.read_csv(report_dir / "experts_rank.csv")
        assert len(rank_df) > 0

        # summary.md が空でないこと
        md_content = (report_dir / "summary.md").read_text(encoding="utf-8")
        assert len(md_content) > 100, "summary.md の内容が短すぎます"

    def test_meta_grid_mode(self, tmp_project: Path) -> None:
        """meta_grid モードでも正常に動作することを確認."""
        hourly_df = _make_preprocessed_data()

        argv = [
            "--data-path", "dummy",
            "--experts", "light30",
            "--eta-mode", "meta_grid",
            "--series-sample", "1",
            "--seed", "99",
        ]

        with (
            patch("src.run_experiment._load_and_preprocess", return_value=hourly_df),
            patch("src.run_experiment._PROJECT_ROOT", tmp_project),
        ):
            main(argv)

        report_dirs = sorted((tmp_project / "reports").iterdir())
        assert len(report_dirs) >= 1
        report_dir = report_dirs[-1]

        assert (report_dir / "predictions.parquet").exists()
        assert (report_dir / "metrics.json").exists()

        with open(report_dir / "experiment_config.json") as f:
            config = json.load(f)
        assert config["eta_mode"] == "meta_grid"


# ---------------------------------------------------------------------------
# 3. CLI引数の境界値テスト
# ---------------------------------------------------------------------------


class TestCLIBoundaryValues:
    """CLI引数の境界値テスト."""

    def test_clip_quantile_0999(self) -> None:
        """--clip-quantile 0.999: 0-1比率がパーセンタイルに正しく変換される."""
        parser = build_parser()
        args = parser.parse_args(["--clip-quantile", "0.999"])

        # _load_and_preprocess 内部ロジックを再現:
        # clip_quantile * 100 = 99.9 (upper_pct)
        # 100.0 - 99.9 = 0.1 (lower_pct)
        assert args.clip_quantile == pytest.approx(0.999)

        upper_pct = args.clip_quantile * 100
        lower_pct = 100.0 - upper_pct
        assert upper_pct == pytest.approx(99.9)
        assert lower_pct == pytest.approx(0.1)

        # 実際に preprocess で clip が動作することを確認
        raw = _make_synthetic_data()
        df = preprocess(
            raw,
            resample_method="sum",
            clip=True,
            clip_lower_pct=lower_pct,
            clip_upper_pct=upper_pct,
        )
        assert len(df) > 0
        # クリップ後は NaN がないこと
        assert df.isna().sum().sum() == 0

    def test_clip_quantile_05(self) -> None:
        """--clip-quantile 0.5: 極端なクリップも動作する."""
        parser = build_parser()
        args = parser.parse_args(["--clip-quantile", "0.5"])
        assert args.clip_quantile == pytest.approx(0.5)

        upper_pct = args.clip_quantile * 100  # 50.0
        lower_pct = 100.0 - upper_pct  # 50.0

        raw = _make_synthetic_data()
        df = preprocess(
            raw,
            resample_method="sum",
            clip=True,
            clip_lower_pct=lower_pct,
            clip_upper_pct=upper_pct,
        )
        assert len(df) > 0
        # upper_pct == lower_pct == 50 の場合、全値が中央値にクリップされる
        # 各系列の値が全て同じ (中央値) になっていること
        for col in df.columns:
            unique_vals = df[col].nunique()
            assert unique_vals == 1, (
                f"clip_quantile=0.5 では全値が中央値になるはず: {col} has {unique_vals} unique values"
            )

    def test_series_sample_1(self, tmp_path: Path) -> None:
        """--series-sample 1: 最小系列数で動作する."""
        hourly_df = _make_preprocessed_data()

        argv = [
            "--data-path", "dummy",
            "--experts", "light30",
            "--eta-mode", "fixed",
            "--etas", "0.1",
            "--series-sample", "1",
            "--seed", "42",
        ]

        with (
            patch("src.run_experiment._load_and_preprocess", return_value=hourly_df),
            patch("src.run_experiment._PROJECT_ROOT", tmp_path),
        ):
            main(argv)

        report_dirs = sorted((tmp_path / "reports").iterdir())
        report_dir = report_dirs[-1]

        pred_df = pd.read_parquet(report_dir / "predictions.parquet")
        # 1系列のみ処理されていること
        assert pred_df["series"].nunique() == 1

    def test_eta_mode_fixed(self, tmp_path: Path) -> None:
        """--eta-mode fixed --etas 0.1: 固定eta動作."""
        hourly_df = _make_preprocessed_data()

        argv = [
            "--data-path", "dummy",
            "--experts", "light30",
            "--eta-mode", "fixed",
            "--etas", "0.1",
            "--series-sample", "1",
            "--seed", "42",
        ]

        with (
            patch("src.run_experiment._load_and_preprocess", return_value=hourly_df),
            patch("src.run_experiment._PROJECT_ROOT", tmp_path),
        ):
            main(argv)

        report_dirs = sorted((tmp_path / "reports").iterdir())
        report_dir = report_dirs[-1]

        with open(report_dir / "experiment_config.json") as f:
            config = json.load(f)
        assert config["eta_mode"] == "fixed"
        assert config["etas"] == [0.1]

        # 正常に予測結果が生成されていること
        pred_df = pd.read_parquet(report_dir / "predictions.parquet")
        assert len(pred_df) > 0

    def test_invalid_experts_preset(self) -> None:
        """--experts unknown_preset でエラーになること."""
        parser = build_parser()
        with pytest.raises(SystemExit):
            # argparse は choices 外の値で SystemExit を発生させる
            parser.parse_args(["--experts", "unknown_preset"])

    def test_invalid_eta_mode(self) -> None:
        """--eta-mode unknown でエラーになること."""
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--eta-mode", "unknown"])


# ---------------------------------------------------------------------------
# 4. 再現性テスト
# ---------------------------------------------------------------------------


class TestReproducibility:
    """同じseedで2回実行して、予測値が一致することを確認."""

    def test_same_seed_same_predictions(self, tmp_path: Path) -> None:
        """同じseedで2回実行し、predictions.parquetの予測値が一致する."""
        hourly_df = _make_preprocessed_data()

        predictions = []
        for run_idx in range(2):
            run_dir = tmp_path / f"run_{run_idx}"
            run_dir.mkdir()

            argv = [
                "--data-path", "dummy",
                "--experts", "light30",
                "--eta-mode", "fixed",
                "--etas", "0.1",
                "--series-sample", "2",
                "--seed", "42",
            ]

            with (
                patch("src.run_experiment._load_and_preprocess", return_value=hourly_df),
                patch("src.run_experiment._PROJECT_ROOT", run_dir),
            ):
                main(argv)

            report_dirs = sorted((run_dir / "reports").iterdir())
            assert len(report_dirs) == 1
            pred_df = pd.read_parquet(report_dirs[0] / "predictions.parquet")
            predictions.append(pred_df)

        # 2回の実行結果を比較
        df1 = predictions[0].sort_values(["series", "timestamp"]).reset_index(drop=True)
        df2 = predictions[1].sort_values(["series", "timestamp"]).reset_index(drop=True)

        # 同じ系列が選択されていること
        assert set(df1["series"].unique()) == set(df2["series"].unique()), (
            "同じseedなのに異なる系列が選択されました"
        )

        # 行数が同じこと
        assert len(df1) == len(df2), "同じseedなのに行数が異なります"

        # 予測値が完全に一致すること
        np.testing.assert_array_almost_equal(
            df1["y_pred"].values,
            df2["y_pred"].values,
            decimal=10,
            err_msg="同じseedなのに予測値が異なります",
        )
        np.testing.assert_array_almost_equal(
            df1["y_pred_equal"].values,
            df2["y_pred_equal"].values,
            decimal=10,
            err_msg="同じseedなのに等重み予測値が異なります",
        )
        np.testing.assert_array_almost_equal(
            df1["y_true"].values,
            df2["y_true"].values,
            decimal=10,
            err_msg="同じseedなのにy_trueが異なります",
        )


# ---------------------------------------------------------------------------
# 5. ヘルパー関数のユニットテスト
# ---------------------------------------------------------------------------


class TestGenerateRunId:
    """_generate_run_id のテスト."""

    def test_same_args_produce_same_hash_suffix(self) -> None:
        """同じ引数で同じrun_idの後半ハッシュが生成されること."""
        parser = build_parser()
        args = parser.parse_args(["--seed", "42", "--experts", "light30"])

        run_id_1 = _generate_run_id(args)
        run_id_2 = _generate_run_id(args)

        # ハッシュ部分 (末尾6文字) が一致すること
        hash_1 = run_id_1.split("_")[-1]
        hash_2 = run_id_2.split("_")[-1]
        assert hash_1 == hash_2, (
            f"同じ引数なのにハッシュが異なります: {hash_1} != {hash_2}"
        )
        assert len(hash_1) == 6


class TestSetSeed:
    """_set_seed のテスト."""

    def test_seed_makes_random_reproducible(self) -> None:
        """seed固定後のrandomとnumpyの出力が再現可能なこと."""
        import random as stdlib_random

        _set_seed(123)
        rand_val_1 = stdlib_random.random()
        np_val_1 = np.random.rand()

        _set_seed(123)
        rand_val_2 = stdlib_random.random()
        np_val_2 = np.random.rand()

        assert rand_val_1 == rand_val_2
        assert np_val_1 == np_val_2


class TestParseEtas:
    """_parse_etas のテスト."""

    def test_none_returns_none(self) -> None:
        """None → None."""
        assert _parse_etas(None) is None

    def test_csv_string_returns_float_list(self) -> None:
        """"0.1,0.5,1.0" → [0.1, 0.5, 1.0]."""
        result = _parse_etas("0.1,0.5,1.0")
        assert result == pytest.approx([0.1, 0.5, 1.0])

    def test_empty_string_returns_empty_list(self) -> None:
        """"" → []."""
        assert _parse_etas("") == []


class TestComputeTrainMaeNaive:
    """_compute_train_mae_naive のテスト."""

    def test_normal_series(self) -> None:
        """正常な系列でMAEが正しい値を返す."""
        # series: [1, 3, 5] → y_true=[3,5], y_pred=[1,3] → |3-1|=2, |5-3|=2 → MAE=2.0
        series = pd.Series([1.0, 3.0, 5.0])
        assert _compute_train_mae_naive(series) == pytest.approx(2.0)

    def test_length_1_returns_fallback(self) -> None:
        """長さ1以下の系列で1.0を返す (fallback)."""
        assert _compute_train_mae_naive(pd.Series([5.0])) == 1.0
        assert _compute_train_mae_naive(pd.Series([], dtype=float)) == 1.0

    def test_constant_series_returns_fallback(self) -> None:
        """定数系列で1.0を返す (MAE=0のfallback)."""
        series = pd.Series([3.0, 3.0, 3.0, 3.0])
        assert _compute_train_mae_naive(series) == 1.0


class TestSelectSeries:
    """_select_series のテスト."""

    def test_n_series_selects_first_n(self) -> None:
        """n_series=3 で先頭3件."""
        cols = pd.Index([f"s{i}" for i in range(10)])
        result = _select_series(cols, n_series=3, series_sample=None)
        assert result == ["s0", "s1", "s2"]

    def test_series_sample_selects_random_k(self) -> None:
        """series_sample=2 でランダム2件."""
        import random as stdlib_random

        cols = pd.Index([f"s{i}" for i in range(10)])
        stdlib_random.seed(0)
        result = _select_series(cols, n_series=None, series_sample=2)
        assert len(result) == 2
        # 全て元の列名に含まれていること
        for s in result:
            assert s in cols

    def test_both_none_returns_all(self) -> None:
        """両方Noneで全件."""
        cols = pd.Index(["a", "b", "c"])
        result = _select_series(cols, n_series=None, series_sample=None)
        assert result == ["a", "b", "c"]


class TestCreateEnsemble:
    """_create_ensemble のテスト."""

    def test_meta_grid_returns_meta_eta_hedge(self) -> None:
        """eta_mode='meta_grid' で MetaEtaHedge が返る."""
        ens = _create_ensemble(n_experts=5, eta_mode="meta_grid", etas=[0.1, 0.5])
        assert isinstance(ens, MetaEtaHedge)

    def test_fixed_returns_hedge(self) -> None:
        """eta_mode='fixed' で Hedge が返る."""
        ens = _create_ensemble(n_experts=5, eta_mode="fixed", etas=[0.2])
        assert isinstance(ens, Hedge)
        assert ens.eta == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# 6. scale-loss relative モードの E2E テスト
# ---------------------------------------------------------------------------


class TestScaleLossRelativeE2E:
    """scale-loss=relative モードのE2Eテスト."""

    def test_relative_mode_produces_valid_results(self, tmp_path: Path) -> None:
        """--scale-loss relative で正常に全パイプラインが通ること."""
        hourly_df = _make_preprocessed_data()

        argv = [
            "--data-path", "dummy",
            "--experts", "light30",
            "--eta-mode", "fixed",
            "--etas", "0.1",
            "--scale-loss", "relative",
            "--series-sample", "1",
            "--seed", "42",
        ]

        with (
            patch("src.run_experiment._load_and_preprocess", return_value=hourly_df),
            patch("src.run_experiment._PROJECT_ROOT", tmp_path),
        ):
            main(argv)

        report_dirs = sorted((tmp_path / "reports").iterdir())
        assert len(report_dirs) >= 1
        report_dir = report_dirs[-1]

        # 設定確認
        with open(report_dir / "experiment_config.json") as f:
            config = json.load(f)
        assert config["scale_loss"] == "relative"

        # predictions が生成されていること
        pred_df = pd.read_parquet(report_dir / "predictions.parquet")
        assert len(pred_df) > 0

        # 損失値が有限であること
        assert np.all(np.isfinite(pred_df["loss_raw"].values))
        assert np.all(pred_df["loss_raw"].values >= 0)
