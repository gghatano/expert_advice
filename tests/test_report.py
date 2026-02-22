"""Tests for report generation: spec compliance checks for generate_report() outputs."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.report import generate_report


# ---------------------------------------------------------------------------
# Helpers: synthetic data
# ---------------------------------------------------------------------------


def _make_result_df() -> pd.DataFrame:
    """generate_reportに渡せる形式の合成結果DataFrameを作成."""
    rng = np.random.RandomState(42)
    rows = []
    series_list = ["MT_001", "MT_002", "MT_003"]
    phases = ["valid", "test"]
    n_per_phase = 50

    for series in series_list:
        for phase in phases:
            base_ts = pd.Timestamp("2014-01-01") if phase == "valid" else pd.Timestamp("2014-07-01")
            for i in range(n_per_phase):
                y_true = 100.0 + rng.randn() * 20
                noise_hedge = rng.randn() * 5
                noise_equal = rng.randn() * 8
                noise_best = rng.randn() * 6
                y_pred = y_true + noise_hedge
                y_pred_equal = y_true + noise_equal
                y_pred_best = y_true + noise_best
                rows.append(
                    {
                        "timestamp": base_ts + pd.Timedelta(hours=i),
                        "y_true": y_true,
                        "y_pred": y_pred,
                        "y_pred_equal": y_pred_equal,
                        "y_pred_best": y_pred_best,
                        "loss_raw": abs(y_true - y_pred),
                        "loss_equal": abs(y_true - y_pred_equal),
                        "loss_best": abs(y_true - y_pred_best),
                        "series": series,
                        "phase": phase,
                    }
                )
    return pd.DataFrame(rows)


def _make_expert_names() -> list[str]:
    """合成Expert名リスト."""
    return [f"Expert_{i}" for i in range(5)]


def _make_expert_stats(with_meta_eta: bool = False) -> dict:
    """合成expert_stats (weight_snapshots含む)."""
    n_experts = 5
    total_steps = 100
    rng = np.random.RandomState(123)

    cum_losses = rng.rand(n_experts) * total_steps * 0.5
    avg_weights = rng.dirichlet(np.ones(n_experts))
    win_counts = rng.randint(0, total_steps, size=n_experts).astype(float)

    # weight_snapshots
    snapshots = []
    for step in range(0, total_steps, 10):
        top_weights = []
        for idx in range(n_experts):
            top_weights.append({"expert_idx": idx, "weight": float(rng.rand())})
        snap: dict = {"step": step, "top_weights": top_weights}
        if with_meta_eta:
            etas = [0.0001, 0.001, 0.01, 0.1]
            snap["eta_weights"] = list(rng.dirichlet(np.ones(len(etas))))
            snap["etas"] = etas
        snapshots.append(snap)

    stats: dict = {
        "expert_cum_losses": cum_losses,
        "expert_avg_weights": avg_weights,
        "expert_win_counts": win_counts,
        "total_steps": total_steps,
        "weight_snapshots": snapshots,
        "is_meta_eta": with_meta_eta,
    }
    return stats


def _make_config() -> dict:
    return {
        "run_id": "test-run-001",
        "experts": "light30",
        "eta_mode": "meta_grid",
        "scale_loss": "by_train_mae",
        "agg": "hourly_sum",
        "series_sample": 3,
        "seed": 42,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def report_output(tmp_path: Path) -> Path:
    """generate_report() を実行し、出力先ディレクトリを返す."""
    result_df = _make_result_df()
    config = _make_config()
    expert_names = _make_expert_names()
    generate_report(
        report_dir=tmp_path,
        result_df=result_df,
        config=config,
        expert_names=expert_names,
    )
    return tmp_path


@pytest.fixture()
def report_output_with_stats(tmp_path: Path) -> Path:
    """expert_stats付きでgenerate_report()を実行し、出力先ディレクトリを返す."""
    result_df = _make_result_df()
    config = _make_config()
    expert_names = _make_expert_names()
    expert_stats = _make_expert_stats(with_meta_eta=False)
    generate_report(
        report_dir=tmp_path,
        result_df=result_df,
        config=config,
        expert_names=expert_names,
        expert_stats=expert_stats,
    )
    return tmp_path


@pytest.fixture()
def report_output_meta_eta(tmp_path: Path) -> Path:
    """meta_eta付きexpert_statsでgenerate_report()を実行."""
    result_df = _make_result_df()
    config = _make_config()
    expert_names = _make_expert_names()
    expert_stats = _make_expert_stats(with_meta_eta=True)
    generate_report(
        report_dir=tmp_path,
        result_df=result_df,
        config=config,
        expert_names=expert_names,
        expert_stats=expert_stats,
    )
    return tmp_path


# =====================================================================
# 2. ファイル存在テスト
# =====================================================================


class TestFileExistence:
    """generate_report()呼び出し後に必要なファイルが存在すること."""

    @pytest.mark.parametrize(
        "filename",
        [
            "metrics.json",
            "experts_rank.csv",
            "summary.md",
            "method_comparison.png",
            "timeseries_comparison.png",
            "cumulative_loss.png",
            "series_mae_scatter.png",
        ],
    )
    def test_required_files_exist(self, report_output: Path, filename: str) -> None:
        assert (report_output / filename).exists(), f"{filename} が生成されていない"


# =====================================================================
# 3. metrics.json の内容テスト
# =====================================================================


class TestMetricsJson:
    """metrics.json の仕様準拠チェック."""

    def test_parseable_as_json(self, report_output: Path) -> None:
        path = report_output / "metrics.json"
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_contains_valid_and_test_keys(self, report_output: Path) -> None:
        with open(report_output / "metrics.json", encoding="utf-8") as f:
            data = json.load(f)
        assert "valid" in data, "metrics.json に 'valid' キーがない"
        assert "test" in data, "metrics.json に 'test' キーがない"

    @pytest.mark.parametrize("phase", ["valid", "test"])
    def test_phase_contains_required_mae_keys(self, report_output: Path, phase: str) -> None:
        with open(report_output / "metrics.json", encoding="utf-8") as f:
            data = json.load(f)
        required = {"hedge_mae", "equal_mae", "best_mae"}
        actual_keys = set(data[phase].keys())
        assert required.issubset(actual_keys), f"{phase} に {required - actual_keys} が不足"

    @pytest.mark.parametrize("phase", ["valid", "test"])
    def test_mae_values_are_nonneg_floats(self, report_output: Path, phase: str) -> None:
        with open(report_output / "metrics.json", encoding="utf-8") as f:
            data = json.load(f)
        for key in ["hedge_mae", "equal_mae", "best_mae"]:
            val = data[phase][key]
            assert isinstance(val, (int, float)), f"{phase}.{key} が数値でない: {type(val)}"
            assert val >= 0, f"{phase}.{key} が負: {val}"
            assert not math.isnan(val), f"{phase}.{key} が NaN"


# =====================================================================
# 4. experts_rank.csv の内容テスト
# =====================================================================


class TestExpertsRankCsv:
    """experts_rank.csv の仕様準拠チェック."""

    def test_parseable_as_csv(self, report_output: Path) -> None:
        df = pd.read_csv(report_output / "experts_rank.csv")
        assert len(df) > 0

    def test_required_columns(self, report_output: Path) -> None:
        df = pd.read_csv(report_output / "experts_rank.csv")
        required = {"rank", "expert_name", "avg_loss", "avg_weight", "win_rate"}
        actual = set(df.columns)
        assert required.issubset(actual), f"不足カラム: {required - actual}"

    def test_rank_is_one_based_sequence(self, report_output: Path) -> None:
        df = pd.read_csv(report_output / "experts_rank.csv")
        expected = list(range(1, len(df) + 1))
        assert df["rank"].tolist() == expected, "rankが1始まりの連番でない"

    def test_stats_not_nan_when_expert_stats_provided(self, report_output_with_stats: Path) -> None:
        """expert_statsを渡した場合、avg_loss等がNaNでない."""
        df = pd.read_csv(report_output_with_stats / "experts_rank.csv")
        assert not df["avg_loss"].isna().any(), "avg_loss に NaN がある"
        assert not df["avg_weight"].isna().any(), "avg_weight に NaN がある"
        assert not df["win_rate"].isna().any(), "win_rate に NaN がある"


# =====================================================================
# 5. summary.md の内容テスト
# =====================================================================


class TestSummaryMd:
    """summary.md の仕様準拠チェック."""

    def test_not_empty(self, report_output: Path) -> None:
        content = (report_output / "summary.md").read_text(encoding="utf-8")
        assert len(content.strip()) > 0, "summary.md が空"

    def test_contains_method_comparison_section(self, report_output: Path) -> None:
        content = (report_output / "summary.md").read_text(encoding="utf-8")
        assert "手法別比較結果" in content, "'手法別比較結果' セクションがない"

    def test_contains_plot_references(self, report_output: Path) -> None:
        content = (report_output / "summary.md").read_text(encoding="utf-8")
        assert "![" in content, "プロットへの参照 (![...]) がない"


# =====================================================================
# 6. weights_plot.png / eta_plot.png のテスト
# =====================================================================


class TestOptionalPlots:
    """expert_stats依存のオプショナルプロット生成テスト."""

    def test_weights_plot_generated_with_expert_stats(self, report_output_with_stats: Path) -> None:
        """expert_statsを渡した場合にweights_plot.pngが生成される."""
        assert (report_output_with_stats / "weights_plot.png").exists(), \
            "expert_stats付きなのに weights_plot.png が生成されていない"

    def test_eta_plot_generated_with_meta_eta(self, report_output_meta_eta: Path) -> None:
        """is_meta_eta=True + eta情報を含む場合にeta_plot.pngが生成される."""
        assert (report_output_meta_eta / "eta_plot.png").exists(), \
            "is_meta_eta=True なのに eta_plot.png が生成されていない"

    def test_eta_plot_not_generated_without_meta_eta(self, report_output_with_stats: Path) -> None:
        """is_meta_eta=Falseの場合はeta_plot.pngが生成されない."""
        assert not (report_output_with_stats / "eta_plot.png").exists(), \
            "is_meta_eta=False なのに eta_plot.png が生成されている"
