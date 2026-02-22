"""レポート生成: 比較プロット・Markdownレポートの出力.

単一Expert予測、等重み平均アンサンブル、Hedge重み付きアンサンブルを
比較できる図表とMarkdownレポートを生成する。
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# プロット共通設定
# ---------------------------------------------------------------------------

_COLORS = {
    "actual": "#333333",
    "hedge": "#2196F3",
    "equal": "#FF9800",
    "best": "#4CAF50",
}

_LABELS = {
    "actual": "実測値 (Actual)",
    "hedge": "Hedge重み付き (Expert Advice)",
    "equal": "等重み平均 (Equal Weight)",
    "best": "ベストExpert (Best Single)",
}


def _setup_plot_style() -> None:
    """プロット共通スタイルを設定."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "font.size": 11,
        }
    )


# ---------------------------------------------------------------------------
# 指標計算
# ---------------------------------------------------------------------------


def _compute_metrics(df: pd.DataFrame) -> dict:
    """各手法のMAEとsMAPEを計算."""
    metrics = {}
    for phase in ["valid", "test"]:
        phase_df = df[df["phase"] == phase]
        if phase_df.empty:
            continue
        y = phase_df["y_true"].values
        metrics[phase] = {
            "hedge_mae": float(phase_df["loss_raw"].mean()),
            "equal_mae": float(phase_df["loss_equal"].mean()),
            "best_mae": float(phase_df["loss_best"].mean()),
            "hedge_smape": float(
                _smape_array(y, phase_df["y_pred"].values).mean()
            ),
            "equal_smape": float(
                _smape_array(y, phase_df["y_pred_equal"].values).mean()
            ),
            "best_smape": float(
                _smape_array(y, phase_df["y_pred_best"].values).mean()
            ),
            "n_points": len(phase_df),
        }

        # 系列別MAE
        series_mae = {}
        for method, col in [
            ("hedge", "loss_raw"),
            ("equal", "loss_equal"),
            ("best", "loss_best"),
        ]:
            series_mae[method] = (
                phase_df.groupby("series")[col].mean().to_dict()
            )
        metrics[phase]["series_mae"] = series_mae

    return metrics


def _smape_array(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """sMAPE (0-200% scale) をベクトル計算."""
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2 + 1e-8
    return np.abs(y_true - y_pred) / denom * 100


# ---------------------------------------------------------------------------
# プロット関数
# ---------------------------------------------------------------------------


def _plot_method_comparison_bar(
    metrics: dict, report_dir: Path
) -> str:
    """手法別MAE比較のバーチャート."""
    _setup_plot_style()

    phases = [p for p in ["valid", "test"] if p in metrics]
    if not phases:
        return ""

    fig, axes = plt.subplots(1, len(phases), figsize=(6 * len(phases), 5))
    if len(phases) == 1:
        axes = [axes]

    for ax, phase in zip(axes, phases):
        m = metrics[phase]
        methods = ["Hedge\n(Expert Advice)", "等重み平均\n(Equal Weight)", "ベストExpert\n(Best Single)"]
        maes = [m["hedge_mae"], m["equal_mae"], m["best_mae"]]
        colors = [_COLORS["hedge"], _COLORS["equal"], _COLORS["best"]]

        bars = ax.bar(methods, maes, color=colors, alpha=0.85, edgecolor="white", linewidth=1.5)
        for bar, val in zip(bars, maes):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(maes) * 0.02,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )
        ax.set_ylabel("MAE")
        ax.set_title(f"{phase.upper()} 期間 — 手法別 MAE 比較", fontsize=13)
        ax.set_ylim(0, max(maes) * 1.25)

    plt.tight_layout()
    fname = "method_comparison.png"
    fig.savefig(report_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fname


def _plot_timeseries_comparison(
    df: pd.DataFrame, report_dir: Path, max_points: int = 500
) -> str:
    """代表系列の時系列比較プロット."""
    _setup_plot_style()

    # テストフェーズの代表系列を選ぶ（MAE差が大きい系列 = Hedgeの効果が見えやすい）
    test_df = df[df["phase"] == "test"]
    if test_df.empty:
        test_df = df[df["phase"] == "valid"]
    if test_df.empty:
        return ""

    series_list = test_df["series"].unique()
    # Hedgeと等重みのMAE差が最も大きい系列を選ぶ
    best_series = None
    best_diff = -1
    for s in series_list:
        sdf = test_df[test_df["series"] == s]
        diff = abs(sdf["loss_equal"].mean() - sdf["loss_raw"].mean())
        if diff > best_diff:
            best_diff = diff
            best_series = s

    if best_series is None:
        best_series = series_list[0]

    sdf = test_df[test_df["series"] == best_series].copy()
    sdf = sdf.sort_values("timestamp")

    # 表示範囲を制限（見やすさのため）
    if len(sdf) > max_points:
        sdf = sdf.iloc[:max_points]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), height_ratios=[3, 1])

    # 上段: 時系列比較
    ts = pd.to_datetime(sdf["timestamp"])
    ax1.plot(ts, sdf["y_true"], color=_COLORS["actual"], linewidth=1.5, label=_LABELS["actual"], alpha=0.8)
    ax1.plot(ts, sdf["y_pred"], color=_COLORS["hedge"], linewidth=1.2, label=_LABELS["hedge"], alpha=0.8)
    ax1.plot(ts, sdf["y_pred_equal"], color=_COLORS["equal"], linewidth=1.0, label=_LABELS["equal"], alpha=0.7, linestyle="--")
    ax1.plot(ts, sdf["y_pred_best"], color=_COLORS["best"], linewidth=1.0, label=_LABELS["best"], alpha=0.7, linestyle=":")

    ax1.set_ylabel("負荷量 (kWh)")
    ax1.set_title(f"時系列予測比較 — 系列: {best_series}", fontsize=13)
    ax1.legend(loc="upper right", fontsize=10)
    ax1.tick_params(axis="x", rotation=30)

    # 下段: 各手法の絶対誤差
    ax2.fill_between(ts, sdf["loss_raw"], alpha=0.4, color=_COLORS["hedge"], label="Hedge MAE")
    ax2.fill_between(ts, sdf["loss_equal"], alpha=0.3, color=_COLORS["equal"], label="等重み MAE")
    ax2.fill_between(ts, sdf["loss_best"], alpha=0.3, color=_COLORS["best"], label="ベスト MAE")
    ax2.set_ylabel("絶対誤差")
    ax2.set_xlabel("時刻")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    fname = "timeseries_comparison.png"
    fig.savefig(report_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fname


def _plot_cumulative_loss(df: pd.DataFrame, report_dir: Path) -> str:
    """累積損失の推移プロット."""
    _setup_plot_style()

    test_df = df[df["phase"] == "test"]
    if test_df.empty:
        test_df = df[df["phase"] == "valid"]
    if test_df.empty:
        return ""

    # 全系列・全時刻の時間順累積損失
    sorted_df = test_df.sort_values("timestamp")

    fig, ax = plt.subplots(figsize=(12, 5))

    for method, loss_col, color, label in [
        ("hedge", "loss_raw", _COLORS["hedge"], _LABELS["hedge"]),
        ("equal", "loss_equal", _COLORS["equal"], _LABELS["equal"]),
        ("best", "loss_best", _COLORS["best"], _LABELS["best"]),
    ]:
        cum = sorted_df[loss_col].cumsum().values
        ax.plot(range(len(cum)), cum, color=color, label=label, linewidth=1.5)

    ax.set_xlabel("時刻ステップ")
    ax.set_ylabel("累積 MAE")
    ax.set_title("累積損失の推移比較 (Test期間)", fontsize=13)
    ax.legend(fontsize=10)

    plt.tight_layout()
    fname = "cumulative_loss.png"
    fig.savefig(report_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fname


def _plot_series_mae_scatter(metrics: dict, report_dir: Path) -> str:
    """系列別MAE散布図: Hedge vs 等重み."""
    _setup_plot_style()

    phase = "test" if "test" in metrics else ("valid" if "valid" in metrics else None)
    if phase is None:
        return ""

    hedge_mae = metrics[phase]["series_mae"]["hedge"]
    equal_mae = metrics[phase]["series_mae"]["equal"]
    series_keys = sorted(hedge_mae.keys())

    h_vals = [hedge_mae[s] for s in series_keys]
    e_vals = [equal_mae[s] for s in series_keys]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(e_vals, h_vals, alpha=0.6, color=_COLORS["hedge"], edgecolors="white", s=50)

    # 対角線
    max_val = max(max(h_vals), max(e_vals)) * 1.1
    ax.plot([0, max_val], [0, max_val], "k--", alpha=0.4, linewidth=1)

    ax.set_xlabel("等重み平均 MAE", fontsize=12)
    ax.set_ylabel("Hedge MAE", fontsize=12)
    ax.set_title(f"系列別 MAE 比較 ({phase.upper()})\n対角線より下 = Hedge が優位", fontsize=13)
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_aspect("equal")

    # 勝率テキスト
    wins = sum(1 for h, e in zip(h_vals, e_vals) if h < e)
    total = len(h_vals)
    ax.text(
        0.05, 0.95,
        f"Hedge 優位: {wins}/{total} 系列 ({100*wins/total:.0f}%)",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=_COLORS["hedge"], alpha=0.2),
    )

    plt.tight_layout()
    fname = "series_mae_scatter.png"
    fig.savefig(report_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fname


# ---------------------------------------------------------------------------
# 重み推移プロット
# ---------------------------------------------------------------------------


def _plot_weights(
    report_dir: Path,
    expert_names: list[str],
    expert_stats: dict | None,
) -> str:
    """上位5 Expert の重み推移プロット (weights_plot.png)."""
    if expert_stats is None:
        return ""
    snapshots = expert_stats.get("weight_snapshots", [])
    if not snapshots:
        return ""

    _setup_plot_style()

    # スナップショットから上位 Expert の重み時系列を構築
    # まず全スナップショットに登場する Expert index を収集し、出現頻度で上位5を選ぶ
    from collections import Counter

    idx_counter: Counter = Counter()
    for snap in snapshots:
        for tw in snap["top_weights"]:
            idx_counter[tw["expert_idx"]] += 1

    top_indices = [idx for idx, _ in idx_counter.most_common(5)]

    steps = [snap["step"] for snap in snapshots]
    # 各 top expert の重みを時系列として取得
    weight_series: dict[int, list[float]] = {idx: [] for idx in top_indices}
    for snap in snapshots:
        snap_dict = {tw["expert_idx"]: tw["weight"] for tw in snap["top_weights"]}
        for idx in top_indices:
            weight_series[idx].append(snap_dict.get(idx, 0.0))

    fig, ax = plt.subplots(figsize=(12, 5))
    for idx in top_indices:
        label = expert_names[idx] if idx < len(expert_names) else f"Expert-{idx}"
        ax.plot(steps, weight_series[idx], linewidth=1.5, label=label, alpha=0.85)

    ax.set_xlabel("時刻ステップ")
    ax.set_ylabel("重み")
    ax.set_title("上位 Expert 重み推移 (Test期間・代表系列)", fontsize=13)
    ax.legend(fontsize=9, loc="best")

    plt.tight_layout()
    fname = "weights_plot.png"
    fig.savefig(report_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fname


def _plot_eta_weights(
    report_dir: Path,
    expert_stats: dict | None,
) -> str:
    """eta 重みの推移プロット (eta_plot.png)。MetaEtaHedge 使用時のみ生成."""
    if expert_stats is None:
        return ""
    if not expert_stats.get("is_meta_eta", False):
        return ""

    snapshots = expert_stats.get("weight_snapshots", [])
    # eta_weights が含まれるスナップショットだけ使う
    eta_snapshots = [s for s in snapshots if "eta_weights" in s]
    if not eta_snapshots:
        return ""

    _setup_plot_style()

    steps = [snap["step"] for snap in eta_snapshots]
    etas = eta_snapshots[0].get("etas", [])
    n_etas = len(etas)

    fig, ax = plt.subplots(figsize=(12, 5))
    for i in range(n_etas):
        weights_i = [snap["eta_weights"][i] for snap in eta_snapshots]
        label = f"eta={etas[i]:.4g}"
        ax.plot(steps, weights_i, linewidth=1.2, label=label, alpha=0.8)

    ax.set_xlabel("時刻ステップ")
    ax.set_ylabel("eta 重み")
    ax.set_title("Meta-eta 重み推移 (Test期間・代表系列)", fontsize=13)
    ax.legend(fontsize=8, loc="best", ncol=2)

    plt.tight_layout()
    fname = "eta_plot.png"
    fig.savefig(report_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fname


# ---------------------------------------------------------------------------
# Markdown レポート生成
# ---------------------------------------------------------------------------


def _generate_markdown(
    report_dir: Path,
    metrics: dict,
    config: dict,
    plot_files: dict[str, str],
) -> str:
    """GitHub向けMarkdownレポートを生成."""

    lines = []
    lines.append("# Expert Advice 実験レポート")
    lines.append("")
    lines.append(f"> Run ID: `{config.get('run_id', 'N/A')}`")
    lines.append("")

    # 実験設定
    lines.append("## 実験設定")
    lines.append("")
    lines.append("| 項目 | 値 |")
    lines.append("|------|-----|")
    lines.append(f"| Expert preset | `{config.get('experts', 'N/A')}` |")
    lines.append(f"| η モード | `{config.get('eta_mode', 'N/A')}` |")
    lines.append(f"| 損失スケーリング | `{config.get('scale_loss', 'N/A')}` |")
    lines.append(f"| 集約方法 | `{config.get('agg', 'N/A')}` |")
    lines.append(f"| 系列サンプル数 | `{config.get('series_sample', config.get('n_series', '全系列'))}` |")
    lines.append(f"| シード | `{config.get('seed', 'N/A')}` |")
    lines.append("")

    # 手法別比較結果
    lines.append("## 手法別比較結果")
    lines.append("")
    lines.append("3つの予測手法を比較しました:")
    lines.append("")
    lines.append("1. **Hedge重み付き (Expert Advice)**: 過去の損失に基づいて指数重み更新で最適な重み配分を学習")
    lines.append("2. **等重み平均 (Equal Weight)**: 全Expertの予測を単純平均")
    lines.append("3. **ベストExpert (Best Single)**: 累積損失が最小の単一Expert")
    lines.append("")

    for phase in ["valid", "test"]:
        if phase not in metrics:
            continue
        m = metrics[phase]
        lines.append(f"### {phase.upper()} 期間")
        lines.append("")
        lines.append("| 手法 | MAE | sMAPE (%) |")
        lines.append("|------|-----|-----------|")
        lines.append(f"| **Hedge重み付き** | **{m['hedge_mae']:.4f}** | **{m['hedge_smape']:.2f}** |")
        lines.append(f"| 等重み平均 | {m['equal_mae']:.4f} | {m['equal_smape']:.2f} |")
        lines.append(f"| ベストExpert | {m['best_mae']:.4f} | {m['best_smape']:.2f} |")
        lines.append(f"| データ点数 | {m['n_points']:,} | |")
        lines.append("")

        # 改善率
        if m["equal_mae"] > 0:
            improv = (m["equal_mae"] - m["hedge_mae"]) / m["equal_mae"] * 100
            direction = "改善" if improv > 0 else "悪化"
            lines.append(f"> Hedge は等重み平均に対して MAE を **{abs(improv):.1f}% {direction}** させました。")
            lines.append("")

    # プロット埋め込み
    if plot_files.get("method_comparison"):
        lines.append("## MAE 比較チャート")
        lines.append("")
        lines.append(f"![手法別MAE比較](./{plot_files['method_comparison']})")
        lines.append("")

    if plot_files.get("timeseries_comparison"):
        lines.append("## 時系列予測比較")
        lines.append("")
        lines.append("代表系列における実測値と各手法の予測値の比較です。")
        lines.append("下段は各時刻での絶対誤差を示しています。")
        lines.append("")
        lines.append(f"![時系列比較](./{plot_files['timeseries_comparison']})")
        lines.append("")

    if plot_files.get("cumulative_loss"):
        lines.append("## 累積損失の推移")
        lines.append("")
        lines.append("時間経過に伴う累積損失の推移です。Hedge手法は時間とともに重みを最適化するため、")
        lines.append("等重み平均より低い累積損失で推移することが期待されます。")
        lines.append("")
        lines.append(f"![累積損失](./{plot_files['cumulative_loss']})")
        lines.append("")

    if plot_files.get("series_mae_scatter"):
        lines.append("## 系列別 MAE 散布図")
        lines.append("")
        lines.append("各系列の MAE を Hedge vs 等重み平均でプロットしました。")
        lines.append("対角線より下の点は Hedge が優位な系列を示します。")
        lines.append("")
        lines.append(f"![系列別MAE散布図](./{plot_files['series_mae_scatter']})")
        lines.append("")

    if plot_files.get("weights_plot"):
        lines.append("## 上位 Expert 重み推移")
        lines.append("")
        lines.append("代表系列における上位5 Expertの重み推移です。")
        lines.append("Hedge アルゴリズムがどの Expert を重視しているかの変遷を示します。")
        lines.append("")
        lines.append(f"![上位Expert重み推移](./{plot_files['weights_plot']})")
        lines.append("")

    if plot_files.get("eta_plot"):
        lines.append("## Meta-eta 重み推移")
        lines.append("")
        lines.append("Meta-eta Hedge における各候補学習率 (eta) の重みの推移です。")
        lines.append("メタレベルの Hedge がどの学習率を選好しているかを示します。")
        lines.append("")
        lines.append(f"![Meta-eta重み推移](./{plot_files['eta_plot']})")
        lines.append("")

    # 考察
    lines.append("## 考察")
    lines.append("")
    lines.append("- Expert Advice (Hedge) アンサンブルは、各Expertの過去の実績に基づいて動的に重みを調整するため、")
    lines.append("  単純な等重み平均と比較して、特に予測が難しい期間での適応能力が期待されます。")
    lines.append("- ベストExpert は事後的に最良のものを選んでいるため参考値ですが、")
    lines.append("  Hedge がこれに近い性能を出せていれば、重み学習が効果的に機能していると言えます。")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# メインエントリポイント
# ---------------------------------------------------------------------------


def generate_report(
    report_dir: Path,
    result_df: pd.DataFrame,
    config: dict,
    expert_names: list[str],
    expert_stats: dict | None = None,
) -> None:
    """レポートを生成して report_dir に保存する.

    Parameters
    ----------
    report_dir : Path
        レポート出力先ディレクトリ
    result_df : pd.DataFrame
        予測結果 DataFrame
    config : dict
        実験設定
    expert_names : list[str]
        Expert 名のリスト
    expert_stats : dict | None
        Expert 別統計 (expert_cum_losses, expert_win_counts,
        expert_avg_weights, total_steps, weight_snapshots, is_meta_eta)
    """
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    # 指標計算
    metrics = _compute_metrics(result_df)

    # 指標JSON保存
    metrics_path = report_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        # series_mae は大きくなるので別途CSVに
        metrics_for_json = {}
        for phase, m in metrics.items():
            metrics_for_json[phase] = {k: v for k, v in m.items() if k != "series_mae"}
        json.dump(metrics_for_json, f, indent=2, ensure_ascii=False)

    # Expert rank CSV (充実版)
    _save_expert_rank_csv(report_dir, result_df, expert_names, expert_stats=expert_stats)

    # プロット生成
    plot_files = {}
    plot_files["method_comparison"] = _plot_method_comparison_bar(metrics, report_dir)
    plot_files["timeseries_comparison"] = _plot_timeseries_comparison(result_df, report_dir)
    plot_files["cumulative_loss"] = _plot_cumulative_loss(result_df, report_dir)
    plot_files["series_mae_scatter"] = _plot_series_mae_scatter(metrics, report_dir)
    plot_files["weights_plot"] = _plot_weights(report_dir, expert_names, expert_stats)
    plot_files["eta_plot"] = _plot_eta_weights(report_dir, expert_stats)

    # Markdown レポート
    md_content = _generate_markdown(report_dir, metrics, config, plot_files)
    md_path = report_dir / "summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)


def _save_expert_rank_csv(
    report_dir: Path,
    result_df: pd.DataFrame,
    expert_names: list[str],
    expert_stats: dict | None = None,
) -> None:
    """Expert ランキングを CSV で保存 (平均損失・平均重み・勝率を含む)."""
    if not expert_names:
        return

    n = len(expert_names)

    if expert_stats is not None and expert_stats.get("total_steps", 0) > 0:
        total = expert_stats["total_steps"]
        avg_losses = expert_stats["expert_cum_losses"] / max(total, 1)
        avg_weights = expert_stats["expert_avg_weights"]
        win_rates = expert_stats["expert_win_counts"] / max(total, 1)

        # 平均損失でソート (昇順 = 良い Expert が上位)
        order = np.argsort(avg_losses)
        rank_df = pd.DataFrame(
            {
                "rank": range(1, n + 1),
                "expert_name": [expert_names[i] for i in order],
                "avg_loss": [float(avg_losses[i]) for i in order],
                "avg_weight": [float(avg_weights[i]) for i in order],
                "win_rate": [float(win_rates[i]) for i in order],
            }
        )
    else:
        # 統計なしの場合はフォールバック（名前のみ + NaN 列）
        rank_df = pd.DataFrame(
            {
                "rank": range(1, n + 1),
                "expert_name": expert_names,
                "avg_loss": [float("nan")] * n,
                "avg_weight": [float("nan")] * n,
                "win_rate": [float("nan")] * n,
            }
        )

    rank_df.to_csv(report_dir / "experts_rank.csv", index=False)
