"""README用の図を合成データで生成するスクリプト.

Usage:
    uv run python scripts/generate_readme_figures.py

生成先: docs/figures/
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# プロジェクトルートを sys.path に追加
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.preprocess import preprocess
from src.data.split import split_temporal
from src.ensemble.loss import mae_loss
from src.ensemble.meta_eta import MetaEtaHedge
from src.ensemble.scaling import by_train_mae
from src.experts.factory import create_experts

OUTPUT_DIR = _PROJECT_ROOT / "docs" / "figures"

# ---------------------------------------------------------------------------
# 色・スタイル定数
# ---------------------------------------------------------------------------

_COLORS = {
    "actual": "#333333",
    "hedge": "#2196F3",
    "equal": "#FF9800",
    "best": "#4CAF50",
}

_LABELS = {
    "actual": "Actual",
    "hedge": "Hedge (Expert Advice)",
    "equal": "Equal Weight",
    "best": "Best Single Expert",
}


def _setup_style() -> None:
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
# 合成データ (test_e2e.py と同様の手法)
# ---------------------------------------------------------------------------


def _make_synthetic_data() -> pd.DataFrame:
    """15分間隔、5系列の合成データを生成."""
    idx = pd.date_range("2011-01-01", "2014-12-31 23:45:00", freq="15min")
    rng = np.random.RandomState(12345)
    data = {}
    for i in range(5):
        hour = idx.hour + idx.minute / 60.0
        base = 100.0 + 50.0 * np.sin(2 * np.pi * hour / 24.0) + i * 20.0
        noise = rng.normal(0, 5.0, size=len(idx))
        data[f"series_{i:03d}"] = np.maximum(base + noise, 0.0)
    df = pd.DataFrame(data, index=idx)
    df.index.name = "datetime"
    return df


# ---------------------------------------------------------------------------
# 実験実行 (1系列)
# ---------------------------------------------------------------------------


def _run_experiment() -> dict:
    """合成データで1系列の実験を実行し、結果を返す."""
    print("  Generating synthetic data...")
    raw = _make_synthetic_data()
    df = preprocess(raw, resample_method="sum", clip=False)
    splits = split_temporal(df)

    col = df.columns[0]
    train_s = splits.train[col]
    valid_s = splits.valid[col]
    test_s = splits.test[col]

    print("  Creating experts (light30)...")
    experts = create_experts(preset="light30")
    n_experts = len(experts)

    # fit
    for exp in experts:
        try:
            exp.fit(train_s)
        except Exception:
            pass

    # train MAE for scaling
    y_true_train = train_s.values[1:]
    y_pred_train = train_s.values[:-1]
    train_mae = float(np.mean(np.abs(y_true_train - y_pred_train)))
    if train_mae <= 0:
        train_mae = 1.0

    # ensemble
    ensemble = MetaEtaHedge(n_experts=n_experts)

    records = []
    weight_snapshots = []
    running_history = train_s.copy()

    # valid + test を順に処理
    phases = [("valid", valid_s), ("test", test_s)]
    for phase_name, phase_data in phases:
        print(f"  Running {phase_name} phase ({len(phase_data)} steps)...")
        expert_cum_losses = np.zeros(n_experts)

        for t_idx, (tstamp, y_true) in enumerate(phase_data.items()):
            expert_preds = np.empty(n_experts)
            for i, exp in enumerate(experts):
                try:
                    pred = exp.predict_next(running_history, tstamp)
                    expert_preds[i] = pred if np.isfinite(pred) else running_history.iloc[-1]
                except Exception:
                    expert_preds[i] = running_history.iloc[-1]

            ensemble_pred = ensemble.predict(expert_preds)
            equal_pred = float(np.mean(expert_preds))

            raw_losses = np.array([mae_loss(y_true, expert_preds[i]) for i in range(n_experts)])
            ensemble_loss = mae_loss(y_true, ensemble_pred)
            equal_loss = mae_loss(y_true, equal_pred)

            best_idx = int(np.argmin(expert_cum_losses)) if t_idx > 0 else 0
            best_loss = raw_losses[best_idx]

            expert_cum_losses += raw_losses

            scaled = np.array([by_train_mae(l, train_mae) for l in raw_losses])
            ensemble.update(scaled)

            # weight snapshot (test phase only, every 24 steps)
            if phase_name == "test" and t_idx % 24 == 0:
                w = ensemble.get_weights()
                top5_idx = np.argsort(w)[-5:][::-1]
                weight_snapshots.append(
                    {
                        "step": t_idx,
                        "weights": {
                            experts[idx].name: float(w[idx]) for idx in top5_idx
                        },
                    }
                )

            records.append(
                {
                    "timestamp": tstamp,
                    "y_true": float(y_true),
                    "y_pred": float(ensemble_pred),
                    "y_pred_equal": float(equal_pred),
                    "y_pred_best": float(expert_preds[best_idx]),
                    "loss_raw": float(ensemble_loss),
                    "loss_equal": float(equal_loss),
                    "loss_best": float(best_loss),
                    "phase": phase_name,
                }
            )

            new_point = pd.Series([y_true], index=[tstamp])
            running_history = pd.concat([running_history, new_point])

    result_df = pd.DataFrame(records)
    expert_names = [exp.name for exp in experts]
    return {
        "df": result_df,
        "expert_names": expert_names,
        "weight_snapshots": weight_snapshots,
    }


# ---------------------------------------------------------------------------
# 図1: アルゴリズム概念図
# ---------------------------------------------------------------------------


def plot_algorithm_concept(output_dir: Path) -> str:
    """Expert Advice アルゴリズムの概念図を描画."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_facecolor("white")

    # Expert boxes
    expert_colors = ["#E3F2FD", "#E8F5E9", "#FFF3E0", "#FCE4EC"]
    expert_labels = [
        "Naive\n(LastValue, Seasonal,\nDrift)",
        "Moving Average\n(SMA, Median,\nEMA)",
        "Regression\n(Ridge, Huber,\nKNN)",
        "Seasonal\n(STL Profile)",
    ]
    expert_x = [0.5, 2.7, 4.9, 7.1]
    for i, (x, label, color) in enumerate(zip(expert_x, expert_labels, expert_colors)):
        rect = mpatches.FancyBboxPatch(
            (x, 3.8), 1.8, 1.8,
            boxstyle="round,pad=0.15",
            facecolor=color,
            edgecolor="#666",
            linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.text(x + 0.9, 4.7, label, ha="center", va="center", fontsize=8.5, fontweight="bold")

    # Arrows from experts to Hedge
    for x in expert_x:
        ax.annotate(
            "",
            xy=(5.0, 3.0),
            xytext=(x + 0.9, 3.8),
            arrowprops=dict(arrowstyle="->", color="#888", lw=1.5),
        )

    # Hedge box
    hedge_rect = mpatches.FancyBboxPatch(
        (3.5, 1.8), 3.0, 1.2,
        boxstyle="round,pad=0.2",
        facecolor="#BBDEFB",
        edgecolor="#1976D2",
        linewidth=2,
    )
    ax.add_patch(hedge_rect)
    ax.text(5.0, 2.65, "Hedge Algorithm", ha="center", va="center", fontsize=12, fontweight="bold", color="#1565C0")
    ax.text(5.0, 2.2, r"$w_i^{(t+1)} = w_i^{(t)} \cdot e^{-\eta \cdot \ell_i^{(t)}}$",
            ha="center", va="center", fontsize=10, color="#333")

    # Arrow to prediction
    ax.annotate(
        "",
        xy=(5.0, 0.9),
        xytext=(5.0, 1.8),
        arrowprops=dict(arrowstyle="-|>", color="#1976D2", lw=2),
    )

    # Prediction box
    pred_rect = mpatches.FancyBboxPatch(
        (3.5, 0.2), 3.0, 0.7,
        boxstyle="round,pad=0.15",
        facecolor="#E8F5E9",
        edgecolor="#388E3C",
        linewidth=2,
    )
    ax.add_patch(pred_rect)
    ax.text(5.0, 0.55, r"Prediction: $\hat{y} = \sum_i w_i \cdot f_i(x)$",
            ha="center", va="center", fontsize=10, fontweight="bold", color="#2E7D32")

    # Meta-eta annotation
    meta_rect = mpatches.FancyBboxPatch(
        (7.5, 1.8), 2.2, 1.2,
        boxstyle="round,pad=0.15",
        facecolor="#F3E5F5",
        edgecolor="#7B1FA2",
        linewidth=1.5,
        linestyle="--",
    )
    ax.add_patch(meta_rect)
    ax.text(8.6, 2.65, "Meta-\u03b7 Layer", ha="center", va="center", fontsize=10, fontweight="bold", color="#6A1B9A")
    ax.text(8.6, 2.2, "Auto-tunes \u03b7\nvia 2nd-level Hedge",
            ha="center", va="center", fontsize=8, color="#555")
    ax.annotate(
        "",
        xy=(6.5, 2.4),
        xytext=(7.5, 2.4),
        arrowprops=dict(arrowstyle="<->", color="#7B1FA2", lw=1.5, linestyle="--"),
    )

    # Title
    ax.text(5.0, 5.85, "Prediction with Expert Advice", ha="center", va="center",
            fontsize=14, fontweight="bold", color="#333")

    fname = "algorithm_concept.png"
    fig.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fname


# ---------------------------------------------------------------------------
# 図2: 手法別MAE比較バーチャート
# ---------------------------------------------------------------------------


def plot_mae_comparison(result_df: pd.DataFrame, output_dir: Path) -> str:
    """Hedge vs Equal Weight vs Best Single Expert のMAE比較."""
    _setup_style()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, phase in zip(axes, ["valid", "test"]):
        pdf = result_df[result_df["phase"] == phase]
        if pdf.empty:
            continue
        methods = ["Hedge\n(Expert Advice)", "Equal\nWeight", "Best Single\nExpert"]
        maes = [pdf["loss_raw"].mean(), pdf["loss_equal"].mean(), pdf["loss_best"].mean()]
        colors = [_COLORS["hedge"], _COLORS["equal"], _COLORS["best"]]

        bars = ax.bar(methods, maes, color=colors, alpha=0.85, edgecolor="white", linewidth=1.5)
        for bar, val in zip(bars, maes):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(maes) * 0.02,
                f"{val:.2f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold",
            )
        ax.set_ylabel("MAE")
        ax.set_title(f"{phase.upper()} Phase \u2014 MAE Comparison", fontsize=13)
        ax.set_ylim(0, max(maes) * 1.3)

    plt.tight_layout()
    fname = "mae_comparison.png"
    fig.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fname


# ---------------------------------------------------------------------------
# 図3: 時系列予測比較
# ---------------------------------------------------------------------------


def plot_timeseries(result_df: pd.DataFrame, output_dir: Path) -> str:
    """実測値と各手法の予測を重ねたプロット (test期間の先頭336時間=2週間)."""
    _setup_style()

    test_df = result_df[result_df["phase"] == "test"].copy()
    test_df = test_df.sort_values("timestamp")
    # 2週間分 (見やすさ)
    test_df = test_df.iloc[:336]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1])

    ts = pd.to_datetime(test_df["timestamp"])
    ax1.plot(ts, test_df["y_true"], color=_COLORS["actual"], linewidth=1.5, label=_LABELS["actual"], alpha=0.9)
    ax1.plot(ts, test_df["y_pred"], color=_COLORS["hedge"], linewidth=1.2, label=_LABELS["hedge"], alpha=0.85)
    ax1.plot(ts, test_df["y_pred_equal"], color=_COLORS["equal"], linewidth=1.0, label=_LABELS["equal"], alpha=0.7, linestyle="--")
    ax1.plot(ts, test_df["y_pred_best"], color=_COLORS["best"], linewidth=1.0, label=_LABELS["best"], alpha=0.7, linestyle=":")

    ax1.set_ylabel("Load (kWh)")
    ax1.set_title("Time Series Prediction Comparison (Test, First 2 Weeks)", fontsize=13)
    ax1.legend(loc="upper right", fontsize=10)
    ax1.tick_params(axis="x", rotation=30)

    # Error subplot
    ax2.fill_between(ts, test_df["loss_raw"], alpha=0.4, color=_COLORS["hedge"], label="Hedge MAE")
    ax2.fill_between(ts, test_df["loss_equal"], alpha=0.3, color=_COLORS["equal"], label="Equal Weight MAE")
    ax2.set_ylabel("Absolute Error")
    ax2.set_xlabel("Time")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    fname = "timeseries_comparison.png"
    fig.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fname


# ---------------------------------------------------------------------------
# 図4: 累積損失推移
# ---------------------------------------------------------------------------


def plot_cumulative_loss(result_df: pd.DataFrame, output_dir: Path) -> str:
    """時間経過に伴う累積損失の比較."""
    _setup_style()

    test_df = result_df[result_df["phase"] == "test"].sort_values("timestamp")

    fig, ax = plt.subplots(figsize=(12, 5))

    for col, color, label in [
        ("loss_raw", _COLORS["hedge"], _LABELS["hedge"]),
        ("loss_equal", _COLORS["equal"], _LABELS["equal"]),
        ("loss_best", _COLORS["best"], _LABELS["best"]),
    ]:
        cum = test_df[col].cumsum().values
        ax.plot(range(len(cum)), cum, color=color, label=label, linewidth=1.5)

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Cumulative MAE")
    ax.set_title("Cumulative Loss Over Time (Test Phase)", fontsize=13)
    ax.legend(fontsize=10)

    plt.tight_layout()
    fname = "cumulative_loss.png"
    fig.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fname


# ---------------------------------------------------------------------------
# 図5: 重み推移
# ---------------------------------------------------------------------------


def plot_weight_evolution(weight_snapshots: list[dict], output_dir: Path) -> str:
    """上位Expert の重みがどう変化するかを描画."""
    _setup_style()

    if not weight_snapshots:
        return ""

    # 全スナップショットに登場する Expert 名を集約
    from collections import Counter
    name_counter: Counter = Counter()
    for snap in weight_snapshots:
        for name in snap["weights"]:
            name_counter[name] += 1

    # 上位5 Expert
    top_names = [name for name, _ in name_counter.most_common(5)]

    steps = [snap["step"] for snap in weight_snapshots]
    fig, ax = plt.subplots(figsize=(12, 5))

    for name in top_names:
        weights = [snap["weights"].get(name, 0.0) for snap in weight_snapshots]
        ax.plot(steps, weights, linewidth=1.5, label=name, alpha=0.85)

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Weight")
    ax.set_title("Top Expert Weight Evolution (Test Phase)", fontsize=13)
    ax.legend(fontsize=9, loc="best")

    plt.tight_layout()
    fname = "weight_evolution.png"
    fig.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fname


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating README figures...")

    # 図1: アルゴリズム概念図
    print("[1/5] Algorithm concept diagram...")
    plot_algorithm_concept(OUTPUT_DIR)

    # 実験実行
    print("[2/5] Running experiment on synthetic data...")
    result = _run_experiment()
    result_df = result["df"]

    # 図2: MAE比較
    print("[3/5] MAE comparison bar chart...")
    plot_mae_comparison(result_df, OUTPUT_DIR)

    # 図3: 時系列比較
    print("[4/5] Time series comparison plot...")
    plot_timeseries(result_df, OUTPUT_DIR)

    # 図4: 累積損失
    print("[4/5] Cumulative loss plot...")
    plot_cumulative_loss(result_df, OUTPUT_DIR)

    # 図5: 重み推移
    print("[5/5] Weight evolution plot...")
    plot_weight_evolution(result["weight_snapshots"], OUTPUT_DIR)

    print(f"\nAll figures saved to {OUTPUT_DIR}/")
    for f in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
