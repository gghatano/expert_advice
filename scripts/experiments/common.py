"""実験スクリプト共通のユーティリティ.

* 集約アルゴリズムの標準セット
* 評価スイート (アルゴリズム群 + 事後ベンチマーク)
* 図のスタイルと配色
* 結果 JSON の保存

図のラベルはすべて英語にしている。matplotlib の既定フォントに日本語
グリフが無く、豆腐文字になるため。解説文は HTML 側で日本語にする。
"""

from __future__ import annotations

import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.ensemble.adahedge import AdaHedge  # noqa: E402
from src.ensemble.fixed_share import FixedShare  # noqa: E402
from src.ensemble.hedge import Hedge  # noqa: E402
from src.ensemble.meta_eta import MetaEtaHedge  # noqa: E402
from src.ensemble.ml_poly import MLPoly  # noqa: E402
from src.ensemble.runner import (  # noqa: E402
    EqualWeight,
    FollowTheLeader,
    RunResult,
    best_fixed_expert,
    expert_loss_matrix,
    run_online,
)

FIG_DIR = _PROJECT_ROOT / "docs" / "figures"
RESULT_DIR = _PROJECT_ROOT / "docs" / "results"
DATA_DIR = _PROJECT_ROOT / "data" / "raw"

# ---------------------------------------------------------------------------
# スタイル
# ---------------------------------------------------------------------------

COLORS: dict[str, str] = {
    "Equal Weight": "#9E9E9E",
    "Follow the Leader": "#795548",
    "Hedge (tuned eta)": "#42A5F5",
    "Meta-eta Hedge": "#1565C0",
    "AdaHedge": "#26A69A",
    "Fixed-Share": "#E53935",
    "ML-Poly": "#8E24AA",
    "Best fixed expert": "#212121",
    "Oracle switching": "#F9A825",
    "Actual": "#333333",
}

# 図と表で共通に使う並び順
ALGO_ORDER: list[str] = [
    "Equal Weight",
    "Follow the Leader",
    "Hedge (tuned eta)",
    "Meta-eta Hedge",
    "AdaHedge",
    "Fixed-Share",
    "ML-Poly",
]


def setup_style() -> None:
    """図の共通スタイルを設定する."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.25,
            "font.size": 11,
            "axes.titlesize": 13,
            "legend.frameon": False,
            "figure.autolayout": False,
        }
    )


def save_fig(fig: plt.Figure, exp: str, name: str) -> Path:
    """図を ``docs/figures/<exp>/<name>.png`` に保存する."""
    out_dir = FIG_DIR / exp
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"    figure -> {path.relative_to(_PROJECT_ROOT)}")
    return path


def save_results(exp: str, payload: dict) -> Path:
    """結果 JSON を ``docs/results/<exp>.json`` に保存する."""
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULT_DIR / f"{exp}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=_json_default)
    print(f"    results -> {path.relative_to(_PROJECT_ROOT)}")
    return path


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


# ---------------------------------------------------------------------------
# 集約アルゴリズムの標準セット
# ---------------------------------------------------------------------------

# Hedge の eta は本来チューニングが必要。ここでは「事後に最良を選べたら」
# という有利な条件 (oracle) を与えて比較する。
HEDGE_ETA_GRID: tuple[float, ...] = (
    0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0,
)


def default_factories(alpha: float = 0.01) -> "OrderedDict[str, Callable[[int], object]]":
    """アルゴリズム名 -> インスタンス生成関数 の辞書を返す.

    ``Hedge (tuned eta)`` だけは別扱い (グリッド探索) なのでここには含めない。
    """
    return OrderedDict(
        [
            ("Equal Weight", lambda n: EqualWeight(n)),
            ("Follow the Leader", lambda n: FollowTheLeader(n)),
            ("Meta-eta Hedge", lambda n: MetaEtaHedge(n)),
            ("AdaHedge", lambda n: AdaHedge(n)),
            ("Fixed-Share", lambda n: FixedShare(n, eta=1.0, alpha=alpha)),
            ("ML-Poly", lambda n: MLPoly(n)),
        ]
    )


def run_suite(
    P: np.ndarray,
    y: np.ndarray,
    *,
    loss_scale: float = 1.0,
    update_loss: str = "abs",
    eval_loss: str = "abs",
    alpha: float = 0.01,
    record_weights: bool = False,
    snapshot_every: int = 1,
    eta_grid: tuple[float, ...] = HEDGE_ETA_GRID,
) -> dict[str, RunResult]:
    """標準セットのアルゴリズムをすべて走らせる.

    ``Hedge (tuned eta)`` は ``eta_grid`` を全探索し、事後的に最も損失が
    小さかったものを採用する (アルゴリズムに有利な oracle 条件)。
    """
    n = P.shape[1]
    results: dict[str, RunResult] = {}

    best: RunResult | None = None
    best_eta = None
    for eta in eta_grid:
        r = run_online(
            P, y, Hedge(n, eta=eta), name=f"Hedge(eta={eta})",
            update_loss=update_loss, eval_loss=eval_loss, loss_scale=loss_scale,
        )
        if best is None or r.mean_loss < best.mean_loss:
            best, best_eta = r, eta
    assert best is not None
    if record_weights:
        best = run_online(
            P, y, Hedge(n, eta=best_eta), name="Hedge (tuned eta)",
            update_loss=update_loss, eval_loss=eval_loss, loss_scale=loss_scale,
            record_weights=True, snapshot_every=snapshot_every,
        )
    best.name = "Hedge (tuned eta)"
    best.extra["eta"] = best_eta
    results["Hedge (tuned eta)"] = best

    for name, factory in default_factories(alpha=alpha).items():
        results[name] = run_online(
            P, y, factory(n), name=name,
            update_loss=update_loss, eval_loss=eval_loss, loss_scale=loss_scale,
            record_weights=record_weights, snapshot_every=snapshot_every,
        )

    return {k: results[k] for k in ALGO_ORDER if k in results}


def benchmark_table(
    results: dict[str, RunResult],
    P: np.ndarray,
    y: np.ndarray,
    expert_names: list[str],
    *,
    eval_loss: str = "abs",
) -> dict:
    """アルゴリズム結果 + 事後ベンチマークを 1 つの辞書にまとめる."""
    L = expert_loss_matrix(P, y, loss=eval_loss)
    best_idx, best_mean = best_fixed_expert(L)

    rows = []
    for name, res in results.items():
        rows.append(
            {
                "algorithm": name,
                "mean_loss": res.mean_loss,
                "vs_best_fixed_pct": 100.0 * (res.mean_loss - best_mean) / best_mean,
                "beats_best_fixed": bool(res.mean_loss < best_mean),
                **({k: v for k, v in res.extra.items()}),
            }
        )

    expert_means = L.mean(axis=0)
    order = np.argsort(expert_means)
    return {
        "n_steps": int(len(y)),
        "n_experts": int(P.shape[1]),
        "eval_loss": eval_loss,
        "best_fixed_expert": {
            "name": expert_names[best_idx],
            "index": int(best_idx),
            "mean_loss": float(best_mean),
        },
        "worst_expert_mean_loss": float(expert_means.max()),
        "expert_ranking": [
            {"name": expert_names[i], "mean_loss": float(expert_means[i])}
            for i in order[:10]
        ],
        "algorithms": rows,
    }


# ---------------------------------------------------------------------------
# 共通の図
# ---------------------------------------------------------------------------


def plot_regret_curves(
    results: dict[str, RunResult],
    P: np.ndarray,
    y: np.ndarray,
    *,
    title: str,
    eval_loss: str = "abs",
    xlabel: str = "time step",
    subset: list[str] | None = None,
) -> plt.Figure:
    """最良固定 Expert に対する累積 regret 曲線.

    0 より下 = 「事後に選べる最良の単一 Expert」より良い、という意味。
    Expert Advice の理論保証はこの曲線が線形に増えないことを主張する。
    """
    L = expert_loss_matrix(P, y, loss=eval_loss)
    best_idx, _ = best_fixed_expert(L)
    baseline = np.cumsum(L[:, best_idx])

    fig, ax = plt.subplots(figsize=(9, 4.6))
    names = subset if subset is not None else list(results)
    for name in names:
        res = results[name]
        regret = res.cumulative_loss - baseline
        ax.plot(regret, label=name, color=COLORS.get(name), lw=1.6)

    ax.axhline(0, color=COLORS["Best fixed expert"], lw=1.4, ls="--",
               label="Best fixed expert (post hoc)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("cumulative regret")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9, ncol=2)
    return fig


def plot_mean_loss_bars(
    table: dict,
    *,
    title: str,
    ylabel: str = "mean absolute error",
) -> plt.Figure:
    """平均損失の棒グラフ (最良固定 Expert の水平線つき)."""
    rows = table["algorithms"]
    names = [r["algorithm"] for r in rows]
    values = [r["mean_loss"] for r in rows]
    best = table["best_fixed_expert"]["mean_loss"]

    fig, ax = plt.subplots(figsize=(9, 4.4))
    bars = ax.bar(range(len(names)), values,
                  color=[COLORS.get(n, "#607D8B") for n in names])
    ax.axhline(best, color=COLORS["Best fixed expert"], ls="--", lw=1.4,
               label=f"Best fixed expert (post hoc): {table['best_fixed_expert']['name']}")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=9)

    lo = min(min(values), best)
    hi = max(max(values), best)
    pad = (hi - lo) * 0.25 + 1e-12
    ax.set_ylim(max(0, lo - pad), hi + pad)
    for b, v in zip(bars, values):
        ax.annotate(f"{v:.4g}", (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=8)
    return fig


def plot_weight_evolution(
    res: RunResult,
    expert_names: list[str],
    *,
    title: str,
    top_k: int = 8,
    xlabel: str = "time step",
) -> plt.Figure:
    """重みの推移を積み上げ面グラフで描く (上位 top_k 以外はまとめる)."""
    if res.weights is None or res.snapshot_steps is None:
        raise ValueError(f"{res.name}: weights were not recorded")

    W = res.weights
    mean_w = W.mean(axis=0)
    top = np.argsort(mean_w)[-top_k:][::-1]
    rest = np.setdiff1d(np.arange(W.shape[1]), top)

    stack = [W[:, i] for i in top]
    labels = [expert_names[i] for i in top]
    if len(rest) > 0:
        stack.append(W[:, rest].sum(axis=1))
        labels.append(f"others ({len(rest)})")

    fig, ax = plt.subplots(figsize=(9, 4.4))
    cmap = plt.get_cmap("tab20")
    ax.stackplot(res.snapshot_steps, *stack, labels=labels,
                 colors=[cmap(i % 20) for i in range(len(stack))])
    ax.set_xlim(res.snapshot_steps[0], res.snapshot_steps[-1])
    ax.set_ylim(0, 1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("weight")
    ax.set_title(title)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    ax.grid(False)
    return fig
