"""実験1: レジーム切替データ — Expert Advice が最も分かりやすく効く条件.

生成過程を途中で 4 回切り替えた合成系列を使う。各レジームで「正解の
Expert」が入れ替わるため、

* 最良の *固定* Expert は、どのレジームでも二番手以下にしかなれない
* Hedge は最良固定 Expert に漸近するだけなので、それを超えられない
* Fixed-Share は重みに下限を設けるので、切り替えに追随して
  最良固定 Expert を **下回る** 損失を達成できる

という差がはっきり出る。実データを持ち出す前に、この人工データで
「効く/効かない」の境目を見せるのが目的。

Usage:
    uv run python scripts/experiments/exp1_regime.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import (  # noqa: E402  (scripts/experiments をパスに追加して実行)
    COLORS,
    benchmark_table,
    plot_mean_loss_bars,
    plot_regret_curves,
    plot_weight_evolution,
    run_suite,
    save_fig,
    save_results,
    setup_style,
)

import matplotlib.pyplot as plt  # noqa: E402

from src.ensemble.fixed_share import FixedShare  # noqa: E402
from src.ensemble.hedge import Hedge  # noqa: E402
from src.ensemble.runner import (  # noqa: E402
    best_fixed_expert,
    expert_loss_matrix,
    oracle_switching_loss,
    run_online,
)
from src.experts.vectorized import build_expert_matrix  # noqa: E402

EXP = "exp1_regime"
REGIME_LEN = 2000
WARMUP = 336  # 2 週間ぶんは Expert の助走に使い、評価には含めない
SEED = 20240817

REGIMES = [
    ("A: strong daily cycle", "#BBDEFB"),
    ("B: random walk", "#C8E6C9"),
    ("C: noisy flat level", "#FFE0B2"),
    ("D: steep trend", "#F8BBD0"),
]


def make_series(n_regimes: int = 4, regime_len: int = REGIME_LEN) -> tuple[pd.Series, list[int]]:
    """4 つのレジームをつないだ合成時系列を作る.

    レジームごとに「勝てる Expert」が変わるよう設計している:

    A) 日周期が強くノイズが小さい  -> SeasonalNaive_24 / SeasonalProfile
    B) ランダムウォーク            -> LastValue (平均を取ると必ず遅れる)
    C) 一定水準まわりの大ノイズ    -> SMA_24 (平均でノイズを潰せる)
    D) 急なトレンド                -> Drift_24 (外挿しないと追いつけない)
    """
    rng = np.random.RandomState(SEED)
    idx = pd.date_range("2021-01-01", periods=n_regimes * regime_len, freq="h")
    y = np.empty(len(idx), dtype=np.float64)

    level = 100.0
    for r in range(n_regimes):
        s, e = r * regime_len, (r + 1) * regime_len
        t = np.arange(regime_len, dtype=np.float64)
        hour = idx[s:e].hour.to_numpy(dtype=np.float64)
        kind = r % 4

        if kind == 0:  # A: 強い日周期・低ノイズ
            seg = level + 30.0 * np.sin(2 * np.pi * hour / 24.0) + rng.normal(0, 1.0, regime_len)
        elif kind == 1:  # B: ランダムウォーク
            steps = rng.normal(0, 3.0, regime_len)
            seg = level + np.cumsum(steps)
        elif kind == 2:  # C: 一定水準 + 大きなノイズ
            seg = level + rng.normal(0, 18.0, regime_len)
        else:  # D: 急トレンド (前半は上昇、後半は下降の三角波)
            half = regime_len // 2
            ramp = np.concatenate([
                0.5 * np.arange(half, dtype=np.float64),
                0.5 * half - 0.5 * np.arange(regime_len - half, dtype=np.float64),
            ])
            seg = level + ramp + rng.normal(0, 3.0, regime_len)

        y[s:e] = seg
        level = float(seg[-1])  # レジーム間の水準を連続させる

    bounds = [r * regime_len for r in range(n_regimes + 1)]
    return pd.Series(y, index=idx), bounds


def main() -> None:
    setup_style()
    print(f"[{EXP}] building series ...")
    series, bounds = make_series()
    y_all = series.to_numpy()

    names, P_all = build_expert_matrix(
        y_all,
        index=series.index,
        period=24,
        week=168,
        preset="core8",
        train_end=WARMUP,  # 学習系 Expert は助走区間だけで学習 (リーク防止)
    )

    # 評価は助走区間のあと。アルゴリズムもここから学習を始める。
    P, y = P_all[WARMUP:], y_all[WARMUP:]
    eval_bounds = [max(0, b - WARMUP) for b in bounds]
    segments = list(zip(eval_bounds[:-1], eval_bounds[1:]))

    # 損失スケール: 助走区間における「直近値予測」の MAE
    scale = float(np.mean(np.abs(np.diff(y_all[:WARMUP]))))
    print(f"[{EXP}] T={len(y)}, experts={len(names)}, loss_scale={scale:.3f}")

    print(f"[{EXP}] running aggregators ...")
    results = run_suite(
        P, y, loss_scale=scale, alpha=0.001,
        record_weights=True, snapshot_every=10,
    )
    table = benchmark_table(results, P, y, names)

    L = expert_loss_matrix(P, y)
    best_idx, best_mean = best_fixed_expert(L)
    oracle = oracle_switching_loss(L, segments)
    table["oracle_switching_mean_loss"] = oracle

    # レジームごとの最良 Expert
    per_regime = []
    for (s, e), (label, _) in zip(segments, REGIMES):
        block = L[s:e].mean(axis=0)
        i = int(np.argmin(block))
        per_regime.append(
            {
                "regime": label,
                "best_expert": names[i],
                "mean_loss": float(block[i]),
                "best_fixed_expert_loss_here": float(block[best_idx]),
            }
        )
    table["per_regime"] = per_regime
    table["regime_bounds"] = eval_bounds
    table["warmup"] = WARMUP
    table["loss_scale"] = scale

    for row in per_regime:
        print(f"    {row['regime']:24s} best = {row['best_expert']}")
    print(f"    best fixed expert overall = {names[best_idx]} (MAE {best_mean:.3f})")
    print(f"    oracle switching MAE      = {oracle:.3f}")
    for r in table["algorithms"]:
        flag = "  <-- beats best fixed expert" if r["beats_best_fixed"] else ""
        print(f"    {r['algorithm']:20s} MAE {r['mean_loss']:.3f} "
              f"({r['vs_best_fixed_pct']:+.1f}%){flag}")

    # ------------------------------------------------------------------
    # 図1: 系列とレジーム
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4.0))
    ax.plot(series.to_numpy(), color=COLORS["Actual"], lw=0.5)
    for (s, e), (label, color), row in zip(
        zip(bounds[:-1], bounds[1:]), REGIMES, per_regime
    ):
        ax.axvspan(s, e, color=color, alpha=0.55, lw=0)
        ax.text((s + e) / 2, ax.get_ylim()[1], f"{label}\nbest: {row['best_expert']}",
                ha="center", va="top", fontsize=8.5)
    ax.set_xlim(0, len(series))
    ax.set_xlabel("time step (hours)")
    ax.set_ylabel("value")
    ax.set_title("Synthetic series: the winning expert changes with the regime")
    save_fig(fig, EXP, "series")

    # ------------------------------------------------------------------
    # 図2: regret 曲線
    # ------------------------------------------------------------------
    # Equal Weight は桁が違うので除外 (棒グラフの方で確認できる)
    fig = plot_regret_curves(
        results, P, y,
        title="Cumulative regret vs the best fixed expert (below 0 = better)",
        subset=[k for k in results if k != "Equal Weight"],
    )
    ax = fig.axes[0]
    for s, _ in segments[1:]:
        ax.axvline(s, color="#9E9E9E", lw=0.9, ls=":")
    save_fig(fig, EXP, "regret")

    # ------------------------------------------------------------------
    # 図3: 重み推移 (Hedge と Fixed-Share)
    # ------------------------------------------------------------------
    for key, fname in [
        ("Hedge (tuned eta)", "weights_hedge"),
        ("Fixed-Share", "weights_fixed_share"),
    ]:
        fig = plot_weight_evolution(
            results[key], names,
            title=f"{key}: expert weights over time (regime borders dotted)",
            top_k=8,
        )
        ax = fig.axes[0]
        for s, _ in segments[1:]:
            ax.axvline(s, color="#212121", lw=1.0, ls=":")
        save_fig(fig, EXP, fname)

    # ------------------------------------------------------------------
    # 図4: 平均損失の棒グラフ
    # ------------------------------------------------------------------
    fig = plot_mean_loss_bars(
        table, title="Mean absolute error on the regime-switching series"
    )
    ax = fig.axes[0]
    ax.axhline(oracle, color=COLORS["Oracle switching"], ls=":", lw=1.6,
               label="Oracle switching (best expert per regime)")
    ax.legend(fontsize=9)
    save_fig(fig, EXP, "bars")

    # ------------------------------------------------------------------
    # 図5: Fixed-Share の alpha 感度
    # ------------------------------------------------------------------
    alphas = [0.0, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]
    alpha_losses = []
    for a in alphas:
        r = run_online(P, y, FixedShare(P.shape[1], eta=1.0, alpha=a),
                       name=f"fs{a}", loss_scale=scale)
        alpha_losses.append(r.mean_loss)
    table["alpha_sensitivity"] = [
        {"alpha": a, "mean_loss": v} for a, v in zip(alphas, alpha_losses)
    ]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    xs = [a if a > 0 else 1e-6 for a in alphas]
    ax.plot(xs, alpha_losses, "o-", color=COLORS["Fixed-Share"], label="Fixed-Share")
    ax.axhline(best_mean, color=COLORS["Best fixed expert"], ls="--", lw=1.4,
               label=f"Best fixed expert ({names[best_idx]})")
    ax.axhline(oracle, color=COLORS["Oracle switching"], ls=":", lw=1.6,
               label="Oracle switching")
    ax.set_xscale("log")
    ax.set_xlabel("share parameter alpha  (leftmost point = 0, i.e. plain Hedge)")
    ax.set_ylabel("mean absolute error")
    ax.set_title("One knob: how much weight to hand back to forgotten experts")
    ax.legend(fontsize=9)
    save_fig(fig, EXP, "alpha_sensitivity")

    save_results(EXP, table)
    print(f"[{EXP}] done.")


if __name__ == "__main__":
    main()
