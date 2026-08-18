"""実験2: M4 Hourly — 「最良の Expert は系列ごとに違う」ので集約が勝つ.

M4 コンペティションの hourly 部門 (414 系列) を 1 ステップ先オンライン予測で
解く。狙いは合成データとは別の効き方の検証:

* どの Expert も全系列では勝てない (系列ごとに勝者が入れ替わる)
* したがって「事前に 1 本選ぶ」戦略は、平均するとどうしても損をする
* 集約は系列ごとに勝手に良い Expert を見つけるので、
  「全系列で見た最良の単一 Expert」を安定して上回る

系列ごとにスケールが違うので、指標は各系列の助走区間における
naive 予測 MAE で割った **scaled MAE** で揃える。

Usage:
    uv run python scripts/experiments/exp2_m4.py [--limit N]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common import (  # noqa: E402
    ALGO_ORDER,
    COLORS,
    DATA_DIR,
    run_suite,
    save_fig,
    save_results,
    setup_style,
)

import matplotlib.pyplot as plt  # noqa: E402

from src.ensemble.runner import expert_loss_matrix  # noqa: E402
from src.experts.vectorized import build_expert_matrix  # noqa: E402

EXP = "exp2_m4"
WARMUP = 168  # 1 週間を助走に使う (学習系 Expert もここまでしか見ない)
MIN_LENGTH = 400


def load_m4_hourly(limit: int | None = None) -> list[tuple[str, np.ndarray]]:
    """M4 Hourly の学習系列を読み込む."""
    path = DATA_DIR / "M4-Hourly-train.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} が見つかりません。README の手順でダウンロードしてください。"
        )
    df = pd.read_csv(path)
    id_col = df.columns[0]
    series: list[tuple[str, np.ndarray]] = []
    for _, row in df.iterrows():
        values = row.drop(labels=[id_col]).to_numpy(dtype=np.float64)
        values = values[np.isfinite(values)]
        if len(values) >= MIN_LENGTH:
            series.append((str(row[id_col]), values))
        if limit is not None and len(series) >= limit:
            break
    return series


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="使用する系列数の上限")
    args = parser.parse_args()

    setup_style()
    print(f"[{EXP}] loading M4 hourly ...")
    series = load_m4_hourly(limit=args.limit)
    print(f"[{EXP}] {len(series)} series loaded")

    expert_names: list[str] | None = None
    algo_scaled: dict[str, list[float]] = {k: [] for k in ALGO_ORDER}
    expert_scaled: list[np.ndarray] = []   # 系列 x Expert の scaled MAE
    per_series_best: list[str] = []
    series_ids: list[str] = []

    for i, (sid, y_all) in enumerate(series):
        names, P_all = build_expert_matrix(
            y_all, index=None, period=24, week=168,
            preset="light30", train_end=WARMUP,
        )
        if expert_names is None:
            expert_names = names

        P, y = P_all[WARMUP:], y_all[WARMUP:]
        scale = float(np.mean(np.abs(np.diff(y_all[:WARMUP]))))
        if not np.isfinite(scale) or scale <= 0:
            continue

        results = run_suite(P, y, loss_scale=scale)
        for k in ALGO_ORDER:
            algo_scaled[k].append(results[k].mean_loss / scale)

        L = expert_loss_matrix(P, y)
        means = L.mean(axis=0) / scale
        expert_scaled.append(means)
        per_series_best.append(names[int(np.argmin(means))])
        series_ids.append(sid)

        if (i + 1) % 25 == 0:
            print(f"    {i + 1}/{len(series)} series done")

    assert expert_names is not None
    E = np.vstack(expert_scaled)          # (n_series, n_experts)
    n_series = E.shape[0]

    # 「事前に 1 本だけ選ぶ」現実的なベースライン: 全系列平均で最良の Expert
    global_means = E.mean(axis=0)
    g_idx = int(np.argmin(global_means))
    global_best_name = expert_names[g_idx]
    global_best_score = float(global_means[g_idx])
    global_best_per_series = E[:, g_idx]

    # 系列ごとに最良を選べたら (事後オラクル)
    oracle_per_series = E.min(axis=1)
    oracle_score = float(oracle_per_series.mean())

    rows = []
    for k in ALGO_ORDER:
        v = np.asarray(algo_scaled[k])
        rows.append(
            {
                "algorithm": k,
                "mean_scaled_mae": float(v.mean()),
                "median_scaled_mae": float(np.median(v)),
                "vs_global_best_pct": 100.0 * (v.mean() - global_best_score) / global_best_score,
                "win_rate_vs_global_best": float(np.mean(v < global_best_per_series)),
                "beats_global_best": bool(v.mean() < global_best_score),
            }
        )

    winner_counts = pd.Series(per_series_best).value_counts()
    payload = {
        "n_series": n_series,
        "n_experts": len(expert_names),
        "warmup": WARMUP,
        "metric": "scaled MAE (divided by in-sample naive MAE)",
        "global_best_expert": {"name": global_best_name, "mean_scaled_mae": global_best_score},
        "per_series_oracle_mean_scaled_mae": oracle_score,
        "expert_ranking": [
            {"name": expert_names[i], "mean_scaled_mae": float(global_means[i])}
            for i in np.argsort(global_means)[:10]
        ],
        "winner_counts": {k: int(v) for k, v in winner_counts.items()},
        "n_distinct_winners": int(winner_counts.size),
        "top_winner_share": float(winner_counts.iloc[0] / n_series),
        "algorithms": rows,
    }

    print(f"    global best single expert = {global_best_name} "
          f"(scaled MAE {global_best_score:.4f})")
    print(f"    per-series oracle         = {oracle_score:.4f}")
    print(f"    distinct per-series winners = {winner_counts.size} "
          f"(top share {winner_counts.iloc[0] / n_series:.1%})")
    for r in rows:
        flag = "  <-- beats it" if r["beats_global_best"] else ""
        print(f"    {r['algorithm']:20s} {r['mean_scaled_mae']:.4f} "
              f"({r['vs_global_best_pct']:+.1f}%, win rate "
              f"{r['win_rate_vs_global_best']:.0%}){flag}")

    # ------------------------------------------------------------------
    # 図1: 系列ごとの勝者の分布
    # ------------------------------------------------------------------
    top = winner_counts.head(15)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.barh(range(len(top))[::-1], top.to_numpy(), color="#5C6BC0")
    ax.set_yticks(range(len(top))[::-1])
    ax.set_yticklabels(top.index)
    ax.set_xlabel(f"number of series where this expert is the best (out of {n_series})")
    ax.set_title("No single expert dominates: the winner changes from series to series")
    save_fig(fig, EXP, "winner_hist")

    # ------------------------------------------------------------------
    # 図2: アルゴリズム比較
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4.6))
    values = [r["mean_scaled_mae"] for r in rows]
    bars = ax.bar(range(len(rows)), values,
                  color=[COLORS.get(r["algorithm"], "#607D8B") for r in rows])
    ax.axhline(global_best_score, color=COLORS["Best fixed expert"], ls="--", lw=1.4,
               label=f"Best single expert picked in advance ({global_best_name})")
    ax.axhline(oracle_score, color=COLORS["Oracle switching"], ls=":", lw=1.6,
               label="Oracle: best expert per series (post hoc)")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r["algorithm"] for r in rows], rotation=20, ha="right")
    ax.set_ylabel("mean scaled MAE (lower is better)")
    ax.set_title(f"M4 Hourly: {n_series} series, 1-step-ahead online forecasting")
    ax.legend(fontsize=9)
    lo, hi = min(values + [oracle_score]), max(values + [global_best_score])
    pad = (hi - lo) * 0.25 + 1e-9
    ax.set_ylim(max(0, lo - pad), hi + pad)
    for b, v in zip(bars, values):
        ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=8)
    save_fig(fig, EXP, "bars")

    # ------------------------------------------------------------------
    # 図3: 系列ごとの勝敗 (勝率)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4.2))
    wr = [r["win_rate_vs_global_best"] for r in rows]
    ax.bar(range(len(rows)), wr,
           color=[COLORS.get(r["algorithm"], "#607D8B") for r in rows])
    ax.axhline(0.5, color="#212121", ls="--", lw=1.2, label="50% (coin flip)")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r["algorithm"] for r in rows], rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("share of series beaten")
    ax.set_title(f"Per-series win rate against the best single expert ({global_best_name})")
    ax.legend(fontsize=9)
    save_fig(fig, EXP, "win_rate")

    save_results(EXP, payload)
    print(f"[{EXP}] done.")


if __name__ == "__main__":
    main()
