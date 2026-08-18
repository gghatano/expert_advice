"""実験3: ETTh1 — 水準がゆっくり動く実データでの追試.

ETT (Electricity Transformer Temperature) データセットの 1 時間粒度版。
2016-07 から 2018-06 までの 17,420 時点、7 変数。負荷関連の実測値で、
明確な季節性を持つ変数 (HUFL など) と、ゆっくり水準が動く変数 (OT =
油温) が混在しているので、「効く条件」を変数ごとに比較できる。

Usage:
    uv run python scripts/experiments/exp3_etth1.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import (  # noqa: E402
    ALGO_ORDER,
    COLORS,
    DATA_DIR,
    benchmark_table,
    plot_regret_curves,
    run_suite,
    save_fig,
    save_results,
    setup_style,
)

import matplotlib.pyplot as plt  # noqa: E402

from src.experts.vectorized import build_expert_matrix  # noqa: E402

EXP = "exp3_etth1"
WARMUP = 24 * 30 * 6   # 最初の約6か月を助走・学習に使う
HORIZON = 24           # 翌日同時刻を予測する (実務的な day-ahead 設定)


def load_etth1() -> pd.DataFrame:
    path = DATA_DIR / "ETTh1.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} が見つかりません。")
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    return df.astype(float)


def main() -> None:
    setup_style()
    print(f"[{EXP}] loading ETTh1 ...")
    df = load_etth1()
    print(f"[{EXP}] {len(df)} rows x {len(df.columns)} columns, "
          f"{df.index[0].date()} .. {df.index[-1].date()}")

    per_column = []
    focus_payload = None

    for col in df.columns:
        y_all = df[col].to_numpy(dtype=np.float64)
        names, P_all = build_expert_matrix(
            y_all, index=df.index, period=24, week=168,
            preset="light30", train_end=WARMUP, horizon=HORIZON,
        )
        P, y = P_all[WARMUP:], y_all[WARMUP:]
        scale = float(np.mean(np.abs(np.diff(y_all[:WARMUP]))))
        if not np.isfinite(scale) or scale <= 0:
            print(f"    {col}: skipped (degenerate scale)")
            continue

        results = run_suite(
            P, y, loss_scale=scale,
            record_weights=(col == "OT"), snapshot_every=24,
        )
        table = benchmark_table(results, P, y, names)
        table["column"] = col
        per_column.append(table)

        best = table["best_fixed_expert"]
        winners = [r["algorithm"] for r in table["algorithms"] if r["beats_best_fixed"]]
        print(f"    {col:5s} best fixed expert = {best['name']:18s} "
              f"MAE {best['mean_loss']:.4f} | beating it: "
              f"{', '.join(winners) if winners else 'none'}")

        if col == "OT":
            focus_payload = (results, P, y, names, table)

    # ------------------------------------------------------------------
    # 図1: 変数ごとの「最良固定 Expert 比」
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    width = 0.11
    xs = np.arange(len(per_column))
    for j, algo in enumerate(ALGO_ORDER):
        vals = []
        for table in per_column:
            row = next(r for r in table["algorithms"] if r["algorithm"] == algo)
            vals.append(row["vs_best_fixed_pct"])
        ax.bar(xs + (j - len(ALGO_ORDER) / 2) * width, vals, width,
               label=algo, color=COLORS.get(algo, "#607D8B"))
    ax.axhline(0, color="#212121", lw=1.2)
    ax.set_xticks(xs)
    ax.set_xticklabels([t["column"] for t in per_column])
    ax.set_ylabel("MAE vs best fixed expert  [%]  (negative = better)")
    ax.set_title(f"ETTh1, {HORIZON}-hour-ahead forecasting: gap to the best fixed expert")
    ax.legend(fontsize=8, ncol=4)
    save_fig(fig, EXP, "per_column")

    # ------------------------------------------------------------------
    # 図2: OT 列の regret 曲線
    # ------------------------------------------------------------------
    if focus_payload is not None:
        results, P, y, names, table = focus_payload
        fig = plot_regret_curves(
            results, P, y,
            title="ETTh1 / OT (oil temperature): cumulative regret vs the best fixed expert",
            xlabel="time step (hours)",
            subset=[k for k in results if k != "Equal Weight"],
        )
        save_fig(fig, EXP, "regret_ot")

    save_results(EXP, {
        "horizon": HORIZON,
        "warmup": WARMUP,
        "columns": per_column,
    })
    print(f"[{EXP}] done.")


if __name__ == "__main__":
    main()
