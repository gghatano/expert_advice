"""実験0: 本プロジェクトの既定設定を UCI 実データで再現し、なぜ退屈なのかを示す.

このリポジトリの既定は「1 ステップ先予測 + light30 + Meta-eta Hedge」。
UCI Electricity Load Diagrams 2011-2014 の実データでこれを回すと、
どの集約アルゴリズムも「事後的に最良の単一 Expert」とほぼ同じ成績になる。
つまり **アンサンブルした意味がほとんど無い**。

ここではその状態を実データで再現したうえで、設定を 1 つだけ変えた
比較 (翌日 = 24 時間先予測) を並べ、何が効いているのかを切り分ける。

Usage:
    uv run python scripts/experiments/exp0_uci_baseline.py [--n-series 30]
"""

from __future__ import annotations

import argparse
import random

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

from src.data.load_uci import load_electricity  # noqa: E402
from src.data.preprocess import load_processed, preprocess, save_processed  # noqa: E402
from src.ensemble.runner import best_fixed_expert, expert_loss_matrix  # noqa: E402
from src.experts.vectorized import build_expert_matrix  # noqa: E402

EXP = "exp0_uci_baseline"
CACHE = "electricity_sum.parquet"
TRAIN_END = pd.Timestamp("2014-01-01")   # Expert の学習はここまで (リポジトリの分割に準拠)
EVAL_START = pd.Timestamp("2014-07-01")  # test 期間
SETTINGS = [
    ("1-step ahead (repo default)", 1),
    ("24-hour ahead (day-ahead)", 24),
]


def load_uci_hourly() -> pd.DataFrame:
    """UCI データを 1 時間集約で読み込む (Parquet キャッシュあり)."""
    try:
        df = load_processed(name=CACHE)
        print(f"    cache hit: {CACHE}")
        return df
    except FileNotFoundError:
        pass

    path = DATA_DIR / "LD2011_2014.txt"
    if not path.exists():
        raise FileNotFoundError(f"{path} が見つかりません。")
    print("    parsing raw UCI text (this takes a minute) ...")
    raw = load_electricity(path=path)
    df = preprocess(raw, resample_method="sum", clip=False)
    save_processed(df, name=CACHE)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-series", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_style()
    print(f"[{EXP}] loading UCI electricity ...")
    df = load_uci_hourly()
    print(f"[{EXP}] {len(df)} hours x {len(df.columns)} series, "
          f"{df.index[0].date()} .. {df.index[-1].date()}")

    random.seed(args.seed)
    cols = random.sample(list(df.columns), min(args.n_series, len(df.columns)))

    idx = df.index
    train_end_pos = int(np.searchsorted(idx, TRAIN_END))
    eval_start_pos = int(np.searchsorted(idx, EVAL_START))

    payload_settings = []
    for label, horizon in SETTINGS:
        print(f"[{EXP}] setting: {label}")
        algo_rel: dict[str, list[float]] = {k: [] for k in ALGO_ORDER}
        best_names: list[str] = []
        n_used = 0

        for col in cols:
            y_all = df[col].to_numpy(dtype=np.float64)
            if not np.all(np.isfinite(y_all)) or np.std(y_all[:train_end_pos]) <= 0:
                continue
            scale = float(np.mean(np.abs(np.diff(y_all[:train_end_pos]))))
            if not np.isfinite(scale) or scale <= 0:
                continue

            names, P_all = build_expert_matrix(
                y_all, index=idx, period=24, week=168,
                preset="light30", train_end=train_end_pos, horizon=horizon,
            )
            P, y = P_all[eval_start_pos:], y_all[eval_start_pos:]

            results = run_suite(P, y, loss_scale=scale)
            L = expert_loss_matrix(P, y)
            b_idx, b_mae = best_fixed_expert(L)
            best_names.append(names[b_idx])
            for k in ALGO_ORDER:
                algo_rel[k].append(100.0 * (results[k].mean_loss - b_mae) / b_mae)
            n_used += 1

        rows = []
        for k in ALGO_ORDER:
            v = np.asarray(algo_rel[k])
            rows.append(
                {
                    "algorithm": k,
                    "median_vs_best_fixed_pct": float(np.median(v)),
                    "mean_vs_best_fixed_pct": float(v.mean()),
                    "win_rate": float(np.mean(v < 0)),
                }
            )
            print(f"    {k:20s} median {np.median(v):+7.2f}%  "
                  f"win rate {np.mean(v < 0):.0%}")

        winner_counts = pd.Series(best_names).value_counts()
        payload_settings.append(
            {
                "label": label,
                "horizon": horizon,
                "n_series": n_used,
                "algorithms": rows,
                "best_fixed_expert_counts": {k: int(v) for k, v in winner_counts.items()},
            }
        )

    # ------------------------------------------------------------------
    # 図: 2 つの設定の比較
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=False)
    for ax, s in zip(axes, payload_settings):
        vals = [r["median_vs_best_fixed_pct"] for r in s["algorithms"]]
        labels = [r["algorithm"] for r in s["algorithms"]]
        ax.bar(range(len(vals)), vals,
               color=[COLORS.get(l, "#607D8B") for l in labels])
        ax.axhline(0, color="#212121", lw=1.3)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8.5)
        ax.set_title(f"{s['label']}\n({s['n_series']} UCI series)", fontsize=11)
        ax.set_ylabel("median MAE vs best fixed expert [%]")
        # Equal Weight は桁が違うので表示範囲を絞る
        finite = [v for l, v in zip(labels, vals) if l != "Equal Weight"]
        ax.set_ylim(min(finite + [0]) * 1.4 - 1, max(finite + [0]) * 1.4 + 1)
    fig.suptitle("UCI Electricity: what the current setup produces, and what one change does",
                 fontsize=12)
    save_fig(fig, EXP, "settings_comparison")

    save_results(EXP, {
        "train_end": str(TRAIN_END.date()),
        "eval_start": str(EVAL_START.date()),
        "requested_series": len(cols),
        "seed": args.seed,
        "note": "Equal Weight is far off-scale in the figure; see the table for its value.",
        "settings": payload_settings,
    })
    print(f"[{EXP}] done.")


if __name__ == "__main__":
    main()
