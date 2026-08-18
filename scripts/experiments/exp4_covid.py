"""実験4: COVID-19 ロックダウン — 実世界で構造変化が起きたときの電力需要予測.

2020 年 3 月のロックダウンで、欧州各国の電力需要は水準も日内形状も
一気に変わった。これは「事前に選んで固定した予測モデルが壊れる」
実世界の代表例で、適応的な集約手法が有効だったことが報告されている
(Obst, de Vilmarest & Goude 2021 など)。

ここでは Open Power System Data の 1 時間粒度実測需要を使い、

  * Expert は 2015-2018 のデータだけで学習 (COVID を知らない)
  * 2019-01 以降を翌日 (24 時間先) 予測でオンライン評価
  * 「ロックダウン前の実績で選んだ最良の 1 本」を固定し続けた場合と、
    集約アルゴリズムを比べる

という、実務でそのまま起きる比較を行う。

Usage:
    uv run python scripts/experiments/exp4_covid.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import (  # noqa: E402
    ALGO_ORDER,
    COLORS,
    DATA_DIR,
    plot_weight_evolution,
    run_suite,
    save_fig,
    save_results,
    setup_style,
)

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.dates as mdates  # noqa: E402

from src.ensemble.runner import best_fixed_expert, expert_loss_matrix  # noqa: E402
from src.experts.vectorized import build_expert_matrix  # noqa: E402

EXP = "exp4_covid"
COUNTRIES = {"FR": "France", "ES": "Spain", "IT": "Italy", "DE": "Germany"}
FOCUS = "FR"

TRAIN_END = pd.Timestamp("2019-01-01", tz="UTC")   # Expert の学習はここまで
EVAL_START = TRAIN_END
LOCKDOWN = pd.Timestamp("2020-03-16", tz="UTC")    # 欧州主要国のロックダウン開始週
EVAL_END = pd.Timestamp("2020-09-30", tz="UTC")
HORIZON = 24


def load_opsd() -> pd.DataFrame:
    """OPSD の 1 時間粒度実測需要から対象国を取り出す."""
    path = DATA_DIR / "opsd_time_series_60min.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} が見つかりません。")
    cols = {f"{c}_load_actual_entsoe_transparency": c for c in COUNTRIES}
    df = pd.read_csv(
        path,
        usecols=["utc_timestamp", *cols],
        parse_dates=["utc_timestamp"],
    ).set_index("utc_timestamp").rename(columns=cols)
    df = df.loc[:EVAL_END]
    # 短い欠損は線形補間、それでも残る行は落とす
    df = df.interpolate(limit=6, limit_direction="both").dropna()
    return df.astype(float)


def main() -> None:
    setup_style()
    print(f"[{EXP}] loading OPSD (this reads a 130 MB csv) ...")
    df = load_opsd()
    print(f"[{EXP}] {len(df)} hours, {df.index[0].date()} .. {df.index[-1].date()}, "
          f"countries: {list(df.columns)}")

    idx = df.index
    train_end_pos = int(np.searchsorted(idx, TRAIN_END))
    eval_start_pos = train_end_pos
    lockdown_pos_global = int(np.searchsorted(idx, LOCKDOWN))
    lockdown_pos = lockdown_pos_global - eval_start_pos

    per_country = []
    focus_payload = None

    for country in df.columns:
        y_all = df[country].to_numpy(dtype=np.float64)
        names, P_all = build_expert_matrix(
            y_all, index=idx, period=24, week=168,
            preset="light30", train_end=train_end_pos, horizon=HORIZON,
        )
        P, y = P_all[eval_start_pos:], y_all[eval_start_pos:]
        eval_idx = idx[eval_start_pos:]
        scale = float(np.mean(np.abs(np.diff(y_all[:train_end_pos]))))

        results = run_suite(
            P, y, loss_scale=scale,
            record_weights=(country == FOCUS), snapshot_every=24,
        )

        L = expert_loss_matrix(P, y)
        pre = slice(0, lockdown_pos)
        cov = slice(lockdown_pos, len(y))

        # ロックダウン前の実績で 1 本選ぶ = 実務で普通にやること
        pre_means = L[pre].mean(axis=0)
        chosen_idx = int(np.argmin(pre_means))
        chosen_name = names[chosen_idx]
        chosen_pre = float(pre_means[chosen_idx])
        chosen_cov = float(L[cov, chosen_idx].mean())

        # 事後に見た最良固定 Expert (COVID 期間)
        best_cov_idx, best_cov_mae = best_fixed_expert(L[cov])

        rows = []
        for algo in ALGO_ORDER:
            res = results[algo]
            rows.append(
                {
                    "algorithm": algo,
                    "mae_pre": float(res.losses[pre].mean()),
                    "mae_covid": float(res.losses[cov].mean()),
                    "vs_chosen_pre_pct": 100.0 * (res.losses[pre].mean() - chosen_pre) / chosen_pre,
                    "vs_chosen_covid_pct": 100.0 * (res.losses[cov].mean() - chosen_cov) / chosen_cov,
                }
            )

        # 季節をそろえた比較: 3/16-6/30 の 2019 年 (平常) と 2020 年 (ロックダウン)。
        # 絶対 MAE は夏に下がるので、同じ季節同士でないと COVID の影響を測れない。
        def _win(a: str, b: str) -> slice:
            return slice(
                int(np.searchsorted(eval_idx, pd.Timestamp(a, tz="UTC"))),
                int(np.searchsorted(eval_idx, pd.Timestamp(b, tz="UTC"))),
            )

        w19, w20 = _win("2019-03-16", "2019-06-30"), _win("2020-03-16", "2020-06-30")
        seasonal = [
            {
                "name": f"Single expert picked before COVID ({chosen_name})",
                "mae_2019": float(L[w19, chosen_idx].mean()),
                "mae_2020": float(L[w20, chosen_idx].mean()),
            }
        ] + [
            {
                "name": algo,
                "mae_2019": float(results[algo].losses[w19].mean()),
                "mae_2020": float(results[algo].losses[w20].mean()),
            }
            for algo in ALGO_ORDER
        ]
        for s in seasonal:
            s["change_pct"] = 100.0 * (s["mae_2020"] - s["mae_2019"]) / s["mae_2019"]

        # 個々の Expert の前年同期比較。ロックダウンで壊れるのは
        # 「暦だけを見る、事前学習した」Expert のはずで、直近ラグを使う
        # Expert は自動的に新しい水準に追随してしまう — その切り分け。
        watch = [
            "SeasonalProfile", "SeasonalNaive_168", "SeasonalNaive_24",
            "RidgeLag_100.0", "LastValue", "SMA_24",
        ]
        expert_seasonal = []
        for w in watch:
            if w not in names:
                continue
            j = names.index(w)
            a, b = float(L[w19, j].mean()), float(L[w20, j].mean())
            expert_seasonal.append(
                {"name": w, "mae_2019": a, "mae_2020": b,
                 "change_pct": 100.0 * (b - a) / a}
            )

        entry = {
            "country": country,
            "country_name": COUNTRIES[country],
            "seasonal_matched_spring": seasonal,
            "seasonal_matched_experts": expert_seasonal,
            "n_eval_hours": int(len(y)),
            "lockdown_index": int(lockdown_pos),
            "expert_chosen_before_covid": {
                "name": chosen_name,
                "mae_pre": chosen_pre,
                "mae_covid": chosen_cov,
                "degradation_pct": 100.0 * (chosen_cov - chosen_pre) / chosen_pre,
            },
            "best_fixed_expert_during_covid": {
                "name": names[best_cov_idx],
                "mae_covid": float(best_cov_mae),
            },
            "algorithms": rows,
        }
        per_country.append(entry)

        best_algo = min(rows, key=lambda r: r["mae_covid"])
        print(f"    {country}: pre-COVID pick = {chosen_name:18s} "
              f"MAE {chosen_pre:.0f} -> {chosen_cov:.0f} MW "
              f"({entry['expert_chosen_before_covid']['degradation_pct']:+.0f}%) | "
              f"best algo during COVID = {best_algo['algorithm']} "
              f"({best_algo['vs_chosen_covid_pct']:+.1f}%)")

        if country == FOCUS:
            focus_payload = (results, P, y, names, eval_idx, L, chosen_idx, chosen_name)

    # ------------------------------------------------------------------
    # 図1: 需要系列とロックダウン
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4.0))
    show = df.loc["2019-01-01":][FOCUS]
    ax.plot(show.index, show.to_numpy(), color=COLORS["Actual"], lw=0.4)
    ax.axvline(LOCKDOWN, color="#E53935", lw=1.6)
    ax.annotate("lockdown\n2020-03-16", (LOCKDOWN, show.max()),
                xytext=(12, -6), textcoords="offset points",
                color="#E53935", fontsize=9, va="top")
    ax.set_ylabel("load [MW]")
    ax.set_title(f"{COUNTRIES[FOCUS]}: hourly electricity demand around the COVID-19 shock")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    save_fig(fig, EXP, "load_series")

    assert focus_payload is not None
    results, P, y, names, eval_idx, L, chosen_idx, chosen_name = focus_payload

    # ------------------------------------------------------------------
    # 図2: 14 日移動平均 MAE
    # ------------------------------------------------------------------
    win = 24 * 14
    fig, ax = plt.subplots(figsize=(10, 4.6))
    # Hedge 系の曲線は事前選択 Expert とほぼ完全に重なるので、
    # 代表として Meta-eta Hedge だけを描き、基準線は最後に上書きする。
    for algo in ["Fixed-Share", "Meta-eta Hedge"]:
        roll = pd.Series(results[algo].losses, index=eval_idx).rolling(win).mean()
        ax.plot(roll.index, roll.to_numpy(), color=COLORS[algo], lw=1.5, label=algo)
    chosen_roll = pd.Series(L[:, chosen_idx], index=eval_idx).rolling(win).mean()
    ax.plot(chosen_roll.index, chosen_roll.to_numpy(), color="#212121", lw=1.6,
            ls="--", zorder=5,
            label=f"Single expert picked before COVID ({chosen_name})")
    ax.axvline(LOCKDOWN, color="#E53935", lw=1.4, ls="--", label="lockdown")
    ax.set_ylabel("14-day rolling MAE [MW]")
    ax.set_title(f"{COUNTRIES[FOCUS]}: day-ahead forecast error before and after the shock")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend(fontsize=9)
    save_fig(fig, EXP, "rolling_mae")

    # ------------------------------------------------------------------
    # 図3: ロックダウン前に選んだ Expert に対する累積 regret
    # ------------------------------------------------------------------
    baseline = np.cumsum(L[:, chosen_idx])
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for algo in ALGO_ORDER:
        if algo == "Equal Weight":
            continue
        regret = results[algo].cumulative_loss - baseline
        ax.plot(eval_idx, regret, color=COLORS[algo], lw=1.5, label=algo)
    ax.axhline(0, color="#212121", ls="--", lw=1.4,
               label=f"Single expert picked before COVID ({chosen_name})")
    ax.axvline(LOCKDOWN, color="#E53935", lw=1.4, ls="--", label="lockdown")
    ax.set_ylabel("cumulative regret [MW-hours]")
    ax.set_title(f"{COUNTRIES[FOCUS]}: cumulative gain over the pre-COVID model choice")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend(fontsize=8, ncol=2)
    save_fig(fig, EXP, "regret")

    # ------------------------------------------------------------------
    # 図4: 重みの張り替え
    # ------------------------------------------------------------------
    fig = plot_weight_evolution(
        results["Fixed-Share"], names,
        title=f"{COUNTRIES[FOCUS]}: Fixed-Share reallocates weight after the lockdown",
        top_k=8, xlabel="time step (hours since 2019-01-01)",
    )
    fig.axes[0].axvline(int(np.searchsorted(eval_idx, LOCKDOWN)),
                        color="#E53935", lw=1.8, ls="--")
    save_fig(fig, EXP, "weights")

    # ------------------------------------------------------------------
    # 図5: 季節をそろえた春 (3/16-6/30) の比較
    # ------------------------------------------------------------------
    focus_entry = next(e for e in per_country if e["country"] == FOCUS)
    seasonal = focus_entry["seasonal_matched_spring"]
    keep = ["Fixed-Share", "ML-Poly", "AdaHedge", "Meta-eta Hedge", "Follow the Leader"]
    shown = [seasonal[0]] + [s for s in seasonal if s["name"] in keep]
    labels = ["pre-COVID pick\n" + chosen_name] + [s["name"] for s in shown[1:]]

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    xs = np.arange(len(shown))
    ax.bar(xs - 0.2, [s["mae_2019"] for s in shown], 0.4,
           label="spring 2019 (normal)", color="#90A4AE")
    ax.bar(xs + 0.2, [s["mae_2020"] for s in shown], 0.4,
           label="spring 2020 (lockdown)", color="#E53935")
    for x, s in zip(xs, shown):
        ax.annotate(f"{s['change_pct']:+.0f}%", (x, max(s["mae_2019"], s["mae_2020"])),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=9)
    ax.set_ylabel("MAE [MW], 16 Mar - 30 Jun")
    ax.set_title(f"{COUNTRIES[FOCUS]}: same season, one year apart")
    ax.legend(fontsize=9)
    save_fig(fig, EXP, "seasonal_matched")

    # ------------------------------------------------------------------
    # 図6: どの Expert がロックダウンで壊れたか
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    es = focus_entry["seasonal_matched_experts"]
    xs = np.arange(len(es))
    ax.bar(xs - 0.2, [e["mae_2019"] for e in es], 0.4,
           label="spring 2019 (normal)", color="#90A4AE")
    ax.bar(xs + 0.2, [e["mae_2020"] for e in es], 0.4,
           label="spring 2020 (lockdown)", color="#E53935")
    for x, e in zip(xs, es):
        ax.annotate(f"{e['change_pct']:+.0f}%", (x, max(e["mae_2019"], e["mae_2020"])),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels([e["name"] for e in es], rotation=18, ha="right", fontsize=9)
    ax.set_ylabel("MAE [MW], 16 Mar - 30 Jun")
    ax.set_title(f"{COUNTRIES[FOCUS]}: only the calendar-only expert breaks in 2020")
    ax.legend(fontsize=9)
    save_fig(fig, EXP, "expert_breakage")

    save_results(EXP, {
        "horizon": HORIZON,
        "train_end": str(TRAIN_END.date()),
        "eval_start": str(EVAL_START.date()),
        "lockdown": str(LOCKDOWN.date()),
        "eval_end": str(EVAL_END.date()),
        "focus_country": FOCUS,
        "countries": per_country,
    })
    print(f"[{EXP}] done.")


if __name__ == "__main__":
    main()
