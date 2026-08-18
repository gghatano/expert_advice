"""docs/*.html の表を docs/results/*.json から再生成する.

HTML 側には

    <!-- AUTO:key -->  ... 自動生成される中身 ...  <!-- /AUTO:key -->

というマーカーを置いておく。本スクリプトは ``key`` に対応するレンダラを
呼び出し、マーカーの内側だけを差し替える。散文は HTML に、数値は JSON に、
という分担にすることで、実験を回し直したときに数値の転記漏れが起きない。

Usage:
    uv run python scripts/build_site.py
    uv run python scripts/build_site.py --check   # 差分があれば非ゼロ終了
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Callable

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS = _PROJECT_ROOT / "docs"
RESULTS = DOCS / "results"

PAGES = ["index.html", "algorithm.html", "experiment.html", "replication.html"]

_MARKER = re.compile(
    r"(<!--\s*AUTO:(?P<key>[\w.]+)\s*-->)(?P<body>.*?)(<!--\s*/AUTO:(?P=key)\s*-->)",
    re.DOTALL,
)


# ---------------------------------------------------------------------------
# 書式ヘルパ
# ---------------------------------------------------------------------------


def pct(value: float, digits: int = 2) -> str:
    """符号つきパーセント表記."""
    return f"{value:+.{digits}f}%"


def verdict_cell(value: float, digits: int = 2) -> str:
    """負なら勝ち (緑)、正なら負け (灰) の td を返す."""
    cls = "win" if value < 0 else "lose"
    return f'<td class="num {cls}">{pct(value, digits)}</td>'


def num(value: float, digits: int = 3) -> str:
    return f'<td class="num">{value:.{digits}f}</td>'


def table(caption: str, headers: list[str], rows: list[str], *, note: str = "") -> str:
    """table-wrap で囲んだ表を組み立てる."""
    # 先頭が "*" の見出しは数値列 (右寄せ) として扱う
    head = "".join(
        (f'<th class="num">{h[1:]}</th>' if h.startswith("*") else f"<th>{h}</th>")
        for h in headers
    )
    body = "\n      ".join(rows)
    extra = f'\n<p class="footnote">{note}</p>' if note else ""
    return (
        '\n<div class="table-wrap">\n'
        "<table>\n"
        f"  <caption>{caption}</caption>\n"
        f"  <thead><tr>{head}</tr></thead>\n"
        "  <tbody>\n"
        f"      {body}\n"
        "  </tbody>\n"
        "</table>\n"
        f"</div>{extra}\n"
    )


def load(name: str) -> dict:
    path = RESULTS / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} がありません。先に scripts/experiments/{name}.py を実行してください。"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _row_of(rows: list[dict], algorithm: str) -> dict:
    return next(r for r in rows if r["algorithm"] == algorithm)


# ---------------------------------------------------------------------------
# レンダラ
# ---------------------------------------------------------------------------


def exp0_nseries() -> str:
    d = load("exp0_uci_baseline")
    return str(d["settings"][0]["n_series"])


def exp0_table() -> str:
    d = load("exp0_uci_baseline")
    one, day = d["settings"][0], d["settings"][1]
    rows = []
    for r1 in one["algorithms"]:
        r24 = _row_of(day["algorithms"], r1["algorithm"])
        highlight = ' class="row-highlight"' if r1["algorithm"] == "Fixed-Share" else ""
        rows.append(
            f"<tr{highlight}><td>{r1['algorithm']}</td>"
            + verdict_cell(r1["median_vs_best_fixed_pct"])
            + f'<td class="num">{r1["win_rate"]:.0%}</td>'
            + verdict_cell(r24["median_vs_best_fixed_pct"])
            + f'<td class="num">{r24["win_rate"]:.0%}</td></tr>'
        )
    return table(
        f"UCI Electricity {one['n_series']} 系列。"
        "「最良固定 Expert 比」は各系列での差の中央値、「勝率」は下回った系列の割合。負の値が勝ち。",
        ["アルゴリズム", "*1 時間先：最良固定 Expert 比", "*勝率",
         "*24 時間先：最良固定 Expert 比", "*勝率"],
        rows,
        note="Equal Weight が大きく劣るのは、急変時にまったく当たらない Expert "
             "（長窓の移動平均など）を等しく信用してしまうためです。",
    )


def _exp0_fs(setting_idx: int, field: str) -> str:
    d = load("exp0_uci_baseline")
    r = _row_of(d["settings"][setting_idx]["algorithms"], "Fixed-Share")
    if field == "median":
        return pct(r["median_vs_best_fixed_pct"], 1)
    return f"{r['win_rate']:.0%}"


def exp0_fs_1step() -> str:
    return _exp0_fs(0, "median")


def exp0_fs_1step_win() -> str:
    return _exp0_fs(0, "win")


def exp0_fs_24step() -> str:
    return _exp0_fs(1, "median")


def exp0_fs_24step_win() -> str:
    return _exp0_fs(1, "win")


def exp1_regimes() -> str:
    d = load("exp1_regime")
    best_fixed = d["best_fixed_expert"]["name"]
    rows = []
    for r in d["per_regime"]:
        rows.append(
            f"<tr><td>{r['regime']}</td><td><strong>{r['best_expert']}</strong></td>"
            + num(r["mean_loss"])
            + num(r["best_fixed_expert_loss_here"])
            + "</tr>"
        )
    return table(
        "レジームごとの最良 Expert（合成データ）。4 区間で 4 種類の勝者に分かれています。",
        ["レジーム", "その区間で最良の Expert", "*その MAE",
         f"*{best_fixed} の MAE"],
        rows,
        note=f"全区間を通した最良の固定 Expert は {best_fixed} ですが、"
             "どの区間でも一番にはなれていません。これが「乗り換えれば勝てる」余地です。",
    )


def exp1_table() -> str:
    d = load("exp1_regime")
    best = d["best_fixed_expert"]
    rows = []
    for r in d["algorithms"]:
        highlight = ' class="row-highlight"' if r["beats_best_fixed"] and r["vs_best_fixed_pct"] < -1 else ""
        rows.append(
            f"<tr{highlight}><td>{r['algorithm']}</td>"
            + num(r["mean_loss"])
            + verdict_cell(r["vs_best_fixed_pct"], 1)
            + "</tr>"
        )
    rows.append(
        f'<tr><td>— 最良固定 Expert（{best["name"]}、事後）</td>'
        + num(best["mean_loss"])
        + '<td class="num">0.0%</td></tr>'
    )
    rows.append(
        "<tr><td>— 切り替えオラクル（区間ごとに最良、事後）</td>"
        + num(d["oracle_switching_mean_loss"])
        + verdict_cell(
            100.0 * (d["oracle_switching_mean_loss"] - best["mean_loss"]) / best["mean_loss"], 1
        )
        + "</tr>"
    )
    return table(
        f"レジーム切替データ（{d['n_steps']:,} 時点、Expert {d['n_experts']} 本）の平均絶対誤差。",
        ["アルゴリズム", "*平均 MAE", "*最良固定 Expert 比"],
        rows,
        note="Fixed-Share は切り替えオラクル（事後に区間ごとの最良を選べた場合）に"
             "かなり近いところまで到達しています。",
    )


def exp2_winners() -> str:
    d = load("exp2_m4")
    return (
        f"{d['n_series']} 系列に対して <strong>{d['n_distinct_winners']} 種類</strong>の Expert が"
        f"勝者になり、最多のものでも全体の {d['top_winner_share']:.0%} にとどまります"
    )


def exp2_table() -> str:
    d = load("exp2_m4")
    g = d["global_best_expert"]
    rows = []
    for r in d["algorithms"]:
        highlight = ' class="row-highlight"' if r["algorithm"] == "Fixed-Share" else ""
        rows.append(
            f"<tr{highlight}><td>{r['algorithm']}</td>"
            + num(r["mean_scaled_mae"], 4)
            + verdict_cell(r["vs_global_best_pct"], 1)
            + f'<td class="num">{r["win_rate_vs_global_best"]:.0%}</td></tr>'
        )
    rows.append(
        f'<tr><td>— 事前に選んだ最良の 1 本（{g["name"]}）</td>'
        + num(g["mean_scaled_mae"], 4)
        + '<td class="num">0.0%</td><td class="num">—</td></tr>'
    )
    rows.append(
        "<tr><td>— 系列ごとに最良を選べた場合（事後オラクル）</td>"
        + num(d["per_series_oracle_mean_scaled_mae"], 4)
        + verdict_cell(
            100.0 * (d["per_series_oracle_mean_scaled_mae"] - g["mean_scaled_mae"])
            / g["mean_scaled_mae"], 1
        )
        + '<td class="num">—</td></tr>'
    )
    return table(
        f"M4 Hourly {d['n_series']} 系列、1 ステップ先オンライン予測。"
        "指標は各系列の naive MAE で割った scaled MAE（小さいほど良い）。",
        ["アルゴリズム", "*scaled MAE", "*事前選択比", "*勝率"],
        rows,
        note="Fixed-Share は事後オラクル（系列ごとに最良の 1 本を選べた場合）も下回りました。"
             "系列内でも重みを動かせるためです。",
    )


def exp3_table() -> str:
    d = load("exp3_etth1")
    algos = ["Meta-eta Hedge", "AdaHedge", "ML-Poly", "Fixed-Share"]
    rows = []
    for c in d["columns"]:
        cells = "".join(
            verdict_cell(_row_of(c["algorithms"], a)["vs_best_fixed_pct"], 1) for a in algos
        )
        rows.append(
            f"<tr><td><strong>{c['column']}</strong></td>"
            f"<td>{c['best_fixed_expert']['name']}</td>"
            + num(c["best_fixed_expert"]["mean_loss"], 3)
            + cells
            + "</tr>"
        )
    return table(
        f"ETTh1 の 7 変数、{d['horizon']} 時間先予測。数値は最良固定 Expert に対する MAE の差。負が勝ち。",
        ["変数", "最良固定 Expert", "*その MAE", "*Meta-η Hedge", "*AdaHedge",
         "*ML-Poly", "*Fixed-Share"],
        rows,
        note="変数によって最良 Expert が Ridge 系だったり EMA 系だったりと入れ替わります。",
    )


def exp4_table() -> str:
    d = load("exp4_covid")
    algos = ["Meta-eta Hedge", "AdaHedge", "ML-Poly", "Fixed-Share"]
    rows = []
    for c in d["countries"]:
        chosen = c["expert_chosen_before_covid"]
        cells = "".join(
            verdict_cell(_row_of(c["algorithms"], a)["vs_chosen_covid_pct"], 1) for a in algos
        )
        rows.append(
            f"<tr><td><strong>{c['country_name']}</strong></td>"
            f"<td>{chosen['name']}</td>"
            + f'<td class="num">{chosen["mae_covid"]:,.0f}</td>'
            + cells
            + "</tr>"
        )
    return table(
        f"{d['lockdown']} 以降（ロックダウン期）の翌日予測 MAE [MW]。"
        "数値は「ロックダウン前の実績で選んだ 1 本」に対する差。負が勝ち。",
        ["国", "事前に選ばれた Expert", "*その MAE [MW]", "*Meta-η Hedge",
         "*AdaHedge", "*ML-Poly", "*Fixed-Share"],
        rows,
        note="Hedge 系がすべて ±0.0% なのは、重みが事実上その 1 本に集中しきっているためです。",
    )


def exp4_experts() -> str:
    d = load("exp4_covid")
    focus = next(c for c in d["countries"] if c["country"] == d["focus_country"])
    rows = []
    for e in focus["seasonal_matched_experts"]:
        cls = "lose" if e["change_pct"] > 0 else "win"
        style = ' class="row-highlight"' if e["change_pct"] > 20 else ""
        rows.append(
            f"<tr{style}><td>{e['name']}</td>"
            + f'<td class="num">{e["mae_2019"]:,.0f}</td>'
            + f'<td class="num">{e["mae_2020"]:,.0f}</td>'
            + f'<td class="num {cls}">{pct(e["change_pct"], 1)}</td></tr>'
        )
    return table(
        f"{focus['country_name']}：季節をそろえた Expert 単体の比較（3/16〜6/30 の MAE [MW]）。",
        ["Expert", "*2019 年春", "*2020 年春（ロックダウン）", "*変化"],
        rows,
        note="直近の観測値を使わない SeasonalProfile だけが大きく悪化しています。"
             "集約の価値は、こうした壊れたモデルを自動的に切り捨てられる点にあります。",
    )


def _fmt_minus(value: float, digits: int = 1) -> str:
    """本文用に、マイナス記号を全角ハイフンにした百分率を返す."""
    return f"{value:+.{digits}f}%".replace("-", "−")


def exp1_headline() -> str:
    d = load("exp1_regime")
    return _fmt_minus(_row_of(d["algorithms"], "Fixed-Share")["vs_best_fixed_pct"])


def exp2_headline() -> str:
    d = load("exp2_m4")
    return _fmt_minus(_row_of(d["algorithms"], "Fixed-Share")["vs_global_best_pct"])


def exp2_headline_win() -> str:
    d = load("exp2_m4")
    return f"{_row_of(d['algorithms'], 'Fixed-Share')['win_rate_vs_global_best']:.0%}"


def exp3_headline() -> str:
    d = load("exp3_etth1")
    n = len(d["columns"])
    wins = sum(
        1 for c in d["columns"]
        if _row_of(c["algorithms"], "Fixed-Share")["vs_best_fixed_pct"] < 0
    )
    return f"{n} 変数中 {wins} 変数で勝利" if wins < n else f"{n} 変数すべてで勝利"


def exp4_headline() -> str:
    d = load("exp4_covid")
    gains = [
        _row_of(c["algorithms"], "Fixed-Share")["vs_chosen_covid_pct"]
        for c in d["countries"]
    ]
    worst = max(gains)  # 最も改善が小さい国
    return f"{len(gains)} か国すべてで {_fmt_minus(worst, 0)} 以上"


def exp0_headline_range() -> str:
    """1 時間先と 24 時間先の Fixed-Share 改善幅を「7〜12%」の形で返す."""
    d = load("exp0_uci_baseline")
    vals = sorted(
        abs(_row_of(s["algorithms"], "Fixed-Share")["median_vs_best_fixed_pct"])
        for s in d["settings"]
    )
    return f"{vals[0]:.0f}〜{vals[-1]:.0f}%"


def exp0_headline_winrange() -> str:
    d = load("exp0_uci_baseline")
    vals = sorted(
        _row_of(s["algorithms"], "Fixed-Share")["win_rate"] for s in d["settings"]
    )
    return f"{vals[0]:.0%}〜{vals[-1]:.0%}"


RENDERERS: dict[str, Callable[[], str]] = {
    "exp0_headline_range": exp0_headline_range,
    "exp0_headline_winrange": exp0_headline_winrange,
    "exp1_headline": exp1_headline,
    "exp2_headline": exp2_headline,
    "exp2_headline_win": exp2_headline_win,
    "exp3_headline": exp3_headline,
    "exp4_headline": exp4_headline,
    "exp0_nseries": exp0_nseries,
    "exp0_table": exp0_table,
    "exp0_fs_1step": exp0_fs_1step,
    "exp0_fs_1step_win": exp0_fs_1step_win,
    "exp0_fs_24step": exp0_fs_24step,
    "exp0_fs_24step_win": exp0_fs_24step_win,
    "exp1_regimes": exp1_regimes,
    "exp1_table": exp1_table,
    "exp2_winners": exp2_winners,
    "exp2_table": exp2_table,
    "exp3_table": exp3_table,
    "exp4_table": exp4_table,
    "exp4_experts": exp4_experts,
}


# ---------------------------------------------------------------------------
# 実行
# ---------------------------------------------------------------------------


def render_page(text: str, used: set[str]) -> str:
    def repl(m: re.Match) -> str:
        key = m.group("key")
        if key not in RENDERERS:
            raise KeyError(f"未知のマーカー: AUTO:{key}")
        used.add(key)
        return m.group(1) + RENDERERS[key]() + m.group(4)

    return _MARKER.sub(repl, text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="書き換えず、差分があれば終了コード 1 を返す")
    args = parser.parse_args()

    used: set[str] = set()
    changed: list[str] = []

    for name in PAGES:
        path = DOCS / name
        if not path.exists():
            print(f"  skip (not found): {name}")
            continue
        original = path.read_text(encoding="utf-8")
        rendered = render_page(original, used)
        if rendered != original:
            changed.append(name)
            if not args.check:
                path.write_text(rendered, encoding="utf-8")
        print(f"  {name}: {'updated' if rendered != original else 'up to date'}")

    unused = sorted(set(RENDERERS) - used)
    if unused:
        print(f"  note: 未使用のレンダラ {unused}")

    if args.check and changed:
        print(f"\n差分があります: {', '.join(changed)}")
        print("uv run python scripts/build_site.py を実行してください。")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
