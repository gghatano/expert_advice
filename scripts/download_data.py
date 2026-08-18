"""実験に使うデータセットを data/raw/ にダウンロードする.

``data/raw/`` は .gitignore されているので、リポジトリを clone した直後は空。
このスクリプトを 1 回走らせれば ``scripts/experiments/*.py`` がすべて動く。

すでに存在するファイルは既定でスキップする (``--force`` で再取得)。

Usage:
    uv run python scripts/download_data.py
    uv run python scripts/download_data.py --only m4 etth1
    uv run python scripts/download_data.py --force
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW = _PROJECT_ROOT / "data" / "raw"

# name -> (url, 保存ファイル名, 概算サイズ, 説明)
DATASETS: dict[str, tuple[str, str, str, str]] = {
    "m4": (
        "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/Train/Hourly-train.csv",
        "M4-Hourly-train.csv",
        "2.3 MB",
        "M4 Competition, hourly 部門の学習系列 (414 系列)",
    ),
    "m4_test": (
        "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/Test/Hourly-test.csv",
        "M4-Hourly-test.csv",
        "0.1 MB",
        "M4 Competition, hourly 部門の評価系列",
    ),
    "etth1": (
        "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
        "ETTh1.csv",
        "2.5 MB",
        "Electricity Transformer Temperature, 1 時間粒度 (17,420 時点 x 7 変数)",
    ),
    "opsd": (
        "https://data.open-power-system-data.org/time_series/latest/time_series_60min_singleindex.csv",
        "opsd_time_series_60min.csv",
        "130 MB",
        "Open Power System Data, 欧州各国の 1 時間粒度実測需要 (2015-2020)",
    ),
    "uci": (
        "https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip",
        "LD2011_2014.txt.zip",
        "76 MB (展開後 711 MB)",
        "UCI Electricity Load Diagrams 2011-2014 (370 系列, 15 分粒度)",
    ),
}


def _report(name: str, blocks: int, block_size: int, total: int) -> None:
    if total <= 0:
        return
    done = min(blocks * block_size, total)
    pct = 100.0 * done / total
    sys.stdout.write(f"\r    {name}: {pct:5.1f}%  ({done / 1e6:.0f}/{total / 1e6:.0f} MB)")
    sys.stdout.flush()


def download(key: str, force: bool = False) -> Path:
    url, filename, size, desc = DATASETS[key]
    dest = RAW / filename
    if dest.exists() and not force:
        print(f"  [skip] {filename} は取得済み")
        return dest

    print(f"  [get ] {filename}  ({size}) — {desc}")
    RAW.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(
        url, tmp, reporthook=lambda b, bs, t: _report(filename, b, bs, t)
    )
    sys.stdout.write("\r" + " " * 70 + "\r")
    tmp.replace(dest)
    return dest


def maybe_unzip(path: Path, expected: str, force: bool = False) -> None:
    """zip を展開する (展開済みならスキップ)."""
    target = path.parent / expected
    if target.exists() and not force:
        print(f"  [skip] {expected} は展開済み")
        return
    print(f"  [unzip] {path.name} -> {expected}")
    with zipfile.ZipFile(path) as zf:
        zf.extractall(path.parent)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", choices=sorted(DATASETS),
                        help="取得するデータセットを限定する")
    parser.add_argument("--force", action="store_true", help="既存ファイルも再取得する")
    args = parser.parse_args()

    keys = args.only or list(DATASETS)
    print(f"保存先: {RAW}")
    for key in keys:
        try:
            path = download(key, force=args.force)
        except Exception as e:  # noqa: BLE001 - ネットワーク要因をそのまま見せる
            print(f"  [FAIL] {key}: {type(e).__name__}: {e}")
            continue
        if key == "uci":
            maybe_unzip(path, "LD2011_2014.txt", force=args.force)

    print("\n完了。実験は scripts/experiments/ 以下のスクリプトで実行できます。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
