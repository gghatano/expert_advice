"""UCI Electricity Load Diagrams 2011-2014 データの読み込み."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# プロジェクトルートからの既定パス
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = _PROJECT_ROOT / "data" / "raw"


def load_electricity(
    path: str | Path | None = None,
    raw_dir: str | Path | None = None,
) -> pd.DataFrame:
    """UCI Electricity データを読み込み、370系列 x 時刻インデックスの DataFrame を返す.

    Parameters
    ----------
    path : str | Path | None
        CSV ファイルへの直接パス。指定時は raw_dir は無視される。
    raw_dir : str | Path | None
        CSV が配置されたディレクトリ。None の場合 ``data/raw/`` を使う。
        ディレクトリ内の最初の ``.csv`` / ``.txt`` ファイルを自動検出する。

    Returns
    -------
    pd.DataFrame
        DatetimeIndex (15 分間隔) × 370 系列 (float64) の DataFrame。

    Raises
    ------
    FileNotFoundError
        CSV ファイルが見つからない場合。
    """
    csv_path = _resolve_csv_path(path, raw_dir)

    # UCI 版は セミコロン区切り・小数点カンマ（ヨーロッパ形式）の場合がある。
    # まず先頭を読んで区切り文字を推定する。
    sep, decimal = _detect_format(csv_path)

    df = pd.read_csv(
        csv_path,
        sep=sep,
        decimal=decimal,
        index_col=0,
        parse_dates=[0],
        dayfirst=True,  # ヨーロッパ形式 dd/mm/yyyy に対応
    )

    # インデックス名を統一
    df.index.name = "datetime"

    # カラム名を文字列に正規化 ("MT_001" 等)
    df.columns = [str(c).strip() for c in df.columns]

    # 末尾に空カラムが混入する場合がある (末尾セミコロン)
    df = df.loc[:, df.columns != ""]
    # Unnamed カラムを除去
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    # 数値型に変換 (読み込みで object になった列を救済)
    df = df.apply(pd.to_numeric, errors="coerce")

    # ソート
    df = df.sort_index()

    return df


# ---------- internal helpers ----------


def _resolve_csv_path(
    path: str | Path | None,
    raw_dir: str | Path | None,
) -> Path:
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"指定されたファイルが見つかりません: {p}")
        return p

    raw = Path(raw_dir) if raw_dir is not None else DEFAULT_RAW_DIR
    candidates = sorted(raw.glob("*.csv")) + sorted(raw.glob("*.txt"))
    if not candidates:
        raise FileNotFoundError(
            f"data/raw/ に CSV ファイルが見つかりません: {raw}"
        )
    return candidates[0]


def _detect_format(path: Path) -> tuple[str, str]:
    """先頭数行を読んで (sep, decimal) を推定する."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        head = f.read(4096)

    # セミコロンがカンマより多ければセミコロン区切り
    if head.count(";") > head.count(","):
        sep = ";"
        # セミコロン区切りの場合、小数点にカンマが使われていることが多い
        decimal = ","
    else:
        sep = ","
        decimal = "."

    return sep, decimal
