"""前処理パイプライン: 集約・欠損処理・外れ値クリップ・キャッシュ."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"


def resample_hourly(
    df: pd.DataFrame,
    method: Literal["sum", "mean"] = "sum",
) -> pd.DataFrame:
    """15 分間隔データを 1 時間に集約する.

    Parameters
    ----------
    df : pd.DataFrame
        15 分間隔の DatetimeIndex を持つ DataFrame。
    method : {"sum", "mean"}
        集約方法。デフォルトは sum。

    Returns
    -------
    pd.DataFrame
        1 時間間隔に集約された DataFrame。
    """
    resampler = df.resample("1h")
    if method == "sum":
        return resampler.sum()
    elif method == "mean":
        return resampler.mean()
    else:
        raise ValueError(f"method は 'sum' または 'mean' を指定してください: {method}")


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """欠損処理: ffill(limit=24) → 残りを 0 埋め.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    df = df.ffill(limit=24)
    df = df.fillna(0)
    return df


def clip_outliers(
    df: pd.DataFrame,
    lower_pct: float = 0.5,
    upper_pct: float = 99.5,
) -> pd.DataFrame:
    """系列ごとにパーセンタイルで外れ値をクリップする.

    Parameters
    ----------
    df : pd.DataFrame
    lower_pct : float
        下限パーセンタイル (0-100)。
    upper_pct : float
        上限パーセンタイル (0-100)。

    Returns
    -------
    pd.DataFrame
    """
    lower = df.quantile(lower_pct / 100)
    upper = df.quantile(upper_pct / 100)
    return df.clip(lower=lower, upper=upper, axis=1)


def drop_high_missing(
    df: pd.DataFrame,
    threshold: float = 0.05,
) -> pd.DataFrame:
    """欠損率が閾値を超える系列を除外する.

    Parameters
    ----------
    df : pd.DataFrame
    threshold : float
        除外閾値 (0-1)。デフォルト 0.05 = 5%。

    Returns
    -------
    pd.DataFrame
    """
    missing_rate = df.isna().mean()
    keep = missing_rate[missing_rate <= threshold].index
    return df[keep]


def preprocess(
    df: pd.DataFrame,
    *,
    resample_method: Literal["sum", "mean"] = "sum",
    clip: bool = False,
    clip_lower_pct: float = 0.5,
    clip_upper_pct: float = 99.5,
    drop_missing: bool = False,
    missing_threshold: float = 0.05,
) -> pd.DataFrame:
    """前処理パイプラインを一括実行する.

    Parameters
    ----------
    df : pd.DataFrame
        生データ (15 分間隔)。
    resample_method : {"sum", "mean"}
    clip : bool
        外れ値クリップを適用するか。デフォルト False。
    clip_lower_pct, clip_upper_pct : float
    drop_missing : bool
        欠損率の高い系列を除外するか。デフォルト False。
    missing_threshold : float

    Returns
    -------
    pd.DataFrame
        前処理済み DataFrame (1 時間間隔)。
    """
    # 欠損率の高い系列を除外（集約前の生データで判定）
    if drop_missing:
        df = drop_high_missing(df, threshold=missing_threshold)

    # 1 時間集約
    df = resample_hourly(df, method=resample_method)

    # 欠損補完
    df = fill_missing(df)

    # 外れ値クリップ
    if clip:
        df = clip_outliers(df, lower_pct=clip_lower_pct, upper_pct=clip_upper_pct)

    return df


# ---------- Parquet キャッシュ ----------


def save_processed(
    df: pd.DataFrame,
    name: str = "electricity_hourly.parquet",
    processed_dir: str | Path | None = None,
) -> Path:
    """処理済みデータを Parquet 形式で保存する.

    Returns
    -------
    Path
        保存先のパス。
    """
    out_dir = Path(processed_dir) if processed_dir else DEFAULT_PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / name
    df.to_parquet(out_path, engine="pyarrow")
    return out_path


def load_processed(
    name: str = "electricity_hourly.parquet",
    processed_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Parquet キャッシュから処理済みデータを読み込む.

    Raises
    ------
    FileNotFoundError
        キャッシュが存在しない場合。
    """
    out_dir = Path(processed_dir) if processed_dir else DEFAULT_PROCESSED_DIR
    out_path = out_dir / name
    if not out_path.exists():
        raise FileNotFoundError(f"処理済みデータが見つかりません: {out_path}")
    return pd.read_parquet(out_path, engine="pyarrow")
