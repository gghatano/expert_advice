"""時系列データの Train / Valid / Test 分割."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class TimeSeriesSplit:
    """分割結果を保持するデータクラス."""

    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame


# デフォルトの分割境界
TRAIN_START = "2011-01-01"
TRAIN_END = "2013-12-31 23:00:00"
VALID_START = "2014-01-01"
VALID_END = "2014-06-30 23:00:00"
TEST_START = "2014-07-01"
TEST_END = "2014-12-31 23:00:00"


def split_temporal(
    df: pd.DataFrame,
    train_start: str = TRAIN_START,
    train_end: str = TRAIN_END,
    valid_start: str = VALID_START,
    valid_end: str = VALID_END,
    test_start: str = TEST_START,
    test_end: str = TEST_END,
) -> TimeSeriesSplit:
    """時間順に Train / Valid / Test に分割する.

    リークを防ぐため、各区間は重複しない。

    Parameters
    ----------
    df : pd.DataFrame
        DatetimeIndex を持つ DataFrame。
    train_start, train_end : str
    valid_start, valid_end : str
    test_start, test_end : str

    Returns
    -------
    TimeSeriesSplit
    """
    train = df.loc[train_start:train_end]
    valid = df.loc[valid_start:valid_end]
    test = df.loc[test_start:test_end]

    # リーク防止の検証
    _assert_no_leak(train, valid, test)

    return TimeSeriesSplit(train=train, valid=valid, test=test)


def _assert_no_leak(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """各分割間に時間的な重複がないことを検証する."""
    if len(train) > 0 and len(valid) > 0:
        assert train.index.max() < valid.index.min(), (
            f"Train と Valid が重複しています: "
            f"train.max={train.index.max()}, valid.min={valid.index.min()}"
        )
    if len(valid) > 0 and len(test) > 0:
        assert valid.index.max() < test.index.min(), (
            f"Valid と Test が重複しています: "
            f"valid.max={valid.index.max()}, test.min={test.index.min()}"
        )
    if len(train) > 0 and len(test) > 0:
        assert train.index.max() < test.index.min(), (
            f"Train と Test が重複しています: "
            f"train.max={train.index.max()}, test.min={test.index.min()}"
        )
