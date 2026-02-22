"""Expertファクトリ: preset名からExpertインスタンス群を一括生成する."""

from __future__ import annotations

from src.experts.base import BaseExpert
from src.experts.moving_avg import SMA, Median
from src.experts.naive import Drift, LastValue, SeasonalNaive
from src.experts.regression import HuberRegressorLag, KNNLag, RidgeLag
from src.experts.seasonal_profile import STLSeasonalMean
from src.experts.smoothing import EMA


def create_experts(preset: str = "light30") -> list[BaseExpert]:
    """preset名に応じたExpertインスタンスリストを返す.

    Parameters
    ----------
    preset : str
        "light30" (約30本) または "light80" (約80本).

    Returns
    -------
    list[BaseExpert]
    """
    if preset == "light30":
        return _light30()
    elif preset == "light80":
        return _light80()
    else:
        raise ValueError(f"Unknown preset: {preset!r}. Choose 'light30' or 'light80'.")


def _light30() -> list[BaseExpert]:
    """約30本の基本パラメータExpert群."""
    experts: list[BaseExpert] = []

    # Naive系 (6本)
    experts.append(LastValue())
    for s in [24, 48, 168]:
        experts.append(SeasonalNaive(season_length=s))
    for w in [24, 168]:
        experts.append(Drift(window=w))

    # 移動平均系 (7本)
    for w in [24, 48, 168]:
        experts.append(SMA(window=w))
    experts.append(Median(window=24))
    for a in [0.1, 0.3, 0.5]:
        experts.append(EMA(alpha=a))

    # 回帰系 (5本)
    for a in [0.1, 1.0, 10.0]:
        experts.append(RidgeLag(alpha=a))
    experts.append(HuberRegressorLag())
    experts.append(KNNLag(k=5))

    # 季節性 (1本)
    experts.append(STLSeasonalMean())

    # 追加バリエーション (11本) → 合計30本
    experts.append(SeasonalNaive(season_length=72))
    experts.append(Drift(window=48))
    experts.append(Drift(window=336))
    experts.append(SMA(window=12))
    experts.append(SMA(window=336))
    experts.append(Median(window=48))
    experts.append(Median(window=168))
    experts.append(EMA(alpha=0.05))
    experts.append(EMA(alpha=0.7))
    experts.append(KNNLag(k=3))
    experts.append(KNNLag(k=10))

    return experts


def _light80() -> list[BaseExpert]:
    """約80本のExpert群（light30 + バリエーション拡張）."""
    experts = _light30()

    # SeasonalNaive追加 (4本)
    for s in [12, 36, 96, 336]:
        experts.append(SeasonalNaive(season_length=s))

    # Drift追加 (3本)
    for w in [12, 72, 504]:
        experts.append(Drift(window=w))

    # SMA追加 (5本)
    for w in [6, 36, 72, 96, 504]:
        experts.append(SMA(window=w))

    # Median追加 (4本)
    for w in [12, 36, 72, 336]:
        experts.append(Median(window=w))

    # EMA追加 (7本)
    for a in [0.01, 0.02, 0.15, 0.2, 0.4, 0.8, 0.9]:
        experts.append(EMA(alpha=a))

    # Ridge追加 (4本)
    for a in [0.01, 0.5, 5.0, 50.0]:
        experts.append(RidgeLag(alpha=a))

    # Huber追加 (なし、パラメータ自由度少ない)

    # KNN追加 (4本)
    for k in [1, 7, 15, 20]:
        experts.append(KNNLag(k=k))

    # KNN max_samples違い (3本)
    for ms in [200, 1000, 2000]:
        experts.append(KNNLag(k=5, max_samples=ms))

    # 追加バリエーション (16本) → 合計80本前後
    for s in [6, 18, 120, 240]:
        experts.append(SeasonalNaive(season_length=s))
    for w in [6, 18, 96, 240]:
        experts.append(Drift(window=w))
    for w in [18, 240]:
        experts.append(SMA(window=w))
    for w in [6, 96]:
        experts.append(Median(window=w))
    for a in [0.03, 0.25, 0.6, 0.95]:
        experts.append(EMA(alpha=a))

    return experts
