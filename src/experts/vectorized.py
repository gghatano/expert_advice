"""Expert の 1 ステップ先予測を行列としてまとめて計算する高速版.

``src.experts`` の各クラスは「毎時刻 ``predict_next`` を呼ぶ」逐次 API で、
これは仕様どおりだが、履歴 ``pd.Series`` を毎回スライスするため
``O(T x N)`` の呼び出しコストが大きい。数万点 x 数百系列の比較実験には
重すぎる。

ここでは同じ Expert 群を **ベクトル化** して、予測行列

    P[t, i] = Expert i が y[:t] だけを見て y[t] を予測した値

を一括生成する。リークがないことは各実装の ``shift`` で保証している。
逐次版との一致は ``tests/test_vectorized.py`` で検証している。

Notes
-----
``period`` は季節周期 (時間データなら 24)、``week`` はより長い周期
(時間データなら 168) を表す。M4 のようにタイムスタンプがない系列でも
位置インデックスの剰余で暦特徴を作れるようにしてある。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 個々の Expert のベクトル化実装
# ---------------------------------------------------------------------------


def _shift(col: np.ndarray, k: int) -> np.ndarray:
    """``col`` を k ステップ後ろにずらす (先頭 k 点は NaN)."""
    out = np.full_like(col, np.nan)
    if k <= 0:
        return col.copy()
    if k < len(col):
        out[k:] = col[:-k]
    return out


def _last_value(y: np.ndarray, h: int = 1) -> np.ndarray:
    """利用可能な最新の観測 (h ステップ先予測なら y[t-h])."""
    return _shift(y, h)


def _seasonal_naive(y: np.ndarray, season: int, h: int = 1) -> np.ndarray:
    """h 以上で最小の season の倍数だけ前の値.

    24 時間先予測で season=24 なら y[t-24]、season=168 なら y[t-168]。
    """
    if season < 1:
        raise ValueError("season must be >= 1")
    lag = season * int(np.ceil(h / season))
    return _shift(y, lag)


def _rolling_sum(y: np.ndarray, w: int) -> np.ndarray:
    """窓 w の移動和 (t を含む)。先頭 w-1 点は NaN."""
    T = len(y)
    out = np.full(T, np.nan, dtype=np.float64)
    if w > T:
        return out
    csum = np.concatenate([[0.0], np.cumsum(y)])
    out[w - 1:] = csum[w:] - csum[:-w]
    return out


def _sma(y: np.ndarray, w: int, h: int = 1) -> np.ndarray:
    """利用可能な直近 w 点の単純平均."""
    incl = _rolling_sum(y, w) / w
    return _shift(incl, h)


def _median(y: np.ndarray, w: int, h: int = 1) -> np.ndarray:
    """利用可能な直近 w 点の中央値."""
    s = pd.Series(y).rolling(w).median().shift(h)
    return s.to_numpy(dtype=np.float64)


def _ema(y: np.ndarray, alpha: float, h: int = 1) -> np.ndarray:
    """指数移動平均。``adjust=False`` の再帰式."""
    s = pd.Series(y).ewm(alpha=alpha, adjust=False).mean().shift(h)
    return s.to_numpy(dtype=np.float64)


def _drift(y: np.ndarray, w: int, h: int = 1) -> np.ndarray:
    """利用可能な直近 w 点に直線を当てはめ h ステップ外挿する.

    窓内の位置 ``x = 0..w-1`` に対する最小二乗直線を求め、``x = w-1+h``
    を予測する。畳み込みで移動加重和を求めるので高速。
    """
    T = len(y)
    out = np.full(T, np.nan, dtype=np.float64)
    if w < 2 or w > T:
        return out

    x = np.arange(w, dtype=np.float64)
    x_mean = x.mean()
    sxx = float(((x - x_mean) ** 2).sum())
    if sxx <= 0:
        return out

    # t を含む窓 [t-w+1, t] に対する sum(y) と sum(x*y)
    sum_y = _rolling_sum(y, w)
    sum_xy = np.convolve(y, x[::-1])[:T]

    y_mean = sum_y / w
    slope = (sum_xy - w * x_mean * y_mean) / sxx
    # 窓の最後が x = w-1、その h ステップ先は x = w-1+h
    pred_incl = y_mean + slope * (w - 1 + h - x_mean)

    out = _shift(pred_incl, h)
    out[: w + h - 1] = np.nan  # 窓が埋まるまでは使わない
    return out


def _calendar_features(T: int, index: pd.DatetimeIndex | None, period: int, week: int) -> np.ndarray:
    """(T, 2) の暦特徴 [周期内位置, 長周期内のブロック番号] を返す."""
    if index is not None:
        pos = index.hour.to_numpy(dtype=np.float64)
        blk = index.dayofweek.to_numpy(dtype=np.float64)
    else:
        t = np.arange(T)
        pos = (t % period).astype(np.float64)
        blk = ((t % week) // period).astype(np.float64)
    return np.column_stack([pos, blk])


def _lag_design(
    y: np.ndarray,
    lags: list[int],
    cal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """ラグ特徴 + 暦特徴の設計行列 ``X`` と有効行マスクを返す.

    ``X[t]`` は ``y[t-lag]`` のみを使うので未来情報は入らない。
    """
    T = len(y)
    cols = []
    valid = np.ones(T, dtype=bool)
    for lag in lags:
        col = np.full(T, np.nan, dtype=np.float64)
        if lag < T:
            col[lag:] = y[:-lag]
        cols.append(col)
        valid &= ~np.isnan(col)
    X = np.column_stack(cols + [cal])
    return X, valid


def _ridge_lag(
    y: np.ndarray,
    X: np.ndarray,
    valid: np.ndarray,
    alpha: float,
    train_end: int,
) -> np.ndarray:
    """train 期間で Ridge を 1 回学習し、全期間を予測する.

    閉形式解 ``(X'X + alpha I)^-1 X'y`` を使う (切片は中心化で処理)。
    """
    T = len(y)
    out = np.full(T, np.nan, dtype=np.float64)

    fit_mask = valid.copy()
    fit_mask[train_end:] = False
    if fit_mask.sum() < X.shape[1] + 1:
        return out

    Xf = X[fit_mask]
    yf = y[fit_mask]
    x_mean = Xf.mean(axis=0)
    y_mean = float(yf.mean())
    Xc = Xf - x_mean
    yc = yf - y_mean

    d = Xc.shape[1]
    coef = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(d), Xc.T @ yc)

    out[valid] = (X[valid] - x_mean) @ coef + y_mean
    return out


def _seasonal_profile(
    y: np.ndarray,
    period: int,
    week: int,
    index: pd.DatetimeIndex | None,
    train_end: int,
) -> np.ndarray:
    """train 期間の「曜日 x 時刻」平均プロファイルを予測に使う."""
    T = len(y)
    if index is not None:
        key = (index.dayofweek.to_numpy() * 24 + index.hour.to_numpy()).astype(int)
        n_key = 7 * 24
    else:
        key = (np.arange(T) % week).astype(int)
        n_key = week

    out = np.full(T, np.nan, dtype=np.float64)
    train_key = key[:train_end]
    train_y = y[:train_end]
    if len(train_y) == 0:
        return out

    sums = np.bincount(train_key, weights=train_y, minlength=n_key)
    counts = np.bincount(train_key, minlength=n_key)
    with np.errstate(invalid="ignore", divide="ignore"):
        profile = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    out[:] = profile[key]
    return out


# ---------------------------------------------------------------------------
# プリセット定義
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExpertSpec:
    """1 本の Expert を表す仕様."""

    kind: str
    param: float
    name: str


def _spec_list(preset: str, period: int, week: int) -> list[ExpertSpec]:
    """preset 名から Expert 仕様のリストを返す."""
    p, w = period, week

    if preset == "core8":
        # レジーム図解用の、性格がはっきり違う少数精鋭。
        return [
            ExpertSpec("last", 0, "LastValue"),
            ExpertSpec("snaive", p, f"SeasonalNaive_{p}"),
            ExpertSpec("snaive", w, f"SeasonalNaive_{w}"),
            ExpertSpec("sma", p, f"SMA_{p}"),
            ExpertSpec("sma", max(2, p // 6), f"SMA_{max(2, p // 6)}"),
            ExpertSpec("ema", 0.3, "EMA_0.3"),
            ExpertSpec("drift", p, f"Drift_{p}"),
            ExpertSpec("profile", 0, "SeasonalProfile"),
        ]

    if preset != "light30":
        raise ValueError(f"Unknown preset: {preset!r}. Choose 'light30' or 'core8'.")

    specs: list[ExpertSpec] = [ExpertSpec("last", 0, "LastValue")]
    for s in [max(2, p // 2), p, 2 * p, 3 * p, w]:
        specs.append(ExpertSpec("snaive", s, f"SeasonalNaive_{s}"))
    for win in [max(2, p // 4), max(2, p // 2), p, 2 * p, w, 2 * w]:
        specs.append(ExpertSpec("sma", win, f"SMA_{win}"))
    for win in [p, 2 * p, w]:
        specs.append(ExpertSpec("median", win, f"Median_{win}"))
    for a in [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
        specs.append(ExpertSpec("ema", a, f"EMA_{a}"))
    for win in [p, 2 * p, w, 2 * w]:
        specs.append(ExpertSpec("drift", win, f"Drift_{win}"))
    for a in [0.1, 1.0, 10.0, 100.0]:
        specs.append(ExpertSpec("ridge", a, f"RidgeLag_{a}"))
    specs.append(ExpertSpec("profile", 0, "SeasonalProfile"))
    return specs


# ---------------------------------------------------------------------------
# 公開 API
# ---------------------------------------------------------------------------


def build_expert_matrix(
    y: np.ndarray,
    *,
    index: pd.DatetimeIndex | None = None,
    period: int = 24,
    week: int = 168,
    preset: str = "light30",
    train_end: int | None = None,
    horizon: int = 1,
) -> tuple[list[str], np.ndarray]:
    """Expert 予測行列を一括生成する.

    Parameters
    ----------
    y : np.ndarray
        観測値 ``(T,)``。
    index : pd.DatetimeIndex | None
        タイムスタンプ。無い場合は位置の剰余で暦特徴を作る。
    period, week : int
        短周期・長周期 (時間粒度なら 24 と 168)。
    preset : str
        ``"light30"`` (30本) または ``"core8"`` (8本)。
    train_end : int | None
        学習が必要な Expert (Ridge・季節プロファイル) が使う訓練区間の
        終端インデックス。既定は全体の 50%。
    horizon : int
        予測ホライズン h。``P[t, i]`` は ``y[:t-h+1]`` のみを使って
        ``y[t]`` を予測する。1 なら 1 ステップ先、24 なら翌日同時刻。

    Returns
    -------
    tuple[list[str], np.ndarray]
        Expert 名リストと予測行列 ``(T, N)``。
    """
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 1:
        raise ValueError(f"y must be 1-D, got shape {y.shape}")
    T = len(y)
    if T < 3:
        raise ValueError("series is too short")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if train_end is None:
        train_end = max(2, T // 2)

    h = horizon
    specs = _spec_list(preset, period, week)
    cal = _calendar_features(T, index, period, week)

    needs_ridge = any(s.kind == "ridge" for s in specs)
    if needs_ridge:
        # ラグはすべて h 以上でなければ未来を覗いてしまう。
        candidates = {h, h + 1, h + 2}
        for s in (period, week):
            candidates.add(s * int(np.ceil(h / s)))
        lags = sorted({l for l in candidates if l < T})
        # 最長ラグが訓練区間に対して長すぎると学習サンプルが作れないので、
        # 十分な行数が残るまで長いラグから落とす。
        while lags and (train_end - max(lags)) < len(lags) + 4:
            lags = lags[:-1]
        X, valid = _lag_design(y, lags, cal)

    cols: list[np.ndarray] = []
    for spec in specs:
        if spec.kind == "last":
            col = _last_value(y, h)
        elif spec.kind == "snaive":
            col = _seasonal_naive(y, int(spec.param), h)
        elif spec.kind == "sma":
            col = _sma(y, int(spec.param), h)
        elif spec.kind == "median":
            col = _median(y, int(spec.param), h)
        elif spec.kind == "ema":
            col = _ema(y, float(spec.param), h)
        elif spec.kind == "drift":
            col = _drift(y, int(spec.param), h)
        elif spec.kind == "ridge":
            col = _ridge_lag(y, X, valid, float(spec.param), train_end)
        elif spec.kind == "profile":
            col = _seasonal_profile(y, period, week, index, train_end)
        else:  # pragma: no cover - guarded by _spec_list
            raise ValueError(f"Unknown expert kind: {spec.kind!r}")
        cols.append(col)

    P = np.column_stack(cols)

    # 履歴不足による NaN は「利用可能な最新値」で埋め、先頭は y[0]。
    fallback = _last_value(y, h)
    fallback[:h] = y[0]
    nan_mask = ~np.isfinite(P)
    if nan_mask.any():
        P[nan_mask] = np.broadcast_to(fallback[:, None], P.shape)[nan_mask]

    names = [s.name for s in specs]
    return names, P
