"""オンライン集約の共通実行ループとベースライン.

Expert の 1 ステップ先予測をあらかじめ行列 ``P`` (shape ``(T, N)``) に
まとめておけば、任意の集約アルゴリズムを同じ土俵で高速に比較できる。

``P[t, i]`` は「Expert i が時刻 t の値 ``y[t]`` を、``y[:t]`` だけを見て
予測した値」でなければならない (リークなし)。行列の作り方は
:mod:`src.experts.vectorized` を参照。

損失は2種類を区別する:

* **update loss** — 重み更新に使う損失。系列のスケールに依存しないよう
  正規化することが多い。
* **eval loss**   — 報告用の損失 (生の MAE など)。

この分離により「重み更新は正規化損失、評価は生 MAE」という実務的な構成を
そのまま表現できる。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np

# ---------------------------------------------------------------------------
# 損失関数
# ---------------------------------------------------------------------------

LossFn = Callable[[np.ndarray, np.ndarray], np.ndarray]

LOSS_FNS: dict[str, LossFn] = {
    "abs": lambda y, p: np.abs(y - p),
    "sq": lambda y, p: (y - p) ** 2,
}


def get_loss_fn(name: str) -> LossFn:
    """名前から損失関数を取得する."""
    if name not in LOSS_FNS:
        raise ValueError(f"Unknown loss {name!r}. Choose from {sorted(LOSS_FNS)}.")
    return LOSS_FNS[name]


# ---------------------------------------------------------------------------
# 集約器プロトコル
# ---------------------------------------------------------------------------


class Aggregator(Protocol):
    """Hedge / FixedShare / AdaHedge / MLPoly が満たすインターフェース."""

    n_experts: int

    def predict(self, expert_predictions: np.ndarray) -> float: ...
    def update(self, losses: np.ndarray) -> None: ...
    def get_weights(self) -> np.ndarray: ...


# ---------------------------------------------------------------------------
# ベースライン集約器
# ---------------------------------------------------------------------------


class EqualWeight:
    """全 Expert を等重みで平均するだけのベースライン."""

    USES_ENSEMBLE_LOSS = False

    def __init__(self, n_experts: int) -> None:
        if n_experts < 1:
            raise ValueError("n_experts must be >= 1")
        self.n_experts = n_experts
        self._w = np.full(n_experts, 1.0 / n_experts, dtype=np.float64)

    def predict(self, expert_predictions: np.ndarray) -> float:
        return float(np.mean(np.asarray(expert_predictions, dtype=np.float64)))

    def update(self, losses: np.ndarray) -> None:  # noqa: D401 - no state
        pass

    def get_weights(self) -> np.ndarray:
        return self._w.copy()


class FollowTheLeader:
    """累積損失が最小の Expert にすべての重みを置くベースライン.

    同点の場合はその Expert 群で等分する。定常データでは非常に強いが、
    レジーム変化には弱い (切り替えの度に大きな損失を出す)。
    """

    USES_ENSEMBLE_LOSS = False

    def __init__(self, n_experts: int) -> None:
        if n_experts < 1:
            raise ValueError("n_experts must be >= 1")
        self.n_experts = n_experts
        self.cum_losses = np.zeros(n_experts, dtype=np.float64)

    def _weights(self) -> np.ndarray:
        best = self.cum_losses <= self.cum_losses.min() + 1e-12
        return best.astype(np.float64) / best.sum()

    def predict(self, expert_predictions: np.ndarray) -> float:
        return float(np.dot(self._weights(), np.asarray(expert_predictions, dtype=np.float64)))

    def update(self, losses: np.ndarray) -> None:
        self.cum_losses += np.asarray(losses, dtype=np.float64)

    def get_weights(self) -> np.ndarray:
        return self._weights()


# ---------------------------------------------------------------------------
# 実行結果
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    """1つの集約アルゴリズムを1系列に走らせた結果."""

    name: str
    predictions: np.ndarray
    losses: np.ndarray
    weights: np.ndarray | None = None
    snapshot_steps: np.ndarray | None = None
    extra: dict = field(default_factory=dict)

    @property
    def mean_loss(self) -> float:
        return float(np.mean(self.losses))

    @property
    def cumulative_loss(self) -> np.ndarray:
        return np.cumsum(self.losses)


# ---------------------------------------------------------------------------
# メインループ
# ---------------------------------------------------------------------------


def run_online(
    P: np.ndarray,
    y: np.ndarray,
    aggregator,
    *,
    name: str,
    update_loss: str = "abs",
    eval_loss: str = "abs",
    loss_scale: float = 1.0,
    record_weights: bool = False,
    snapshot_every: int = 1,
) -> RunResult:
    """予測行列 ``P`` の上で集約アルゴリズムを逐次実行する.

    Parameters
    ----------
    P : np.ndarray
        Expert 予測行列 ``(T, N)``。``P[t, i]`` は ``y[:t]`` のみに依存。
    y : np.ndarray
        実測値 ``(T,)``。
    aggregator
        ``predict`` / ``update`` / ``get_weights`` を持つオブジェクト。
        ``USES_ENSEMBLE_LOSS`` が真なら ``update(losses, ensemble_loss)``
        の形で呼ばれる。
    update_loss, eval_loss : str
        重み更新用・評価用の損失名 (``"abs"`` または ``"sq"``)。
    loss_scale : float
        重み更新に渡す損失をこの値で割る。系列スケールの正規化に使う。
    record_weights : bool
        True なら ``snapshot_every`` ステップごとに重みを記録する。

    Returns
    -------
    RunResult
    """
    P = np.asarray(P, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if P.ndim != 2:
        raise ValueError(f"P must be 2-D, got shape {P.shape}")
    if y.shape != (P.shape[0],):
        raise ValueError(f"y must have shape ({P.shape[0]},), got {y.shape}")
    if P.shape[1] != aggregator.n_experts:
        raise ValueError(
            f"aggregator expects {aggregator.n_experts} experts, P has {P.shape[1]}"
        )
    if loss_scale <= 0:
        raise ValueError("loss_scale must be positive")

    T = P.shape[0]
    upd_fn = get_loss_fn(update_loss)
    ev_fn = get_loss_fn(eval_loss)
    uses_ensemble_loss = getattr(aggregator, "USES_ENSEMBLE_LOSS", False)

    preds = np.empty(T, dtype=np.float64)
    ev_losses = np.empty(T, dtype=np.float64)

    snapshots: list[np.ndarray] = []
    snapshot_steps: list[int] = []

    for t in range(T):
        row = P[t]
        pred = aggregator.predict(row)
        preds[t] = pred

        y_t = y[t]
        ev_losses[t] = float(ev_fn(y_t, pred))

        expert_upd = upd_fn(y_t, row) / loss_scale
        if uses_ensemble_loss:
            ens_upd = float(upd_fn(y_t, pred)) / loss_scale
            aggregator.update(expert_upd, ens_upd)
        else:
            aggregator.update(expert_upd)

        if record_weights and (t % snapshot_every == 0):
            snapshots.append(aggregator.get_weights())
            snapshot_steps.append(t)

    return RunResult(
        name=name,
        predictions=preds,
        losses=ev_losses,
        weights=np.vstack(snapshots) if snapshots else None,
        snapshot_steps=np.asarray(snapshot_steps, dtype=int) if snapshot_steps else None,
    )


# ---------------------------------------------------------------------------
# 事後 (oracle) ベンチマーク
# ---------------------------------------------------------------------------


def expert_loss_matrix(P: np.ndarray, y: np.ndarray, loss: str = "abs") -> np.ndarray:
    """Expert ごと・時刻ごとの損失行列 ``(T, N)`` を返す."""
    fn = get_loss_fn(loss)
    return fn(np.asarray(y, dtype=np.float64)[:, None], np.asarray(P, dtype=np.float64))


def best_fixed_expert(L: np.ndarray) -> tuple[int, float]:
    """事後的に最良だった単一 Expert の ``(index, 平均損失)`` を返す.

    これが Hedge 系アルゴリズムの理論保証の比較対象 (regret の基準) になる。
    """
    means = L.mean(axis=0)
    idx = int(np.argmin(means))
    return idx, float(means[idx])


def oracle_switching_loss(L: np.ndarray, segments: list[tuple[int, int]]) -> float:
    """区間ごとに最良 Expert を選べた場合の平均損失 (切り替えオラクル).

    Parameters
    ----------
    L : np.ndarray
        損失行列 ``(T, N)``。
    segments : list[tuple[int, int]]
        ``[start, end)`` の区間リスト。
    """
    total = 0.0
    count = 0
    for start, end in segments:
        block = L[start:end]
        if len(block) == 0:
            continue
        total += float(block.sum(axis=0).min())
        count += len(block)
    if count == 0:
        raise ValueError("segments cover no time steps")
    return total / count
