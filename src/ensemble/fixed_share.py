"""Fixed-Share: Hedge with a uniform "share" step (Herbster & Warmuth, 1998).

Plain :class:`~src.ensemble.hedge.Hedge` competes with the best *fixed*
expert.  Once it has concentrated its weight on one expert, recovering
after a regime change costs it a large number of rounds, because the
log-weight of the abandoned experts has drifted arbitrarily far below the
leader's.

Fixed-Share adds a mixing step after every exponential update::

    w <- (1 - alpha) * w + alpha / N

which floors every weight at ``alpha / N``.  This bounds how far behind an
expert can fall, so the algorithm can switch leaders in O(log(1/alpha))
rounds.  Its regret is measured against the best *sequence* of experts with
at most ``m`` switches, which is why it can beat the best fixed expert on
non-stationary data.

``alpha = 0`` recovers Hedge exactly.
"""

from __future__ import annotations

import numpy as np


class FixedShare:
    """Fixed-Share forecaster.

    Parameters
    ----------
    n_experts : int
        Number of experts to aggregate.
    eta : float
        Learning rate (positive).
    alpha : float
        Share parameter in ``[0, 1)``.  The mass redistributed uniformly at
        every round.  A good default when the number of switches ``m`` and
        the horizon ``T`` are known is ``alpha = m / (T - 1)``.
    """

    USES_ENSEMBLE_LOSS = False

    def __init__(self, n_experts: int, eta: float, alpha: float = 0.01) -> None:
        if n_experts < 1:
            raise ValueError("n_experts must be >= 1")
        if eta <= 0:
            raise ValueError("eta must be positive")
        if not 0.0 <= alpha < 1.0:
            raise ValueError("alpha must be in [0, 1)")

        self.n_experts = n_experts
        self.eta = eta
        self.alpha = alpha
        self.w: np.ndarray = np.full(n_experts, 1.0 / n_experts, dtype=np.float64)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, expert_predictions: np.ndarray) -> float:
        """Weighted-average prediction."""
        expert_predictions = np.asarray(expert_predictions, dtype=np.float64)
        if expert_predictions.shape != (self.n_experts,):
            raise ValueError(
                f"Expected array of length {self.n_experts}, "
                f"got shape {expert_predictions.shape}"
            )
        return float(np.dot(self.w, expert_predictions))

    def update(self, losses: np.ndarray) -> None:
        """Exponential update followed by the uniform share step."""
        losses = np.asarray(losses, dtype=np.float64)
        if losses.shape != (self.n_experts,):
            raise ValueError(
                f"Expected array of length {self.n_experts}, "
                f"got shape {losses.shape}"
            )

        # Exponential weights, computed in log-space for stability.
        log_w = np.log(np.maximum(self.w, 1e-300)) - self.eta * losses
        log_w -= log_w.max()
        w = np.exp(log_w)
        w /= w.sum()

        # Share step: move `alpha` of the mass back to the uniform prior.
        if self.alpha > 0.0:
            w = (1.0 - self.alpha) * w + self.alpha / self.n_experts

        self.w = w

    def get_weights(self) -> np.ndarray:
        """Return the current normalised weights."""
        return self.w.copy()

    def get_top_k(self, k: int) -> list[tuple[int, float]]:
        """Return the top-k experts as ``(index, weight)``, descending."""
        if k < 1:
            raise ValueError("k must be >= 1")
        k = min(k, self.n_experts)
        top = np.argsort(self.w)[-k:][::-1]
        return [(int(i), float(self.w[i])) for i in top]
