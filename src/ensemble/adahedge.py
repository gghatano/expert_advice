"""AdaHedge: Hedge with a self-tuning learning rate.

Reference
---------
de Rooij, van Erven, Grünwald & Koolen (2014),
*Follow the leader if you can, hedge if you must*, JMLR 15, 1281-1316.

Idea
----
The learning rate is driven by the *mixability gap* actually observed so
far rather than by a horizon-dependent formula::

    h_t     = <w_t, l_t>                                   (expected loss)
    m_t     = -(1/eta_t) * log <w_t, exp(-eta_t * l_t)>    (mix loss)
    delta_t = h_t - m_t                     >= 0
    Delta_t = Delta_{t-1} + delta_t
    eta_{t+1} = log(N) / Delta_t

``Delta`` is exactly the quantity that appears in the regret bound, so
AdaHedge pays no tuning cost: it needs neither the horizon ``T`` nor the
loss range, and it degrades gracefully to Follow-the-Leader on easy data
(``Delta`` stays small, ``eta`` stays large).

This makes it the natural parameter-free comparison point for the grid of
learning rates used by :class:`~src.ensemble.meta_eta.MetaEtaHedge`.
"""

from __future__ import annotations

import math

import numpy as np


class AdaHedge:
    """AdaHedge forecaster (no hyper-parameters).

    Parameters
    ----------
    n_experts : int
        Number of experts to aggregate.
    """

    USES_ENSEMBLE_LOSS = False

    def __init__(self, n_experts: int) -> None:
        if n_experts < 1:
            raise ValueError("n_experts must be >= 1")

        self.n_experts = n_experts
        self.cum_losses: np.ndarray = np.zeros(n_experts, dtype=np.float64)
        # Cumulative mixability gap. Zero means "eta = infinity", i.e. the
        # algorithm behaves like Follow-the-Leader.
        self.delta: float = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def eta(self) -> float:
        """Current learning rate (``inf`` before any mixability gap is seen)."""
        if self.delta <= 0.0:
            return math.inf
        return math.log(self.n_experts) / self.delta

    def _weights(self) -> np.ndarray:
        eta = self.eta
        if math.isinf(eta):
            # Follow-the-Leader: uniform over the current minimisers.
            best = self.cum_losses <= self.cum_losses.min() + 1e-12
            return best / best.sum()
        shifted = -eta * (self.cum_losses - self.cum_losses.min())
        w = np.exp(shifted)
        return w / w.sum()

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
        return float(np.dot(self._weights(), expert_predictions))

    def update(self, losses: np.ndarray) -> None:
        """Accumulate losses and grow the learning-rate budget."""
        losses = np.asarray(losses, dtype=np.float64)
        if losses.shape != (self.n_experts,):
            raise ValueError(
                f"Expected array of length {self.n_experts}, "
                f"got shape {losses.shape}"
            )

        w = self._weights()
        eta = self.eta

        expected_loss = float(np.dot(w, losses))
        l_min = float(losses.min())

        if math.isinf(eta):
            mix_loss = l_min
        else:
            # -(1/eta) * log sum_i w_i exp(-eta l_i), shifted for stability.
            z = float(np.dot(w, np.exp(-eta * (losses - l_min))))
            mix_loss = l_min - math.log(max(z, 1e-300)) / eta

        # delta_t >= 0 up to floating-point noise.
        self.delta += max(0.0, expected_loss - mix_loss)
        self.cum_losses += losses

    def get_weights(self) -> np.ndarray:
        """Return the current normalised weights."""
        return self._weights()

    def get_effective_eta(self) -> float:
        """Return the current learning rate (``inf`` at the very start)."""
        return self.eta

    def get_top_k(self, k: int) -> list[tuple[int, float]]:
        """Return the top-k experts as ``(index, weight)``, descending."""
        if k < 1:
            raise ValueError("k must be >= 1")
        k = min(k, self.n_experts)
        w = self._weights()
        top = np.argsort(w)[-k:][::-1]
        return [(int(i), float(w[i])) for i in top]
