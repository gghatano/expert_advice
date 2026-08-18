"""ML-Poly: polynomially weighted average forecaster with per-expert learning rates.

Reference
---------
Gaillard, Stoltz & van Erven (2014), *A second-order bound with excess
losses*, COLT.  This is the ``MLpol`` rule popularised by the R package
``opera`` and used in operational electricity-load forecasting at EDF.

Why it behaves differently from Hedge
-------------------------------------
Hedge accumulates each expert's **loss**.  ML-Poly accumulates each
expert's **regret** relative to the ensemble itself::

    r_{i,t} = loss(ensemble_t) - loss(expert_i,t)
    R_{i,t} = R_{i,t-1} + r_{i,t}

and puts weight only on experts the ensemble currently *regrets* not having
followed::

    p_{i,t} propto eta_{i,t-1} * max(R_{i,t-1}, 0)
    eta_{i,t} = 1 / (1 + sum_s r_{i,s}^2)

The learning rate is per-expert and self-tuned from the observed variance of
the excess losses, so there is nothing to choose by hand.  Because the
reference point is the ensemble rather than a fixed prior, the aggregate can
end up strictly better than every individual expert when their errors are
complementary -- something plain Hedge cannot do under absolute loss.
"""

from __future__ import annotations

import numpy as np


class MLPoly:
    """ML-Poly (MLpol) forecaster.

    Unlike :class:`~src.ensemble.hedge.Hedge`, :meth:`update` also needs the
    loss the *ensemble* itself incurred; the runner checks the
    ``USES_ENSEMBLE_LOSS`` flag to know this.

    Parameters
    ----------
    n_experts : int
        Number of experts to aggregate.
    """

    USES_ENSEMBLE_LOSS = True

    def __init__(self, n_experts: int) -> None:
        if n_experts < 1:
            raise ValueError("n_experts must be >= 1")

        self.n_experts = n_experts
        # Cumulative regret of the ensemble against each expert.
        self.regret: np.ndarray = np.zeros(n_experts, dtype=np.float64)
        # Cumulative squared instantaneous regret, driving the learning rates.
        self.cum_sq: np.ndarray = np.zeros(n_experts, dtype=np.float64)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _learning_rates(self) -> np.ndarray:
        return 1.0 / (1.0 + self.cum_sq)

    def _weights(self) -> np.ndarray:
        scores = self._learning_rates() * np.maximum(self.regret, 0.0)
        total = scores.sum()
        if total <= 0.0:
            # No expert is regretted yet (t = 1, or the ensemble dominates).
            return np.full(self.n_experts, 1.0 / self.n_experts, dtype=np.float64)
        return scores / total

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

    def update(self, losses: np.ndarray, ensemble_loss: float) -> None:
        """Update cumulative regrets and per-expert learning rates.

        Parameters
        ----------
        losses : np.ndarray
            Per-expert losses for the round just played.
        ensemble_loss : float
            The loss incurred by this aggregator's own prediction.
        """
        losses = np.asarray(losses, dtype=np.float64)
        if losses.shape != (self.n_experts,):
            raise ValueError(
                f"Expected array of length {self.n_experts}, "
                f"got shape {losses.shape}"
            )

        instantaneous = float(ensemble_loss) - losses
        self.regret += instantaneous
        self.cum_sq += instantaneous ** 2

    def get_weights(self) -> np.ndarray:
        """Return the current normalised weights."""
        return self._weights()

    def get_top_k(self, k: int) -> list[tuple[int, float]]:
        """Return the top-k experts as ``(index, weight)``, descending."""
        if k < 1:
            raise ValueError("k must be >= 1")
        k = min(k, self.n_experts)
        w = self._weights()
        top = np.argsort(w)[-k:][::-1]
        return [(int(i), float(w[i])) for i in top]
