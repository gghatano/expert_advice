"""Hedge (Exponentially Weighted) Ensemble Algorithm."""

from __future__ import annotations

import numpy as np


class Hedge:
    """Hedge algorithm for online expert aggregation.

    Maintains log-weights for numerical stability and produces
    normalised weights via a softmax-style computation.

    Parameters
    ----------
    n_experts : int
        Number of experts to aggregate.
    eta : float
        Learning rate (positive). Larger values make the algorithm
        more aggressive in down-weighting poor experts.
    """

    def __init__(self, n_experts: int, eta: float) -> None:
        if n_experts < 1:
            raise ValueError("n_experts must be >= 1")
        if eta <= 0:
            raise ValueError("eta must be positive")

        self.n_experts = n_experts
        self.eta = eta
        # Uniform initialisation in log-space (all zeros).
        self.log_w: np.ndarray = np.zeros(n_experts, dtype=np.float64)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalised_weights(self) -> np.ndarray:
        """Return normalised weights via numerically-stable softmax."""
        # Shift by max for numerical stability.
        shifted = self.log_w - self.log_w.max()
        exp_w = np.exp(shifted)
        return exp_w / exp_w.sum()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, expert_predictions: np.ndarray) -> float:
        """Weighted average prediction.

        Parameters
        ----------
        expert_predictions : np.ndarray
            1-D array of length ``n_experts`` with each expert's
            prediction for the current time step.

        Returns
        -------
        float
            The ensemble prediction.
        """
        expert_predictions = np.asarray(expert_predictions, dtype=np.float64)
        if expert_predictions.shape != (self.n_experts,):
            raise ValueError(
                f"Expected array of length {self.n_experts}, "
                f"got shape {expert_predictions.shape}"
            )
        weights = self._normalised_weights()
        return float(np.dot(weights, expert_predictions))

    def update(self, losses: np.ndarray) -> None:
        """Exponential weight update.

        ``log_w[i] += -eta * losses[i]``

        Parameters
        ----------
        losses : np.ndarray
            1-D array of length ``n_experts`` with non-negative losses
            observed for each expert.
        """
        losses = np.asarray(losses, dtype=np.float64)
        if losses.shape != (self.n_experts,):
            raise ValueError(
                f"Expected array of length {self.n_experts}, "
                f"got shape {losses.shape}"
            )
        self.log_w -= self.eta * losses

        # Re-centre log-weights to prevent drift toward -inf or +inf.
        self.log_w -= self.log_w.max()

    def get_weights(self) -> np.ndarray:
        """Return current normalised weights as a 1-D array."""
        return self._normalised_weights()

    def get_top_k(self, k: int) -> list[tuple[int, float]]:
        """Return the top-k experts by weight.

        Parameters
        ----------
        k : int
            Number of top experts to return.

        Returns
        -------
        list[tuple[int, float]]
            List of ``(expert_index, weight)`` sorted descending by weight.
        """
        if k < 1:
            raise ValueError("k must be >= 1")
        k = min(k, self.n_experts)
        weights = self._normalised_weights()
        # argsort ascending, take last k, reverse.
        top_indices = np.argsort(weights)[-k:][::-1]
        return [(int(idx), float(weights[idx])) for idx in top_indices]
