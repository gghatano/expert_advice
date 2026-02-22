"""Meta-eta Hedge: two-level Hedge with automatic learning rate selection."""

from __future__ import annotations

import numpy as np

from src.ensemble.hedge import Hedge


class MetaEtaHedge:
    """Two-level Hedge that automatically selects the best learning rate.

    Maintains a pool of Hedge instances, each with a different learning
    rate (eta). A meta-level Hedge tracks which eta performs best and
    blends their outputs accordingly.

    Parameters
    ----------
    n_experts : int
        Number of base experts to aggregate.
    etas : list[float] | None
        Candidate learning rates. Defaults to ``[2**(-k) for k in range(11)]``.
    """

    def __init__(self, n_experts: int, etas: list[float] | None = None) -> None:
        if n_experts < 1:
            raise ValueError("n_experts must be >= 1")

        self.n_experts = n_experts
        self.etas: list[float] = etas if etas is not None else [2 ** (-k) for k in range(11)]

        if len(self.etas) == 0:
            raise ValueError("etas must be non-empty")

        # One Hedge instance per candidate eta.
        self.hedge_pool: list[Hedge] = [
            Hedge(n_experts=n_experts, eta=eta) for eta in self.etas
        ]

        # Meta-level Hedge over the eta candidates.
        # Use a moderate fixed meta learning rate.
        self._meta_eta = 0.5
        self.meta_hedge = Hedge(n_experts=len(self.etas), eta=self._meta_eta)

        # Store the last prediction from each eta-Hedge so we can
        # compute meta-level losses after observing the true outcome.
        self._last_predictions: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, expert_predictions: np.ndarray) -> float:
        """Weighted-average prediction using meta-weighted eta-Hedges.

        Parameters
        ----------
        expert_predictions : np.ndarray
            1-D array of length ``n_experts``.

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

        # Collect predictions from each eta-Hedge.
        eta_preds = np.array(
            [h.predict(expert_predictions) for h in self.hedge_pool],
            dtype=np.float64,
        )
        self._last_predictions = eta_preds

        # Meta-weighted combination.
        meta_weights = self.meta_hedge.get_weights()
        return float(np.dot(meta_weights, eta_preds))

    def update(self, losses: np.ndarray) -> None:
        """Update all eta-Hedges and the meta-level Hedge.

        Parameters
        ----------
        losses : np.ndarray
            1-D array of length ``n_experts`` with non-negative losses
            for each base expert.
        """
        losses = np.asarray(losses, dtype=np.float64)
        if losses.shape != (self.n_experts,):
            raise ValueError(
                f"Expected array of length {self.n_experts}, "
                f"got shape {losses.shape}"
            )

        # Compute meta-level losses *before* updating the sub-Hedges.
        # Each eta-Hedge's meta-loss is its weighted-average expert loss
        # (i.e., the loss the eta-Hedge "incurred" via its weight allocation).
        # This keeps all values in the same loss scale.
        meta_losses = np.array(
            [float(np.dot(h.get_weights(), losses)) for h in self.hedge_pool],
            dtype=np.float64,
        )
        # Normalise meta-losses to [0, 1] range to keep meta Hedge stable.
        max_ml = meta_losses.max()
        if max_ml > 0:
            meta_losses = meta_losses / max_ml
        self.meta_hedge.update(meta_losses)

        # Update every eta-Hedge with the expert losses.
        for h in self.hedge_pool:
            h.update(losses)

    def get_weights(self) -> np.ndarray:
        """Return meta-weighted composite expert weights.

        Returns
        -------
        np.ndarray
            1-D array of length ``n_experts``, summing to 1.
        """
        meta_w = self.meta_hedge.get_weights()
        composite = np.zeros(self.n_experts, dtype=np.float64)
        for mw, h in zip(meta_w, self.hedge_pool):
            composite += mw * h.get_weights()
        return composite

    def get_eta_weights(self) -> np.ndarray:
        """Return the meta-level weights over candidate etas.

        Returns
        -------
        np.ndarray
            1-D array of length ``len(etas)``, summing to 1.
        """
        return self.meta_hedge.get_weights()

    def get_effective_eta(self) -> float:
        """Return the effective learning rate as a meta-weighted average.

        Returns
        -------
        float
            Weighted average of candidate etas.
        """
        meta_w = self.meta_hedge.get_weights()
        return float(np.dot(meta_w, self.etas))
