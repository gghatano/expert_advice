"""Tests for the Hedge ensemble, loss functions, and scaling utilities."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.ensemble.hedge import Hedge
from src.ensemble.loss import mae_loss, rmse_loss, smape_loss
from src.ensemble.meta_eta import MetaEtaHedge
from src.ensemble.scaling import by_train_mae, relative


# =====================================================================
# Hedge – weight monotonicity
# =====================================================================


class TestHedgeMonotonicity:
    """Experts with smaller losses must gain relative weight."""

    def test_lower_loss_gets_higher_weight(self) -> None:
        h = Hedge(n_experts=3, eta=1.0)
        # Expert 0 is best, expert 2 is worst.
        losses = np.array([0.1, 0.5, 1.0])
        h.update(losses)
        w = h.get_weights()
        assert w[0] > w[1] > w[2]

    def test_repeated_updates_amplify_gap(self) -> None:
        h = Hedge(n_experts=2, eta=0.5)
        for _ in range(10):
            h.update(np.array([0.2, 0.8]))
        w = h.get_weights()
        assert w[0] > w[1]
        # After 10 rounds the gap should be substantial.
        assert w[0] > 0.9

    def test_equal_losses_keep_uniform(self) -> None:
        h = Hedge(n_experts=4, eta=1.0)
        h.update(np.array([1.0, 1.0, 1.0, 1.0]))
        w = h.get_weights()
        np.testing.assert_allclose(w, 0.25, atol=1e-12)


# =====================================================================
# Hedge – numerical stability
# =====================================================================


class TestHedgeStability:
    """log-weight management must avoid overflow / underflow."""

    def test_huge_losses_no_overflow(self) -> None:
        h = Hedge(n_experts=3, eta=1.0)
        h.update(np.array([1e6, 1e6 + 1, 1e6 + 2]))
        w = h.get_weights()
        assert np.all(np.isfinite(w))
        assert math.isclose(w.sum(), 1.0, rel_tol=1e-9)

    def test_zero_losses(self) -> None:
        h = Hedge(n_experts=3, eta=1.0)
        h.update(np.array([0.0, 0.0, 0.0]))
        w = h.get_weights()
        np.testing.assert_allclose(w, 1.0 / 3, atol=1e-12)

    def test_many_rounds_stay_finite(self) -> None:
        rng = np.random.default_rng(42)
        h = Hedge(n_experts=5, eta=0.1)
        for _ in range(10_000):
            losses = rng.random(5)
            h.update(losses)
        w = h.get_weights()
        assert np.all(np.isfinite(w))
        assert math.isclose(w.sum(), 1.0, rel_tol=1e-9)

    def test_extreme_eta(self) -> None:
        h = Hedge(n_experts=2, eta=100.0)
        h.update(np.array([0.0, 1.0]))
        w = h.get_weights()
        assert np.all(np.isfinite(w))
        # Expert 0 should dominate.
        assert w[0] > 0.99


# =====================================================================
# Hedge – predict
# =====================================================================


class TestHedgePredict:
    """predict() must return a sensible weighted average."""

    def test_uniform_weights_give_mean(self) -> None:
        h = Hedge(n_experts=3, eta=1.0)
        preds = np.array([10.0, 20.0, 30.0])
        assert math.isclose(h.predict(preds), 20.0, rel_tol=1e-9)

    def test_after_update_closer_to_best(self) -> None:
        h = Hedge(n_experts=2, eta=2.0)
        # Expert 0 had lower loss.
        h.update(np.array([0.1, 1.0]))
        pred = h.predict(np.array([100.0, 200.0]))
        # Should be closer to 100 than to 200.
        assert pred < 150.0

    def test_prediction_within_expert_range(self) -> None:
        h = Hedge(n_experts=4, eta=0.5)
        h.update(np.array([0.3, 0.7, 0.1, 0.9]))
        preds = np.array([5.0, 10.0, 15.0, 20.0])
        result = h.predict(preds)
        assert 5.0 <= result <= 20.0


# =====================================================================
# Hedge – get_top_k
# =====================================================================


class TestHedgeTopK:
    def test_top_k_order(self) -> None:
        h = Hedge(n_experts=4, eta=1.0)
        h.update(np.array([3.0, 1.0, 4.0, 2.0]))
        top2 = h.get_top_k(2)
        assert len(top2) == 2
        # Expert 1 (loss=1) should be first, expert 3 (loss=2) second.
        assert top2[0][0] == 1
        assert top2[1][0] == 3

    def test_top_k_clamps_to_n_experts(self) -> None:
        h = Hedge(n_experts=2, eta=1.0)
        top = h.get_top_k(100)
        assert len(top) == 2


# =====================================================================
# Loss functions
# =====================================================================


class TestLossFunctions:
    def test_mae_basic(self) -> None:
        assert mae_loss(3.0, 1.0) == 2.0
        assert mae_loss(1.0, 3.0) == 2.0

    def test_mae_zero(self) -> None:
        assert mae_loss(5.0, 5.0) == 0.0

    def test_smape_symmetric(self) -> None:
        a = smape_loss(100.0, 110.0)
        b = smape_loss(110.0, 100.0)
        assert math.isclose(a, b, rel_tol=1e-9)

    def test_smape_zero_both(self) -> None:
        # Should not raise; result should be near 0.
        result = smape_loss(0.0, 0.0)
        assert result < 1e-4

    def test_rmse_equals_mae_single_point(self) -> None:
        assert math.isclose(rmse_loss(7.0, 4.0), mae_loss(7.0, 4.0))


# =====================================================================
# Scaling utilities
# =====================================================================


class TestScaling:
    def test_by_train_mae_basic(self) -> None:
        assert math.isclose(by_train_mae(10.0, 5.0), 2.0)

    def test_by_train_mae_rejects_non_positive(self) -> None:
        with pytest.raises(ValueError):
            by_train_mae(1.0, 0.0)
        with pytest.raises(ValueError):
            by_train_mae(1.0, -1.0)

    def test_relative_basic(self) -> None:
        result = relative(5.0, 100.0)
        assert math.isclose(result, 0.05, rel_tol=1e-4)

    def test_relative_zero_ytrue(self) -> None:
        # Should not raise; denominator uses epsilon.
        result = relative(1.0, 0.0)
        assert result > 0


# =====================================================================
# MetaEtaHedge – basic behaviour
# =====================================================================


class TestMetaEtaHedgeBasic:
    """MetaEtaHedge must initialise and run without errors."""

    def test_default_etas(self) -> None:
        m = MetaEtaHedge(n_experts=3)
        assert len(m.etas) == 11
        assert math.isclose(m.etas[0], 1.0)
        assert m.etas[-1] < 0.002

    def test_custom_etas(self) -> None:
        m = MetaEtaHedge(n_experts=2, etas=[0.1, 0.5, 1.0])
        assert len(m.etas) == 3

    def test_predict_returns_float(self) -> None:
        m = MetaEtaHedge(n_experts=3)
        preds = np.array([10.0, 20.0, 30.0])
        result = m.predict(preds)
        assert isinstance(result, float)

    def test_predict_within_expert_range(self) -> None:
        m = MetaEtaHedge(n_experts=3)
        preds = np.array([5.0, 15.0, 25.0])
        result = m.predict(preds)
        assert 5.0 <= result <= 25.0

    def test_predict_update_cycle(self) -> None:
        """predict/update cycle should work repeatedly without error."""
        m = MetaEtaHedge(n_experts=3)
        for _ in range(20):
            pred = m.predict(np.array([1.0, 2.0, 3.0]))
            assert np.isfinite(pred)
            m.update(np.array([0.1, 0.5, 0.9]))


# =====================================================================
# MetaEtaHedge – eta weight dynamics
# =====================================================================


class TestMetaEtaWeights:
    """Eta weights should change over time as the meta Hedge learns."""

    def test_eta_weights_change_after_update(self) -> None:
        m = MetaEtaHedge(n_experts=3)
        initial_eta_w = m.get_eta_weights().copy()

        # Run several predict/update rounds.
        for _ in range(5):
            m.predict(np.array([1.0, 2.0, 3.0]))
            m.update(np.array([0.1, 0.5, 1.0]))

        updated_eta_w = m.get_eta_weights()
        # Weights must have shifted from uniform.
        assert not np.allclose(initial_eta_w, updated_eta_w, atol=1e-10)

    def test_get_eta_weights_sum_to_one(self) -> None:
        m = MetaEtaHedge(n_experts=4)
        m.predict(np.array([1.0, 2.0, 3.0, 4.0]))
        m.update(np.array([0.2, 0.4, 0.6, 0.8]))
        w = m.get_eta_weights()
        assert math.isclose(w.sum(), 1.0, rel_tol=1e-9)


# =====================================================================
# MetaEtaHedge – effective eta
# =====================================================================


class TestMetaEtaEffective:
    """get_effective_eta must return a valid float."""

    def test_effective_eta_is_float(self) -> None:
        m = MetaEtaHedge(n_experts=2)
        assert isinstance(m.get_effective_eta(), float)

    def test_effective_eta_in_range(self) -> None:
        etas = [0.1, 0.5, 1.0]
        m = MetaEtaHedge(n_experts=2, etas=etas)
        eff = m.get_effective_eta()
        assert min(etas) <= eff <= max(etas)


# =====================================================================
# MetaEtaHedge – composite weights
# =====================================================================


class TestMetaEtaCompositeWeights:
    """get_weights must return proper expert weights."""

    def test_weights_sum_to_one(self) -> None:
        m = MetaEtaHedge(n_experts=4)
        w = m.get_weights()
        assert math.isclose(w.sum(), 1.0, rel_tol=1e-9)
        assert len(w) == 4

    def test_weights_respond_to_losses(self) -> None:
        m = MetaEtaHedge(n_experts=3)
        for _ in range(10):
            m.predict(np.array([1.0, 2.0, 3.0]))
            m.update(np.array([0.1, 0.5, 1.0]))
        w = m.get_weights()
        # Expert 0 (lowest loss) should have highest weight.
        assert w[0] > w[1] > w[2]
