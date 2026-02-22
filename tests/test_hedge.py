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


class TestHedgeValidation:
    """Hedge constructor and method input validation."""

    def test_n_experts_less_than_one_raises(self) -> None:
        with pytest.raises(ValueError, match="n_experts must be >= 1"):
            Hedge(n_experts=0, eta=1.0)

    def test_n_experts_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="n_experts must be >= 1"):
            Hedge(n_experts=-5, eta=1.0)

    def test_eta_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="eta must be positive"):
            Hedge(n_experts=3, eta=0.0)

    def test_eta_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="eta must be positive"):
            Hedge(n_experts=3, eta=-0.5)

    def test_predict_wrong_shape_raises(self) -> None:
        h = Hedge(n_experts=3, eta=1.0)
        with pytest.raises(ValueError, match="Expected array of length 3"):
            h.predict(np.array([1.0, 2.0]))

    def test_predict_2d_array_raises(self) -> None:
        h = Hedge(n_experts=3, eta=1.0)
        with pytest.raises(ValueError):
            h.predict(np.array([[1.0, 2.0, 3.0]]))

    def test_update_wrong_shape_raises(self) -> None:
        h = Hedge(n_experts=3, eta=1.0)
        with pytest.raises(ValueError, match="Expected array of length 3"):
            h.update(np.array([0.1, 0.2]))

    def test_get_top_k_zero_raises(self) -> None:
        h = Hedge(n_experts=3, eta=1.0)
        with pytest.raises(ValueError, match="k must be >= 1"):
            h.get_top_k(0)


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


# =====================================================================
# MetaEtaHedge – eta selection validity
# =====================================================================


class TestMetaEtaSelection:
    """When loss scale is known, MetaEtaHedge should favour an appropriate eta."""

    def test_consistent_pattern_small_scale_concentrates_on_large_eta(self) -> None:
        """With consistent loss patterns at small scale, large eta is favoured.

        When losses have a stable ordering (expert 0 always best) and the
        scale is small, the high-eta sub-Hedges can track the pattern
        without destabilising, so the meta-level should concentrate weight
        on larger eta values.
        """
        etas = [0.01, 0.1, 0.5, 1.0]
        m = MetaEtaHedge(n_experts=3, etas=etas)
        for _ in range(200):
            m.predict(np.array([1.0, 2.0, 3.0]))
            # Expert 0 is always best, consistent pattern, small scale.
            m.update(np.array([0.001, 0.005, 0.009]))
        eff = m.get_effective_eta()
        # The largest eta should dominate; effective eta should be close
        # to max(etas).
        assert eff > 0.8, (
            f"effective eta {eff:.4f} should be large (>0.8) for small-scale consistent losses"
        )

    def test_large_scale_effective_eta_lower_than_small_scale(self) -> None:
        """With identical loss *pattern* but larger scale, effective eta is lower.

        The meta-level normalises losses, but high-eta sub-Hedges become
        more aggressive and may over-concentrate when losses are large,
        reducing their meta-level performance relative to moderate-eta
        sub-Hedges.
        """
        etas = [0.01, 0.1, 0.5, 1.0]

        # Small scale.
        m_small = MetaEtaHedge(n_experts=3, etas=etas)
        for _ in range(200):
            m_small.predict(np.array([1.0, 2.0, 3.0]))
            m_small.update(np.array([0.1, 0.5, 0.9]))
        eta_small = m_small.get_effective_eta()

        # Large scale.
        m_large = MetaEtaHedge(n_experts=3, etas=etas)
        for _ in range(200):
            m_large.predict(np.array([1.0, 2.0, 3.0]))
            m_large.update(np.array([1.0, 5.0, 9.0]))
        eta_large = m_large.get_effective_eta()

        assert eta_small >= eta_large, (
            f"Small-scale effective eta ({eta_small:.4f}) should be >= "
            f"large-scale effective eta ({eta_large:.4f})"
        )

    def test_effective_eta_within_candidate_range(self) -> None:
        """After updates, effective eta must remain within candidate bounds."""
        etas = [0.01, 0.1, 0.5, 1.0]
        m = MetaEtaHedge(n_experts=3, etas=etas)
        rng = np.random.default_rng(2)
        for _ in range(100):
            m.predict(rng.random(3))
            m.update(rng.random(3) * 0.5)
        eff = m.get_effective_eta()
        assert min(etas) <= eff <= max(etas)


# =====================================================================
# MetaEtaHedge – regret non-divergence
# =====================================================================


class TestMetaEtaRegretNonDivergence:
    """Weights must stay finite and well-formed under sustained random losses."""

    def test_random_losses_1000_rounds(self) -> None:
        rng = np.random.default_rng(42)
        m = MetaEtaHedge(n_experts=5)
        for _ in range(1000):
            preds = rng.random(5) * 100
            m.predict(preds)
            m.update(rng.random(5))

        w = m.get_weights()
        assert w.shape == (5,)
        assert np.all(np.isfinite(w)), "Weights contain NaN or Inf"
        assert np.all(w >= 0), "Weights contain negative values"
        assert math.isclose(w.sum(), 1.0, rel_tol=1e-9), f"Weights sum to {w.sum()}, not 1"

    def test_eta_weights_stay_valid_after_many_rounds(self) -> None:
        rng = np.random.default_rng(99)
        m = MetaEtaHedge(n_experts=4)
        for _ in range(1000):
            m.predict(rng.random(4))
            m.update(rng.random(4))

        eta_w = m.get_eta_weights()
        assert np.all(np.isfinite(eta_w)), "Eta weights contain NaN or Inf"
        assert np.all(eta_w >= 0), "Eta weights contain negative values"
        assert math.isclose(eta_w.sum(), 1.0, rel_tol=1e-9)

    def test_effective_eta_stays_finite(self) -> None:
        rng = np.random.default_rng(7)
        m = MetaEtaHedge(n_experts=3)
        for _ in range(1000):
            m.predict(rng.random(3))
            m.update(rng.random(3) * 10)
        eff = m.get_effective_eta()
        assert np.isfinite(eff), "Effective eta is not finite"
        assert eff > 0, "Effective eta should be positive"


# =====================================================================
# MetaEtaHedge – consistency with single-eta Hedge
# =====================================================================


class TestMetaEtaSingleEtaConsistency:
    """MetaEtaHedge with one candidate eta should behave like plain Hedge."""

    def test_single_eta_matches_hedge_weights(self) -> None:
        eta = 0.1
        n = 4
        meta = MetaEtaHedge(n_experts=n, etas=[eta])
        hedge = Hedge(n_experts=n, eta=eta)

        rng = np.random.default_rng(12)
        for _ in range(50):
            losses = rng.random(n)
            meta.predict(rng.random(n))  # required before update
            meta.update(losses)
            hedge.update(losses)

        np.testing.assert_allclose(
            meta.get_weights(), hedge.get_weights(), atol=1e-12,
            err_msg="Single-eta MetaEtaHedge weights diverge from Hedge",
        )

    def test_single_eta_matches_hedge_predict(self) -> None:
        eta = 0.3
        n = 3
        meta = MetaEtaHedge(n_experts=n, etas=[eta])
        hedge = Hedge(n_experts=n, eta=eta)

        rng = np.random.default_rng(55)
        for _ in range(30):
            losses = rng.random(n)
            expert_preds = rng.random(n) * 100

            meta_pred = meta.predict(expert_preds)
            hedge_pred = hedge.predict(expert_preds)

            # Before any update both should agree; after updates they
            # should still track closely.
            assert math.isclose(meta_pred, hedge_pred, rel_tol=1e-9), (
                f"Predictions differ: meta={meta_pred}, hedge={hedge_pred}"
            )

            meta.update(losses)
            hedge.update(losses)


# =====================================================================
# MetaEtaHedge – eta weight responsiveness
# =====================================================================


class TestMetaEtaResponsiveness:
    """Effective eta should adapt when the loss regime changes."""

    def test_effective_eta_shifts_on_regime_change(self) -> None:
        etas = [0.01, 0.05, 0.1, 0.5, 1.0]
        m = MetaEtaHedge(n_experts=3, etas=etas)
        rng = np.random.default_rng(77)

        # Phase 1: large losses (scale ~1.0) for 100 rounds.
        for _ in range(100):
            m.predict(rng.random(3))
            m.update(rng.random(3) * 1.0)
        eta_after_large = m.get_effective_eta()

        # Phase 2: small losses (scale ~0.01) for 100 rounds.
        for _ in range(100):
            m.predict(rng.random(3))
            m.update(rng.random(3) * 0.01)
        eta_after_small = m.get_effective_eta()

        # The effective eta should have increased after switching to small
        # losses (smaller losses allow more aggressive learning rates).
        assert eta_after_small > eta_after_large, (
            f"Effective eta did not increase after regime change: "
            f"large-loss phase={eta_after_large:.4f}, "
            f"small-loss phase={eta_after_small:.4f}"
        )

    def test_eta_weights_are_non_stationary(self) -> None:
        """Eta weights at round 50 should differ from those at round 150."""
        m = MetaEtaHedge(n_experts=3)
        rng = np.random.default_rng(33)

        for _ in range(50):
            m.predict(rng.random(3))
            m.update(rng.random(3) * 1.0)
        snapshot_50 = m.get_eta_weights().copy()

        for _ in range(100):
            m.predict(rng.random(3))
            m.update(rng.random(3) * 0.01)
        snapshot_150 = m.get_eta_weights().copy()

        assert not np.allclose(snapshot_50, snapshot_150, atol=1e-6), (
            "Eta weights did not change across regime shift"
        )
