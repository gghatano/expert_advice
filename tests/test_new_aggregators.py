"""FixedShare / AdaHedge / MLPoly と runner の単体テスト."""

from __future__ import annotations

import numpy as np
import pytest

from src.ensemble.adahedge import AdaHedge
from src.ensemble.fixed_share import FixedShare
from src.ensemble.hedge import Hedge
from src.ensemble.meta_eta import MetaEtaHedge
from src.ensemble.ml_poly import MLPoly
from src.ensemble.runner import (
    EqualWeight,
    FollowTheLeader,
    best_fixed_expert,
    expert_loss_matrix,
    oracle_switching_loss,
    run_online,
)


# ---------------------------------------------------------------------------
# 共通フィクスチャ
# ---------------------------------------------------------------------------


@pytest.fixture
def regime_data() -> tuple[np.ndarray, np.ndarray]:
    """前半は expert0、後半は expert1 が正解になる予測行列と真値."""
    rng = np.random.RandomState(0)
    T = 400
    y = rng.normal(0, 1, T)
    P = np.zeros((T, 3))
    P[:, 0] = np.where(np.arange(T) < T // 2, y, y + 3.0)  # 前半だけ正確
    P[:, 1] = np.where(np.arange(T) < T // 2, y + 3.0, y)  # 後半だけ正確
    P[:, 2] = y + 1.5  # 常にそこそこ
    return y, P


# ---------------------------------------------------------------------------
# FixedShare
# ---------------------------------------------------------------------------


class TestFixedShare:
    def test_rejects_bad_params(self) -> None:
        with pytest.raises(ValueError):
            FixedShare(n_experts=0, eta=0.1)
        with pytest.raises(ValueError):
            FixedShare(n_experts=3, eta=0.0)
        with pytest.raises(ValueError):
            FixedShare(n_experts=3, eta=0.1, alpha=1.0)

    def test_alpha_zero_matches_hedge(self) -> None:
        """alpha=0 の Fixed-Share は Hedge と完全に一致する."""
        rng = np.random.RandomState(1)
        fs = FixedShare(n_experts=5, eta=0.3, alpha=0.0)
        hg = Hedge(n_experts=5, eta=0.3)
        for _ in range(50):
            losses = rng.rand(5)
            np.testing.assert_allclose(fs.get_weights(), hg.get_weights(), atol=1e-12)
            fs.update(losses)
            hg.update(losses)
        np.testing.assert_allclose(fs.get_weights(), hg.get_weights(), atol=1e-12)

    def test_weights_are_floored_by_alpha(self) -> None:
        """share ステップにより、どの重みも alpha/N を下回らない."""
        alpha, n = 0.1, 4
        fs = FixedShare(n_experts=n, eta=2.0, alpha=alpha)
        losses = np.array([0.0, 10.0, 10.0, 10.0])
        for _ in range(200):
            fs.update(losses)
        w = fs.get_weights()
        assert np.all(w >= alpha / n - 1e-12)
        assert w.sum() == pytest.approx(1.0)

    def test_recovers_after_regime_change(self, regime_data) -> None:
        """レジーム変化後、Fixed-Share は Hedge より速く追随する."""
        y, P = regime_data
        T = len(y)
        fs = run_online(P, y, FixedShare(3, eta=1.0, alpha=0.05), name="fs")
        hg = run_online(P, y, Hedge(3, eta=1.0), name="hedge")
        # 後半 (expert1 が正解の区間) の平均損失で比較
        assert fs.losses[T // 2:].mean() < hg.losses[T // 2:].mean()

    def test_beats_best_fixed_expert_on_switching_data(self, regime_data) -> None:
        """切り替えデータでは最良の固定 Expert すら上回りうる."""
        y, P = regime_data
        L = expert_loss_matrix(P, y)
        _, best_mean = best_fixed_expert(L)
        fs = run_online(P, y, FixedShare(3, eta=1.0, alpha=0.05), name="fs")
        assert fs.mean_loss < best_mean

    def test_predict_validates_shape(self) -> None:
        fs = FixedShare(n_experts=3, eta=0.1)
        with pytest.raises(ValueError):
            fs.predict(np.array([1.0, 2.0]))
        with pytest.raises(ValueError):
            fs.update(np.array([1.0, 2.0]))

    def test_top_k(self) -> None:
        fs = FixedShare(n_experts=4, eta=1.0, alpha=0.01)
        fs.update(np.array([0.0, 1.0, 2.0, 3.0]))
        top = fs.get_top_k(2)
        assert len(top) == 2
        assert top[0][0] == 0
        assert top[0][1] >= top[1][1]
        with pytest.raises(ValueError):
            fs.get_top_k(0)


# ---------------------------------------------------------------------------
# AdaHedge
# ---------------------------------------------------------------------------


class TestAdaHedge:
    def test_starts_as_follow_the_leader(self) -> None:
        """初期状態は eta=inf、重みは一様."""
        ah = AdaHedge(n_experts=4)
        assert np.isinf(ah.get_effective_eta())
        np.testing.assert_allclose(ah.get_weights(), np.full(4, 0.25))

    def test_eta_becomes_finite_and_weights_valid(self) -> None:
        rng = np.random.RandomState(2)
        ah = AdaHedge(n_experts=5)
        for _ in range(100):
            ah.update(rng.rand(5))
        w = ah.get_weights()
        assert np.isfinite(ah.get_effective_eta())
        assert ah.get_effective_eta() > 0
        assert w.sum() == pytest.approx(1.0)
        assert np.all(w >= 0)

    def test_mixability_gap_is_non_decreasing(self) -> None:
        rng = np.random.RandomState(3)
        ah = AdaHedge(n_experts=4)
        prev = ah.delta
        for _ in range(50):
            ah.update(rng.rand(4))
            assert ah.delta >= prev - 1e-12
            prev = ah.delta

    def test_concentrates_on_the_good_expert(self) -> None:
        ah = AdaHedge(n_experts=3)
        for _ in range(300):
            ah.update(np.array([0.0, 1.0, 1.0]))
        assert ah.get_weights()[0] > 0.9

    def test_competitive_with_tuned_hedge(self) -> None:
        """チューニング不要でも、固定 eta の Hedge と同程度の性能になる."""
        rng = np.random.RandomState(4)
        T, N = 500, 6
        y = rng.normal(0, 1, T)
        P = y[:, None] + rng.normal(0, 1, (T, N)) * np.linspace(0.2, 2.0, N)
        ada = run_online(P, y, AdaHedge(N), name="ada")
        best_hedge = min(
            run_online(P, y, Hedge(N, eta=e), name=f"h{e}").mean_loss
            for e in [0.01, 0.1, 1.0]
        )
        assert ada.mean_loss < best_hedge * 1.10

    def test_validates_shape(self) -> None:
        ah = AdaHedge(n_experts=3)
        with pytest.raises(ValueError):
            ah.update(np.array([1.0]))
        with pytest.raises(ValueError):
            ah.predict(np.array([1.0]))
        with pytest.raises(ValueError):
            AdaHedge(n_experts=0)


# ---------------------------------------------------------------------------
# MLPoly
# ---------------------------------------------------------------------------


class TestMLPoly:
    def test_starts_uniform(self) -> None:
        mp = MLPoly(n_experts=4)
        np.testing.assert_allclose(mp.get_weights(), np.full(4, 0.25))
        assert mp.USES_ENSEMBLE_LOSS is True

    def test_weights_track_positive_regret(self) -> None:
        mp = MLPoly(n_experts=3)
        # expert0 だけがアンサンブルより良い → regret が正になる
        mp.update(np.array([0.0, 1.0, 1.0]), ensemble_loss=0.7)
        w = mp.get_weights()
        assert w[0] > w[1]
        assert w.sum() == pytest.approx(1.0)

    def test_beats_every_expert_when_errors_cancel(self) -> None:
        """誤差が打ち消し合う Expert 群では、全 Expert を上回りうる."""
        rng = np.random.RandomState(5)
        T = 3000
        y = np.zeros(T)
        noise = rng.normal(0, 1.0, T)
        P = np.column_stack([y + noise, y - noise, y + rng.normal(0, 1.0, T)])
        res = run_online(P, y, MLPoly(3), name="mlpol", eval_loss="sq", update_loss="sq")
        L = expert_loss_matrix(P, y, loss="sq")
        assert res.mean_loss < L.mean(axis=0).min()

    def test_validates_shape(self) -> None:
        mp = MLPoly(n_experts=3)
        with pytest.raises(ValueError):
            mp.update(np.array([1.0]), 0.5)
        with pytest.raises(ValueError):
            MLPoly(n_experts=0)


# ---------------------------------------------------------------------------
# ベースラインと runner
# ---------------------------------------------------------------------------


class TestRunner:
    def test_equal_weight_is_the_plain_mean(self) -> None:
        rng = np.random.RandomState(6)
        P = rng.rand(20, 4)
        y = rng.rand(20)
        res = run_online(P, y, EqualWeight(4), name="eq")
        np.testing.assert_allclose(res.predictions, P.mean(axis=1))

    def test_follow_the_leader_picks_the_best_so_far(self) -> None:
        y = np.zeros(50)
        P = np.column_stack([np.zeros(50), np.ones(50)])
        res = run_online(P, y, FollowTheLeader(2), name="ftl")
        # 初手は同点で 0.5、以降は expert0 に張り付く
        assert res.predictions[0] == pytest.approx(0.5)
        np.testing.assert_allclose(res.predictions[1:], 0.0)

    def test_best_fixed_expert(self) -> None:
        L = np.array([[1.0, 0.5], [1.0, 0.0]])
        idx, mean = best_fixed_expert(L)
        assert idx == 1
        assert mean == pytest.approx(0.25)

    def test_oracle_switching_beats_best_fixed(self, regime_data) -> None:
        y, P = regime_data
        L = expert_loss_matrix(P, y)
        T = len(y)
        _, best_mean = best_fixed_expert(L)
        oracle = oracle_switching_loss(L, [(0, T // 2), (T // 2, T)])
        assert oracle < best_mean

    def test_oracle_switching_rejects_empty_segments(self) -> None:
        with pytest.raises(ValueError):
            oracle_switching_loss(np.zeros((10, 2)), [])

    def test_update_loss_and_eval_loss_are_independent(self) -> None:
        """update_loss を変えても eval_loss の定義は変わらない."""
        rng = np.random.RandomState(7)
        P = rng.rand(100, 3)
        y = rng.rand(100)
        res = run_online(P, y, Hedge(3, eta=0.5), name="h", update_loss="sq", eval_loss="abs")
        np.testing.assert_allclose(res.losses, np.abs(y - res.predictions))

    def test_loss_scale_only_affects_weights(self) -> None:
        rng = np.random.RandomState(8)
        P = rng.rand(100, 3) * 100
        y = rng.rand(100) * 100
        a = run_online(P, y, Hedge(3, eta=0.5), name="a", loss_scale=1.0)
        b = run_online(P, y, Hedge(3, eta=0.5), name="b", loss_scale=50.0)
        assert not np.allclose(a.predictions, b.predictions)

    def test_validates_inputs(self) -> None:
        P = np.zeros((10, 3))
        y = np.zeros(10)
        with pytest.raises(ValueError):
            run_online(P, np.zeros(9), Hedge(3, eta=0.1), name="x")
        with pytest.raises(ValueError):
            run_online(P, y, Hedge(4, eta=0.1), name="x")
        with pytest.raises(ValueError):
            run_online(P, y, Hedge(3, eta=0.1), name="x", loss_scale=0.0)
        with pytest.raises(ValueError):
            run_online(np.zeros(10), y, Hedge(3, eta=0.1), name="x")

    def test_records_weight_snapshots(self) -> None:
        rng = np.random.RandomState(9)
        P = rng.rand(100, 3)
        y = rng.rand(100)
        res = run_online(
            P, y, Hedge(3, eta=0.5), name="h", record_weights=True, snapshot_every=10
        )
        assert res.weights is not None
        assert res.weights.shape == (10, 3)
        assert res.snapshot_steps is not None
        np.testing.assert_allclose(res.weights.sum(axis=1), 1.0)

    def test_cumulative_loss_property(self) -> None:
        P = np.zeros((5, 2))
        y = np.ones(5)
        res = run_online(P, y, EqualWeight(2), name="eq")
        np.testing.assert_allclose(res.cumulative_loss, np.arange(1, 6))
        assert res.mean_loss == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# CLI への接続 (run_experiment)
# ---------------------------------------------------------------------------


class TestCLIAggregatorWiring:
    """--aggregator で新しい集約アルゴリズムを選べることを確認する."""

    def _args(self, argv: list[str]):
        from src.run_experiment import build_parser

        return build_parser().parse_args(argv)

    def test_defaults_to_eta_mode_for_backward_compatibility(self) -> None:
        from src.run_experiment import _resolve_aggregator

        assert _resolve_aggregator(self._args([])) == "meta_grid"
        assert _resolve_aggregator(self._args(["--eta-mode", "fixed"])) == "fixed"

    def test_aggregator_takes_precedence(self) -> None:
        from src.run_experiment import _resolve_aggregator

        args = self._args(["--eta-mode", "fixed", "--aggregator", "adahedge"])
        assert _resolve_aggregator(args) == "adahedge"

    @pytest.mark.parametrize(
        "name, cls",
        [
            ("meta_grid", MetaEtaHedge),
            ("fixed", Hedge),
            ("fixed_share", FixedShare),
            ("adahedge", AdaHedge),
            ("ml_poly", MLPoly),
        ],
    )
    def test_creates_the_requested_algorithm(self, name: str, cls) -> None:
        from src.run_experiment import _create_ensemble

        ens = _create_ensemble(n_experts=5, aggregator=name, etas=None, alpha=0.01)
        assert isinstance(ens, cls)
        assert ens.n_experts == 5

    def test_alpha_is_passed_to_fixed_share(self) -> None:
        from src.run_experiment import _create_ensemble

        ens = _create_ensemble(n_experts=4, aggregator="fixed_share", etas=None, alpha=0.25)
        assert ens.alpha == pytest.approx(0.25)

    def test_unknown_aggregator_raises(self) -> None:
        from src.run_experiment import _create_ensemble

        with pytest.raises(ValueError, match="Unknown aggregator"):
            _create_ensemble(n_experts=3, aggregator="nope", etas=None)

    def test_online_phase_feeds_ensemble_loss_to_ml_poly(self) -> None:
        """ML-Poly は update(losses, ensemble_loss) で呼ばれる必要がある."""
        import pandas as pd

        from src.experts.naive import LastValue, SeasonalNaive
        from src.run_experiment import _run_online_phase

        idx = pd.date_range("2020-01-01", periods=80, freq="h")
        rng = np.random.RandomState(0)
        s = pd.Series(100 + rng.normal(0, 5, 80), index=idx)

        experts = [LastValue(), SeasonalNaive(season_length=24)]
        ensemble = MLPoly(n_experts=2)

        records, stats = _run_online_phase(
            experts=experts,
            ensemble=ensemble,
            history=s.iloc[:40],
            phase_data=s.iloc[40:],
            scale_loss_mode="by_train_mae",
            train_mae=5.0,
        )

        assert len(records) == 40
        assert stats["total_steps"] == 40
        # regret が蓄積し、重みが有効な分布のままであること
        assert np.isfinite(ensemble.regret).all()
        assert ensemble.get_weights().sum() == pytest.approx(1.0)
