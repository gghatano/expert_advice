"""実験実行ループ・CLI エントリポイント.

Usage
-----
    python -m src.run_experiment --help
    python -m src.run_experiment --data-path data/raw/ --experts light30
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.load_uci import load_electricity
from src.data.preprocess import load_processed, preprocess, save_processed
from src.data.split import split_temporal
from src.ensemble.hedge import Hedge
from src.ensemble.loss import mae_loss
from src.ensemble.meta_eta import MetaEtaHedge
from src.ensemble.scaling import by_train_mae, relative
from src.experts.factory import create_experts

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# CLI 引数定義
# ------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """CLI 引数パーサーを構築する."""
    parser = argparse.ArgumentParser(
        description="Expert-Advise 実験実行ループ",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/raw/",
        help="生データパス (CSV ファイルまたはディレクトリ)",
    )
    parser.add_argument(
        "--agg",
        type=str,
        choices=["hourly_sum", "hourly_mean"],
        default="hourly_sum",
        help="集約方法",
    )
    parser.add_argument(
        "--impute",
        type=str,
        choices=["ffill0"],
        default="ffill0",
        help="欠損補完方法",
    )
    parser.add_argument(
        "--clip-quantile",
        type=float,
        default=None,
        help="外れ値クリップ閾値 (0-1の比率, 例: 0.999)。未指定なら OFF",
    )
    parser.add_argument(
        "--experts",
        type=str,
        choices=["light30", "light80"],
        default="light30",
        help="Expert preset 名",
    )
    parser.add_argument(
        "--eta-mode",
        type=str,
        choices=["meta_grid", "fixed"],
        default="meta_grid",
        help="学習率選択モード",
    )
    parser.add_argument(
        "--etas",
        type=str,
        default=None,
        help="eta 候補リスト (カンマ区切り、meta_grid 時に使用)",
    )
    parser.add_argument(
        "--scale-loss",
        type=str,
        choices=["by_train_mae", "relative"],
        default="by_train_mae",
        help="損失スケーリング方法",
    )
    parser.add_argument(
        "--n-series",
        type=int,
        default=None,
        help="処理する系列数 (先頭から)。未指定なら全系列",
    )
    parser.add_argument(
        "--series-sample",
        type=int,
        default=None,
        help="ランダムサンプルする系列数 (デバッグ用)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="乱数シード",
    )
    return parser


# ------------------------------------------------------------------
# ヘルパー関数
# ------------------------------------------------------------------


def _generate_run_id(args: argparse.Namespace) -> str:
    """日時 + 主要引数のハッシュから run_id を生成する."""
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    key_str = f"{args.agg}_{args.experts}_{args.eta_mode}_{args.scale_loss}_{args.seed}"
    short_hash = hashlib.md5(key_str.encode()).hexdigest()[:6]
    return f"{now}_{short_hash}"


def _set_seed(seed: int) -> None:
    """乱数シードを固定して再現性を確保する."""
    random.seed(seed)
    np.random.seed(seed)


def _parse_etas(etas_str: str | None) -> list[float] | None:
    """カンマ区切り文字列を float のリストに変換する."""
    if etas_str is None:
        return None
    return [float(x.strip()) for x in etas_str.split(",") if x.strip()]


def _compute_train_mae_naive(train_series: pd.Series) -> float:
    """Train 期間の LastValue (Naive) 予測の MAE を計算する.

    LastValue: y_hat[t] = y[t-1] として MAE を算出。
    """
    if len(train_series) < 2:
        return 1.0  # fallback
    y_true = train_series.values[1:]
    y_pred = train_series.values[:-1]
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return mae if mae > 0 else 1.0  # ゼロ除算回避


def _load_and_preprocess(args: argparse.Namespace) -> pd.DataFrame:
    """データ読み込みと前処理 (Parquet キャッシュ対応)."""
    # キャッシュ名を引数から一意に決定
    agg_method = "sum" if args.agg == "hourly_sum" else "mean"
    clip_tag = f"_clip{args.clip_quantile}" if args.clip_quantile is not None else ""
    cache_name = f"electricity_{agg_method}{clip_tag}.parquet"

    try:
        logger.info("Parquet キャッシュを探索: %s", cache_name)
        df = load_processed(name=cache_name)
        logger.info("キャッシュから読み込み完了: %d 行 x %d 列", len(df), len(df.columns))
        return df
    except FileNotFoundError:
        logger.info("キャッシュが見つかりません。生データから処理します。")

    # 生データ読み込み
    data_path = Path(args.data_path)
    if not data_path.is_absolute():
        data_path = _PROJECT_ROOT / data_path

    if data_path.is_file():
        raw_df = load_electricity(path=data_path)
    elif data_path.is_dir():
        raw_df = load_electricity(raw_dir=data_path)
    else:
        raise FileNotFoundError(
            f"データが見つかりません: {data_path}\n"
            "  --data-path に CSV ファイルまたはディレクトリを指定してください。"
        )

    # 前処理
    clip = args.clip_quantile is not None
    # clip_quantile は 0-1 比率 (例: 0.999) → 0-100 パーセンタイルに変換
    if clip:
        upper_pct = args.clip_quantile * 100  # 0.999 → 99.9
        lower_pct = 100.0 - upper_pct  # 0.1
    else:
        upper_pct = 99.5
        lower_pct = 0.5

    df = preprocess(
        raw_df,
        resample_method=agg_method,
        clip=clip,
        clip_lower_pct=lower_pct,
        clip_upper_pct=upper_pct,
    )

    # キャッシュ保存
    save_processed(df, name=cache_name)
    logger.info("前処理完了・キャッシュ保存: %d 行 x %d 列", len(df), len(df.columns))
    return df


def _select_series(
    columns: pd.Index,
    n_series: int | None,
    series_sample: int | None,
) -> list[str]:
    """系列を選択する."""
    all_cols = list(columns)
    if series_sample is not None:
        k = min(series_sample, len(all_cols))
        return random.sample(all_cols, k)
    if n_series is not None:
        return all_cols[: n_series]
    return all_cols


def _create_ensemble(
    n_experts: int,
    eta_mode: str,
    etas: list[float] | None,
) -> Hedge | MetaEtaHedge:
    """eta_mode に応じて Ensemble オブジェクトを生成する."""
    if eta_mode == "meta_grid":
        return MetaEtaHedge(n_experts=n_experts, etas=etas)
    else:
        # fixed mode: etas の先頭の値、または既定値 0.1
        eta = etas[0] if etas else 0.1
        return Hedge(n_experts=n_experts, eta=eta)


def _run_online_phase(
    experts: list,
    ensemble: Hedge | MetaEtaHedge,
    history: pd.Series,
    phase_data: pd.Series,
    scale_loss_mode: str,
    train_mae: float,
    desc: str = "",
    snapshot_interval: int = 24,
) -> tuple[list[dict], dict]:
    """逐次予測・重み更新ループを実行する.

    Returns
    -------
    tuple[list[dict], dict]
        (records, phase_stats) のタプル。
        records: 各時刻の予測結果レコードのリスト (既存と同一構造)
        phase_stats: Expert別統計・重みスナップショット
    """
    records = []
    n_experts = len(experts)
    expert_cum_losses = np.zeros(n_experts, dtype=np.float64)
    expert_cum_counts = np.zeros(n_experts, dtype=np.float64)
    expert_win_counts = np.zeros(n_experts, dtype=np.float64)
    expert_cum_weights = np.zeros(n_experts, dtype=np.float64)

    weight_snapshots: list[dict] = []

    # history を伸ばしていくためコピー
    running_history = history.copy()

    for t_idx, (tstamp, y_true) in enumerate(
        tqdm(phase_data.items(), desc=desc, leave=False, total=len(phase_data))
    ):
        # 1) Expert 全員の予測
        expert_preds = np.empty(n_experts, dtype=np.float64)
        for i, exp in enumerate(experts):
            try:
                pred = exp.predict_next(running_history, tstamp)
                expert_preds[i] = pred if np.isfinite(pred) else running_history.iloc[-1]
            except Exception:
                # fallback: 直近値
                expert_preds[i] = running_history.iloc[-1] if len(running_history) > 0 else 0.0

        # 2) Ensemble 予測 (Hedge重み付き)
        ensemble_pred = ensemble.predict(expert_preds)

        # 2b) 等重み平均予測
        equal_weight_pred = float(np.mean(expert_preds))

        # 3) Loss 計算
        raw_losses = np.array(
            [mae_loss(y_true, expert_preds[i]) for i in range(n_experts)],
            dtype=np.float64,
        )
        ensemble_raw_loss = mae_loss(y_true, ensemble_pred)
        equal_weight_loss = mae_loss(y_true, equal_weight_pred)

        # 4) スケーリング
        if scale_loss_mode == "by_train_mae":
            scaled_losses = np.array(
                [by_train_mae(l, train_mae) for l in raw_losses],
                dtype=np.float64,
            )
        else:  # relative
            scaled_losses = np.array(
                [relative(l, y_true) for l in raw_losses],
                dtype=np.float64,
            )

        # 5) 重み更新
        ensemble.update(scaled_losses)

        # 6) ベスト Expert を加算前の累積損失で選択（事前選択ベース）
        best_idx = int(np.argmin(expert_cum_losses))

        # 7) Expert別累積損失を更新（best_idx選択の後に加算）
        for i in range(n_experts):
            expert_cum_losses[i] += raw_losses[i]
            expert_cum_counts[i] += 1

        # 勝率計算用: 各時刻で最小損失の Expert をカウント
        min_loss = raw_losses.min()
        winners = np.where(np.isclose(raw_losses, min_loss))[0]
        for w in winners:
            expert_win_counts[w] += 1

        # 現在の重みを累積（平均重み計算用）
        current_weights = ensemble.get_weights()
        expert_cum_weights += current_weights

        # 重みスナップショット (snapshot_interval ごと)
        if t_idx % snapshot_interval == 0:
            # 上位5 Expert の重みを記録
            top_k = min(5, n_experts)
            top_indices = np.argsort(current_weights)[-top_k:][::-1]
            top_weights = [
                {"expert_idx": int(idx), "weight": float(current_weights[idx])}
                for idx in top_indices
            ]

            snapshot: dict = {
                "step": t_idx,
                "timestamp": str(tstamp),
                "top_weights": top_weights,
            }

            # eta 重み (MetaEtaHedge の場合のみ)
            if isinstance(ensemble, MetaEtaHedge):
                eta_w = ensemble.get_eta_weights()
                snapshot["eta_weights"] = [float(x) for x in eta_w]
                snapshot["etas"] = [float(x) for x in ensemble.etas]

            weight_snapshots.append(snapshot)

        records.append(
            {
                "timestamp": tstamp,
                "y_true": float(y_true),
                "y_pred": float(ensemble_pred),
                "y_pred_equal": float(equal_weight_pred),
                "y_pred_best": float(expert_preds[best_idx]),
                "loss_raw": float(ensemble_raw_loss),
                "loss_equal": float(equal_weight_loss),
                "loss_best": float(raw_losses[best_idx]),
            }
        )

        # 7) history を更新
        new_point = pd.Series([y_true], index=[tstamp])
        running_history = pd.concat([running_history, new_point])

    total_steps = len(phase_data)
    phase_stats = {
        "expert_cum_losses": expert_cum_losses,
        "expert_win_counts": expert_win_counts,
        "expert_avg_weights": expert_cum_weights / max(total_steps, 1),
        "total_steps": total_steps,
        "weight_snapshots": weight_snapshots,
    }

    return records, phase_stats


# ------------------------------------------------------------------
# メイン実行関数
# ------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """実験を実行する."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # シード固定
    _set_seed(args.seed)

    # eta パース
    etas = _parse_etas(args.etas)

    # run_id 生成
    run_id = _generate_run_id(args)
    report_dir = _PROJECT_ROOT / "reports" / run_id
    report_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Run ID: %s", run_id)
    logger.info("Report dir: %s", report_dir)

    # 実験設定保存
    config = {
        "run_id": run_id,
        "data_path": args.data_path,
        "agg": args.agg,
        "impute": args.impute,
        "clip_quantile": args.clip_quantile,
        "experts": args.experts,
        "eta_mode": args.eta_mode,
        "etas": etas,
        "scale_loss": args.scale_loss,
        "n_series": args.n_series,
        "series_sample": args.series_sample,
        "seed": args.seed,
    }
    config_path = report_dir / "experiment_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    logger.info("実験設定を保存: %s", config_path)

    # データ読み込み・前処理
    df = _load_and_preprocess(args)

    # 時系列分割
    splits = split_temporal(df)
    logger.info(
        "分割: train=%d, valid=%d, test=%d",
        len(splits.train),
        len(splits.valid),
        len(splits.test),
    )

    # 系列選択
    selected_cols = _select_series(df.columns, args.n_series, args.series_sample)
    logger.info("処理対象系列数: %d", len(selected_cols))

    # 全系列の結果を格納
    all_records: list[dict] = []
    all_expert_names: list[str] = []
    # Expert 別統計を集約（全系列の test フェーズを合算）
    agg_expert_cum_losses: np.ndarray | None = None
    agg_expert_win_counts: np.ndarray | None = None
    agg_expert_avg_weights_sum: np.ndarray | None = None
    agg_total_steps: int = 0
    # 代表系列の weight_snapshots を保持（最初の系列の test フェーズ）
    representative_weight_snapshots: list[dict] = []
    is_meta_eta: bool = args.eta_mode == "meta_grid"

    # 系列ごとのループ
    for col_idx, col in enumerate(tqdm(selected_cols, desc="Series")):
        train_series = splits.train[col]
        valid_series = splits.valid[col]
        test_series = splits.test[col]

        # a) Expert 群を生成
        experts = create_experts(preset=args.experts)
        n_experts = len(experts)

        # b) Train 期間で fit
        for exp in experts:
            try:
                exp.fit(train_series)
            except Exception as e:
                logger.warning("Expert %s の fit に失敗: %s", exp.name, e)

        # c) Train 期間の naive MAE (scale-loss 用)
        train_mae = _compute_train_mae_naive(train_series)

        # Expert 名リスト (初回のみ)
        if not all_expert_names:
            all_expert_names = [exp.name for exp in experts]

        # d) Ensemble 生成
        ensemble = _create_ensemble(n_experts, args.eta_mode, etas)

        # e) Valid 期間で逐次予測・重み更新
        valid_records, _valid_stats = _run_online_phase(
            experts=experts,
            ensemble=ensemble,
            history=train_series,
            phase_data=valid_series,
            scale_loss_mode=args.scale_loss,
            train_mae=train_mae,
            desc=f"Valid({col})",
        )
        for r in valid_records:
            r["series"] = col
            r["phase"] = "valid"
        all_records.extend(valid_records)

        # f) Test 期間で逐次予測・重み更新
        # history は train + valid を合わせたもの
        train_valid_series = pd.concat([train_series, valid_series])
        test_records, test_stats = _run_online_phase(
            experts=experts,
            ensemble=ensemble,
            history=train_valid_series,
            phase_data=test_series,
            scale_loss_mode=args.scale_loss,
            train_mae=train_mae,
            desc=f"Test({col})",
        )
        for r in test_records:
            r["series"] = col
            r["phase"] = "test"
        all_records.extend(test_records)

        # Expert 別統計を全系列で集約
        if agg_expert_cum_losses is None:
            agg_expert_cum_losses = test_stats["expert_cum_losses"].copy()
            agg_expert_win_counts = test_stats["expert_win_counts"].copy()
            agg_expert_avg_weights_sum = test_stats["expert_avg_weights"].copy()
        else:
            agg_expert_cum_losses += test_stats["expert_cum_losses"]
            agg_expert_win_counts += test_stats["expert_win_counts"]
            agg_expert_avg_weights_sum += test_stats["expert_avg_weights"]
        agg_total_steps += test_stats["total_steps"]

        # 代表系列 (最初) の weight snapshots を保持
        if col_idx == 0:
            representative_weight_snapshots = test_stats["weight_snapshots"]

    # Expert 統計をまとめる
    expert_stats: dict | None = None
    if agg_expert_cum_losses is not None:
        n_series_processed = len(selected_cols)
        expert_stats = {
            "expert_cum_losses": agg_expert_cum_losses,
            "expert_win_counts": agg_expert_win_counts,
            "expert_avg_weights": agg_expert_avg_weights_sum / max(n_series_processed, 1),
            "total_steps": agg_total_steps,
            "weight_snapshots": representative_weight_snapshots,
            "is_meta_eta": is_meta_eta,
        }

    # 結果保存
    if all_records:
        result_df = pd.DataFrame(all_records)
        pred_path = report_dir / "predictions.parquet"
        result_df.to_parquet(pred_path, engine="pyarrow", index=False)
        logger.info("予測結果を保存: %s (%d 行)", pred_path, len(result_df))

        # サマリ表示
        for phase in ["valid", "test"]:
            phase_df = result_df[result_df["phase"] == phase]
            if len(phase_df) > 0:
                mean_loss = phase_df["loss_raw"].mean()
                logger.info("%s 平均 MAE: %.4f", phase.upper(), mean_loss)

        # レポート生成
        try:
            from src.report import generate_report

            generate_report(report_dir, result_df, config, all_expert_names, expert_stats=expert_stats)
            logger.info("レポート生成完了: %s", report_dir)
        except Exception as e:
            logger.warning("レポート生成に失敗: %s", e)
    else:
        logger.warning("結果が空です。データを確認してください。")

    logger.info("実験完了: %s", run_id)


if __name__ == "__main__":
    main()
