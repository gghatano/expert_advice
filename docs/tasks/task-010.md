# Task 010: 実験実行ループ・CLI実装

## 概要
逐次予測の実験ループとCLIエントリポイントを実装する。（spec §9, §11）

## 成果物
- `src/run_experiment.py`:
  - CLIエントリポイント（`python -m src.run_experiment`）
  - 引数:
    - `--data-path`: 生データパス
    - `--agg`: hourly_sum | hourly_mean
    - `--impute`: ffill0
    - `--clip-quantile`: 外れ値クリップ閾値（未指定ならOFF）
    - `--experts`: preset名（light30, light80）
    - `--eta-mode`: meta_grid | fixed
    - `--etas`: η候補リスト（meta_grid時）
    - `--scale-loss`: by_train_mae | relative
    - `--n-series` / `--series-sample`: 処理系列数
    - `--seed`: 乱数シード
  - 実験ループ（系列ごと）:
    1. Train期間でexpert学習
    2. Valid期間で逐次予測＋重み更新
    3. Test期間で逐次予測＋重み更新
  - 出力ログ: ensemble予測値、真値、loss、上位k expert重み、η重み
  - run_id生成: 日時＋引数ハッシュ
  - 進捗表示（tqdm等）

## 受け入れ条件
- `--series-sample 5` で正常動作（数分以内）
- 結果が `reports/<run_id>/` に保存される
- 再実行で同じ結果（再現性、seed固定時）
- エラーハンドリング（データ不在時等）

## 依存タスク
- task-002, task-007, task-009

## 見積もり
大規模
