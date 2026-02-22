# Task 002: データ取得・前処理パイプライン

## 概要
UCI Electricity Load Diagrams 2011–2014 データの取得・読み込み・前処理を実装する。（spec §4）

## 成果物
- `src/data/load_uci.py`: データ読み込み（CSV→DataFrame）
  - ダウンロード機能（CLI/手動配置の両対応）
  - 370系列 × 時刻インデックスのDataFrameを返す
- `src/data/preprocess.py`: 前処理
  - 15分→1時間集約（sum/mean選択可能）
  - 欠損処理: ffill(limit=24) → 残りは0
  - 外れ値クリップ（オプション、デフォルトOFF）: 系列ごとにpパーセンタイルでクリップ
  - 欠損率が高い系列の除外（オプション: >5%）
- `src/data/split.py`: 時系列分割
  - Train: 2011-01-01〜2013-12-31
  - Valid: 2014-01-01〜2014-06-30
  - Test: 2014-07-01〜2014-12-31
- 処理済みデータの Parquet キャッシュ（`data/processed/`）

## 受け入れ条件
- `data/raw/` にデータ配置後、前処理パイプラインが正常動作
- 集約後のデータ形状が想定通り（約35,064時点 × 370系列）
- キャッシュの読み書きが正常
- `tests/test_preprocess.py` が通る

## 依存タスク
- task-001

## 見積もり
中規模
