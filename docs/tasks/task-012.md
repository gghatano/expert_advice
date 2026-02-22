# Task 012: テスト整備・動作確認

## 概要
最小限のテストスイートを整備し、全体の動作確認を行う。（spec §13）

## 成果物
- `tests/test_experts.py`:
  - 全Expert種別のインターフェーステスト
  - predict_nextがfloatを返す
  - NaNを返さない
  - パラメータ違いの生成テスト
- `tests/test_hedge.py`:
  - 重み更新の単調性テスト
  - log-weight管理の数値安定性テスト
  - MetaEtaHedgeの基本動作テスト
- `tests/test_preprocess.py`:
  - 集約処理の正確性テスト
  - 欠損補完の再現性テスト
  - 分割のリーク無しテスト
- E2Eテスト（小規模データ）:
  - `--series-sample 2` で全パイプラインが通る

## 受け入れ条件
- `pytest tests/` が全パス
- カバレッジ: 主要パスが網羅されている

## 依存タスク
- task-010, task-011

## 見積もり
中規模
