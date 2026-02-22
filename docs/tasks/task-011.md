# Task 011: レポート出力

## 概要
実験結果のレポートを生成する機能を実装する。（spec §10）

## 成果物
- `src/report.py`:
  - `reports/<run_id>/` に以下を出力:
    - `metrics.json`: MAE, sMAPE（valid/test別、全体・系列別）
    - `experts_rank.csv`: expert別の平均損失、平均重み、勝率
    - `weights_plot.png`: 上位expert重み推移（代表系列を数本）
    - `eta_plot.png`: ηの重み推移（meta-η使用時）
    - `summary.md`: 結果サマリ（テキスト）
  - 代表系列の選定ロジック:
    - train平均負荷の高い/低い系列
    - 欠損多い/少ない系列
    - 各カテゴリから1本ずつ

## 受け入れ条件
- 全出力ファイルが正常に生成される
- プロットが読みやすい（凡例、軸ラベル付き）
- summary.mdに主要指標が含まれる

## 依存タスク
- task-010

## 見積もり
中規模
