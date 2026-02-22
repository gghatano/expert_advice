# Task 004: 移動平均・平滑系Expert実装

## 概要
移動平均・指数平滑・中央値系のExpertを実装する。（spec §7.2-B）

## 成果物
- `src/experts/moving_avg.py`: 移動平均系
  - `SMA`: 単純移動平均（window=24, 168等）
  - `Median`: 移動中央値（window=24等、外れ値耐性）
- `src/experts/smoothing.py`: 指数平滑
  - `EMA`: 指数移動平均（α=0.1, 0.3, 0.5等）

## 受け入れ条件
- BaseExpertインターフェース準拠
- predict_nextがfloatを返し、NaNを返さない
- パラメータ違いで複数インスタンス生成可能
- テストが通る

## 依存タスク
- task-003（基底クラス）

## 見積もり
小規模
