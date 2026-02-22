# Task 009: Meta-η（二段Hedge / 学習率自動調整）実装

## 概要
学習率ηの自動調整を実現するmeta-expert化（Grid-ηの二段Hedge）を実装する。（spec §8.3）

## 成果物
- `src/ensemble/meta_eta.py`:
  - `MetaEtaHedge` クラス
    - η候補集合: `etas = [2^-k for k in 0..10]`（スケール調整済み）
    - 各ηごとにHedgeインスタンスを保持
    - 上位メタレイヤーでη間の重みを管理（二段Hedge）
    - `predict(expert_predictions: np.ndarray) -> float`: メタ重み付き平均
    - `update(y_true: float, expert_predictions: np.ndarray) -> None`
    - `get_eta_weights() -> np.ndarray`: η間の重み取得
    - `get_effective_eta() -> float`: 実効η（重み付き平均）

## 受け入れ条件
- 固定η版（task-008）と同じインターフェースで差し替え可能
- η重みが時間とともに適切に変化する
- 損失スケーリングと組み合わせて動作する
- テストが通る

## 依存タスク
- task-008

## 見積もり
中規模
