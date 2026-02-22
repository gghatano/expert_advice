# Task 008: Hedge（指数重み）アンサンブル実装

## 概要
Expert Adviceの中核となる指数重み更新（Hedge）アルゴリズムを実装する。（spec §8.1, §8.2）

## 成果物
- `src/ensemble/hedge.py`:
  - `Hedge` クラス
    - 初期化: expert数、学習率η
    - log-weight管理（数値安定化）
    - `predict(expert_predictions: np.ndarray) -> float`: 重み付き平均予測
    - `update(losses: np.ndarray) -> None`: 指数重み更新
    - `get_weights() -> np.ndarray`: 現在の正規化重み取得
    - `get_top_k(k: int) -> list[tuple[int, float]]`: 上位k expertのindex,重み
- `src/ensemble/loss.py`:
  - `mae_loss(y_true, y_pred) -> float`
  - `smape_loss(y_true, y_pred) -> float`（オプション）
  - `rmse_loss(y_true, y_pred) -> float`（オプション）
- `src/ensemble/scaling.py`:
  - 損失スケーリング
    - `by_train_mae`: 系列ごとのtrain MAEで除算
    - `relative`: `abs_err / (abs(y_true) + 1e-6)`

## 受け入れ条件
- 損失が小さいexpertの重みが相対的に増加する（単調性テスト）
- log-weight管理でオーバーフロー/アンダーフローが発生しない
- `tests/test_hedge.py` が通る

## 依存タスク
- task-001

## 見積もり
中規模
