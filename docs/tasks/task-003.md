# Task 003: Expert基底クラス・Naive系Expert実装

## 概要
Expert共通インターフェースと、Naive系（最も単純な予測器群）を実装する。（spec §7.1, §7.2-A）

## 成果物
- `src/experts/base.py`: 基底クラス `BaseExpert`
  - `fit(history: pd.Series, **kwargs) -> None`
  - `predict_next(history: pd.Series, tstamp: pd.Timestamp, exog: dict | None) -> float`
  - `name: str` プロパティ
- `src/experts/naive.py`: Naive系Expert
  1. `LastValue`: 直近値をそのまま返す
  2. `SeasonalNaive`: N時間前の値を返す（N=24, 48, 168等、パラメータ化）
  3. `Drift`: 直近W時間の線形トレンドで外挿（W=24, 168等）

## 受け入れ条件
- 各Expertの `predict_next` がfloatを返す
- NaNを返さない（欠損対策のfallbackあり）
- パラメータ違いで複数インスタンス生成可能
- `tests/test_experts.py` のNaive系テストが通る

## 依存タスク
- task-001

## 見積もり
小〜中規模
