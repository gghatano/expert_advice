# Task 007: Expertファクトリ（preset生成）

## 概要
パラメータ違いのExpertインスタンス群を一括生成するファクトリを実装する。（spec §7.3）

## 成果物
- `src/experts/factory.py`:
  - `create_experts(preset: str) -> list[BaseExpert]`
  - preset定義:
    - `light30`: 約30本（基本パラメータのみ）
    - `light80`: 約80本（パラメータのバリエーション拡張）
  - 各preset内訳の定義（docstring or config）
    - Naive系: LastValue(1), SeasonalNaive(24,48,168=3), Drift(24,168=2) → 6本
    - 移動平均系: SMA(24,48,168=3), Median(24=1), EMA(0.1,0.3,0.5=3) → 7本
    - 回帰系: Ridge(α=0.1,1,10=3), Huber(1), KNN(1) → 5本
    - 季節性: STLSeasonalMean(1) → 1本
    - light30合計 ≈ 19+α（バリエーション追加で30前後に）
    - light80: さらにwindow/αのバリエーション追加

## 受け入れ条件
- `create_experts("light30")` が30前後のExpertリストを返す
- `create_experts("light80")` が80前後のExpertリストを返す
- 全Expertがインターフェース準拠
- Expert名が一意（パラメータを含む名前）

## 依存タスク
- task-003, task-004, task-005, task-006

## 見積もり
小規模
