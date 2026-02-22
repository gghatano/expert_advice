# Task 005: 回帰系Expert実装

## 概要
軽量な回帰モデルベースのExpertを実装する。（spec §7.2-C）

## 成果物
- `src/experts/regression.py`:
  - `RidgeLag`: Ridgeリグレッション（ラグ特徴: 1,2,3,24,168 + 時刻特徴: hour, dow）
    - 正則化α違いで複数（α=0.1, 1, 10）
    - 学習はtrain期間で1回
  - `HuberRegressorLag`: 外れ値耐性のある回帰（同じ特徴量）
  - `KNNLag`: K近傍（k=5、直近Nサンプルから近傍探索に限定して軽量化）

## 受け入れ条件
- BaseExpertインターフェース準拠
- fit時にscikit-learnモデルを学習
- predict_nextは特徴量を構成してモデルで推論
- 学習データ不足時のfallback動作あり
- テストが通る

## 依存タスク
- task-003（基底クラス）

## 見積もり
中規模
