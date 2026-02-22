# Task 001: プロジェクト初期化・基盤セットアップ

## 概要
Pythonプロジェクトの骨格を作成し、依存関係・ディレクトリ構成・基本設定を整備する。

## 成果物
- `pyproject.toml`（依存: numpy, pandas, scipy, scikit-learn, statsmodels, matplotlib, pyarrow）
- ディレクトリ構成（spec §12準拠）
  - `src/data/`, `src/experts/`, `src/ensemble/`
  - `tests/`, `data/raw/`, `data/processed/`, `reports/`
- `src/__init__.py` 等の各パッケージ初期化
- `.gitignore`（data/raw/, reports/, __pycache__, .venv 等）
- `README.md`（最小限）

## 受け入れ条件
- `pip install -e .` または `uv sync` でインストール可能
- `python -c "import src"` がエラーなく通る

## 依存タスク
なし（最初に実施）

## 見積もり
小規模
