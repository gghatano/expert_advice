以下は、Claude Code にそのまま渡して実装できる粒度の**仕様書（実装仕様＋評価仕様）**です。Electricity Load Diagrams を **軽量 expert 多数 + expert advice（学習率自動調整つき）**で逐次予測する構成にしています。

---

# 仕様書：Electricity Load Diagrams に対する Expert Advice アンサンブル（軽量expert多数）

## 0. 目的

* Electricity Load Diagrams 2011–2014（UCI）を用い、**多数の軽量予測器（experts）**を構築する。
* 逐次予測の枠組み（Prediction with Expert Advice）で、各expertの過去損失に基づいて重み更新し、**重み付き平均**で最終予測を出す。
* 学習率 η は **自動適応**（Koolen et al. 2014の「learning the learning rate」系の考え方）を採用し、手動チューニングを最小化する。

## 1. スコープ

* 対象：Electricity Load Diagrams 2011–2014（UCI）
* 粒度：**15分→1時間に集約**（デフォルト）
* 予測：**1ステップ先（次の1時間）**を逐次で予測（まずはここに固定）

  * 24時間先など多ステップは拡張ポイントとして別issue化

## 2. 成果物

1. データ取得・前処理パイプライン
2. expert群（軽量）実装
3. expert advice アンサンブル（重み更新＋η自動調整）
4. 実験実行CLI
5. レポート出力（集計指標・重み推移・上位expert寄与など）
6. 最小限のテスト

## 3. 実行環境

* Python 3.11+
* 依存（推奨）

  * numpy, pandas, scipy
  * scikit-learn
  * statsmodels（指数平滑やSARIMAXを使う場合のみ。重いなら任意）
  * matplotlib（レポート図）
  * pyarrow（任意：データキャッシュ高速化）

※「軽量」を優先し、深層学習フレームワークは不使用。

## 4. データ仕様

### 4.1 取得

* UCI ML Repository “ElectricityLoadDiagrams20112014”
* 取得方法は以下のいずれか：

  * (A) 手動でzipをダウンロードして `data/raw/` に配置
  * (B) CLI でURLからダウンロード（可能なら）

### 4.2 入力形式

* 元データ：時刻インデックス + 370系列（顧客別）
* 欠損があり得る（0値/NaN/飛び）

### 4.3 変換

* 15分粒度 → 1時間粒度へ集約

  * `sum` をデフォルト（消費量の合計として扱う）
  * オプションで `mean` を選択可能

### 4.4 欠損処理（デフォルト）

* 1時間集約後、各系列で

  * 連続欠損が短い場合：前方補完（ffill）→残りは0
  * 具体：`ffill(limit=24)`、それでも残る欠損は 0
* 追加オプション

  * “drop series”: 欠損率が一定以上の系列は除外（例：>5%）

### 4.5 外れ値処理（デフォルトOFF）

* OFFがデフォルト
* ONの場合：系列ごとに上位pパーセンタイルでクリップ（例：p=99.9）

## 5. データ分割（時間順固定）

* Train：2011-01-01 〜 2013-12-31
* Valid：2014-01-01 〜 2014-06-30
* Test ：2014-07-01 〜 2014-12-31
* 厳密な日付はデータに合わせて丸めて良いが、必ず時間順でリークなし。

## 6. 予測タスク定義

### 6.1 1ステップ逐次予測（デフォルト）

各系列 s について、各時刻 t で

* 入力：過去の観測値列 (y_{s,1:t})
* 出力：次時刻の予測値 (\hat{y}_{s,t+1})

### 6.2 損失関数

* デフォルト：MAE（絶対誤差）

  * (\ell_{s,t} = |y_{s,t} - \hat{y}_{s,t}|)
* 追加（オプション）：sMAPE, RMSE

### 6.3 評価集計

* 系列×時刻の平均（macro）

  * `mean_over_all_points`
* 系列平均→全体平均（2段階macro）も出す

  * seriesごとの平均MAE → series間平均

## 7. Expert（軽量予測器）設計

**目的：多数で多様性を作る（誤差相関を下げる）**

### 7.1 必須インターフェース

* `fit(history: pd.Series, **kwargs) -> None`（必要なら）
* `predict_next(history: pd.Series, tstamp: pd.Timestamp, exog: dict | None) -> float`
* 逐次運用を想定し、**predictは軽量**にする（毎時刻呼ばれる）

### 7.2 Expert一覧（MVP：最低12本）

A. Naive系

1. LastValue（直近値）
2. SeasonalNaive_24（24h前）
3. SeasonalNaive_168（168h前）
4. Drift（直近W時間の傾向で外挿：W=24/168の2種）

B. 移動平均/平滑
5. SMA_24（24h平均）
6. SMA_168（168h平均）
7. EMA_alpha（α=0.1/0.3/0.5の3種）
8. Median_24（24h中央値：外れ値耐性）

C. 回帰（軽量）
9. RidgeLag（ラグ特徴：1,2,3,24,168 + 時刻特徴 hour,dow。学習はtrain期間で1回）
10. HuberRegressorLag（外れ値耐性）
11. KNNLag（k=5、距離=ラグベクトル、軽量に注意：系列ごとに学習は重いので、実装は「直近Nサンプルから近傍探索」に限定）

D. 分解・季節性簡易
12. STLSeasonalMean（週次/日次の平均プロファイルを学習し予測に混ぜる）

* 例：曜日×時刻(0-23)の平均（trainから算出）をベース予測にする

### 7.3 Expertのパラメータ生成（“たくさん”を作る）

* 同種expertはパラメータ違いを複数作って expert数を増やす

  * EMA：αを複数
  * Drift：windowを複数
  * Ridge：正則化αを複数（0.1,1,10）
  * SeasonalNaive：24/48/168など
* 目標 expert 数：**30〜80**（実行時間とのトレードオフ）

## 8. Expert Advice（アンサンブル）設計

### 8.1 基本

* 各時刻 t において、expert i の予測 (\hat{y}_{t,i}) を得る
* アンサンブル予測：

  * (\hat{y}*{t} = \sum_i p*{t,i} \hat{y}_{t,i})
  * (p_{t,i} = w_{t,i} / \sum_j w_{t,j})

### 8.2 重み更新（指数重み）

* 損失 (\ell_{t,i}) に基づき

  * (w_{t+1,i} = w_{t,i}\exp(-\eta_t \ell_{t,i}))
* 数値安定化のため log-weight 管理を採用

  * `log_w[i] += -eta_t * loss[i]`
  * 正規化はsoftmax相当

### 8.3 η（学習率）の自動調整（要件）

* 目的：チューニング不要に近づける
* 仕様としては以下のいずれかを採用（実装容易な順）

  1. **Grid-η の meta-expert化**（推奨：最短でKoolen系の効果を再現しやすい）

     * η候補集合：例 `etas = [2^-k for k in 0..10]` をスケール調整
     * 各ηごとにHedgeを走らせ、その上位にさらにHedgeで混ぜる（二段Hedge）
     * これで “learning the learning rate” を実務的に実現
  2. **AdaHedge/Squint系の近似実装**（時間があれば）

     * 文献ベースの更新則に従いηを逐次更新

MVPは(1)で良い。結果が出たら(2)へ拡張。

### 8.4 クリッピング/スケーリング（重要）

* 損失スケールによりηが効きすぎる問題を避けるため、損失を正規化する

  * 例：`loss = abs_err / (abs(y_true) + 1e-6)`（相対誤差）を内部損失にするオプション
  * あるいは、系列ごとに train のMAEで割ってスケーリング（推奨）
* デフォルト：**系列ごとの標準化スケール**（train MAE or train mean）で割る

## 9. 実験ループ（逐次処理）

### 9.1 単位

* 系列ごとに独立に実験するのが基本
* ただし実装効率のため、同一時刻で370系列をベクトル処理してもよい

### 9.2 手順（series s）

1. Train期間で expertの学習（必要なもののみ）
2. Valid期間で逐次予測＋重み更新（online）
3. Test期間で逐次予測＋重み更新（online）

   * “評価のために重み更新を止める”モードもオプションで用意

### 9.3 出力ログ

* 各時刻で保存（全保存は重いので間引き可能）

  * ensemble予測値、真値、loss
  * 上位k expert の重み（k=5）
  * η（meta-ηの場合は上位η重み）

## 10. レポート仕様

* `reports/` に以下を保存

  * `metrics.json`：MAE, sMAPE（valid/test別）
  * `experts_rank.csv`：expert別の平均損失、平均重み、勝率（最小損失になった割合）
  * `weights_plot.png`：上位expert重み推移（代表系列を数本）
  * `eta_plot.png`：ηの重み推移（meta-ηの場合）
  * `summary.md`：結果サマリ

代表系列の選び方：

* train平均負荷の高い/低い、欠損多い/少ない、外れ値多い/少ない、などから各1本

## 11. CLI設計

* `python -m src.run_experiment ...`
* 引数例

  * `--data-path data/raw/...`
  * `--agg hourly_sum|hourly_mean`
  * `--impute ffill0`
  * `--clip-quantile 0.999`（未指定ならOFF）
  * `--experts preset=light80`（preset名でexpert群生成）
  * `--eta-mode meta_grid|fixed`
  * `--etas 1.0,0.5,0.25,...`（meta_grid時）
  * `--scale-loss by_train_mae|relative`
  * `--n-series 370` / `--series-sample 50`（デバッグ用）
  * `--seed 42`

## 12. ディレクトリ構成（提案）

```
project/
  data/
    raw/
    processed/
  src/
    data/
      load_uci.py
      preprocess.py
      split.py
    experts/
      base.py
      naive.py
      moving_avg.py
      smoothing.py
      regression.py
      seasonal_profile.py
      factory.py
    ensemble/
      hedge.py
      meta_eta.py
      loss.py
      scaling.py
    run_experiment.py
    report.py
  tests/
    test_experts.py
    test_hedge.py
    test_preprocess.py
  reports/
  pyproject.toml
  README.md
```

## 13. テスト要件（最小）

* Expert interfaceが壊れていない

  * predictがfloatを返す、NaNを返さない（許容範囲のクリップも検討）
* Hedge更新が単調性を満たす（損失が小さいexpertの重みが相対的に増える）
* 前処理（集約・欠損補完）が再現可能

## 14. 非機能要件

* 370系列 × 4年 hourly（約35k時点/系列）なので計算量注意
* 目標：`--series-sample 50` で数分、全系列で数十分程度（expert数に依存）
* ベクトル化（numpy）とキャッシュ（processed保存）を活用

## 15. 拡張ポイント（別issue）

* 24-step ahead（seq-to-vector）への拡張
* “specialist experts”（一部時刻だけ予測するexpert）
* 外生変数（祝日、気温など）追加（今回は無し）

---

## Claude Code への指示文（貼り付け用）

* 上記仕様に基づき、Pythonプロジェクト一式を生成せよ。
* まず `--series-sample 5` で動く最小実装（MVP）を作り、次に `preset=light30` → `light80` と段階的に増やせ。
* 速度最適化より正しさ・再現性を優先し、処理済みデータのキャッシュを入れよ。
* レポートは `reports/<run_id>/` に保存し、run_idは日時＋主要引数のハッシュで生成せよ。

---

必要なら、**expertの具体的なパラメータ一覧（light30/light80の内訳）**と、**meta-η（二段Hedge）の擬似コード**まで追記して、Claude Code が迷わず実装できるレベルに落とします。

