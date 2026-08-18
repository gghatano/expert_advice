# Expert Advise

**Prediction with Expert Advice** の枠組みを用いた電力負荷予測アンサンブルシステム。30〜80の軽量Expertの予測をHedgeアルゴリズムで動的に重み付けし、オンライン学習で継続的に最適化します。

📖 **[解説・実験レポート・追試結果（GitHub Pages）](https://gghatano.github.io/expert_advice/)**

> **わかったこと**
> Hedge 系のアルゴリズム（Hedge / Meta-η / AdaHedge）は「事後に選べる最良の単一 Expert」に*追いつく*ためのもので、原理的にそれを*超えません*。UCI 実データでも最良固定Expert比 +0.2%（＝集約した意味がほぼゼロ）でした。
> 一方 **Fixed-Share** に変えるだけで、同じデータ・同じExpert群で最良固定Expert比 **−7.6%（1時間先）/ −12.4%（翌日）**、系列の 96〜100% で勝ちます。詳細は [実験レポート](https://gghatano.github.io/expert_advice/experiment.html) と [追試結果](https://gghatano.github.io/expert_advice/replication.html) を参照。

## アルゴリズム概要

### Prediction with Expert Advice

本システムは、オンライン学習理論における **Prediction with Expert Advice** の枠組みに基づいています。各時刻で複数のExpert（予測器）が予測を出し、過去の実績に基づく重みで統合することで、事後的に最良のExpertに匹敵する性能を達成します。

![アルゴリズム概念図](docs/figures/algorithm_concept.png)

### Hedge アルゴリズム

各Expertの重みを指数的に更新する multiplicative weights アルゴリズムです。

**重み更新則:**

$$w_i^{(t+1)} = w_i^{(t)} \cdot e^{-\eta \cdot \ell_i^{(t)}}$$

- $w_i^{(t)}$: 時刻 $t$ における Expert $i$ の重み
- $\ell_i^{(t)}$: 時刻 $t$ における Expert $i$ の損失 (MAE)
- $\eta$: 学習率（大きいほど損失の大きい Expert を素早く下方修正）

**アンサンブル予測:**

$$\hat{y}^{(t)} = \sum_{i=1}^{N} \bar{w}_i^{(t)} \cdot f_i(x^{(t)})$$

ここで $\bar{w}_i$ は正規化された重み（softmax）です。数値安定性のため対数重み空間で管理しています。

### Meta-η（二段Hedge）

学習率 $\eta$ の選択を自動化する仕組みです。複数の $\eta$ 候補（デフォルト: $2^{0}, 2^{-1}, \ldots, 2^{-10}$）それぞれに独立したHedgeインスタンスを持ち、メタレベルのHedgeが最適な $\eta$ を追跡します。

```
η候補: [1.0, 0.5, 0.25, ..., 0.001]
       ↓     ↓     ↓           ↓
    Hedge₁ Hedge₂ Hedge₃ ... Hedge₁₁   ← 各ηでExpert重みを独立管理
       ↓     ↓     ↓           ↓
    pred₁  pred₂  pred₃  ... pred₁₁
       ↓     ↓     ↓           ↓
    ========== Meta Hedge ==========    ← メタレベルで最良のηを追跡
                  ↓
            最終予測 ŷ
```

### 追随・自動調整アルゴリズム

Hedge の弱点（重みが累積損失だけで決まるため、一度見捨てたExpertが復帰できない）に対応する集約アルゴリズムを実装しています。

| アルゴリズム | 実装 | 比較対象（何に追いつくか） | つまみ |
|---|---|---|---|
| Hedge | `ensemble/hedge.py` | 最良の固定Expert | η |
| Meta-η Hedge | `ensemble/meta_eta.py` | 最良の固定Expert | η候補集合 |
| **AdaHedge** | `ensemble/adahedge.py` | 最良の固定Expert | **なし** |
| **Fixed-Share** | `ensemble/fixed_share.py` | **最良の切り替え系列** | α, η |
| **ML-Poly** | `ensemble/ml_poly.py` | アンサンブル自身への後悔 | **なし** |

**Fixed-Share** は指数更新のあとに重みの一部を全員へ配り直します:

$$w \leftarrow (1-\alpha)\, w + \frac{\alpha}{N}$$

これで全Expertの重みが $\alpha/N$ 以上に保たれ、レジーム変化に数十ステップで追随できます（$\alpha = 0$ で Hedge に一致）。非定常データでは最良の固定Expertを実際に下回れる、唯一のアルゴリズムでした。

### Expert群の構成

4カテゴリ・約30種類（`light30`プリセット）のExpertを使用します:

| カテゴリ | Expert | パラメータ例 | 数 |
|---------|--------|------------|---:|
| **Naive** | LastValue, SeasonalNaive, Drift | season=24,48,168h; window=24,168h | 6 |
| **Moving Average** | SMA, Median, EMA | window=12-336h; α=0.05-0.7 | 7+ |
| **Regression** | RidgeLag, HuberLag, KNNLag | α=0.1-10; k=3-10 | 5 |
| **Seasonal** | STLSeasonalMean | 7×24 曜日×時間プロファイル | 1 |

## 実験

実データを使った 5 つの実験を `scripts/experiments/` に用意しています。図と数値は `docs/` に出力され、[GitHub Pages のサイト](https://gghatano.github.io/expert_advice/)から読めます。

| 実験 | データ | 結果（最良固定Expert比） |
|---|---|---|
| `exp0_uci_baseline.py` | UCI Electricity（実データ） | 既定設定は +0.2%、Fixed-Share は −7.6% / −12.4% |
| `exp1_regime.py` | 合成レジーム切替 | Fixed-Share **−29.8%**（切り替えオラクルにほぼ到達） |
| `exp2_m4.py` | M4 Hourly 414系列 | Fixed-Share **−22.9%**、系列の 62% で勝利 |
| `exp3_etth1.py` | ETTh1 7変数・翌日予測 | Fixed-Share が 7変数すべてで勝利 |
| `exp4_covid.py` | COVID-19期の欧州電力需要 | Fixed-Share が 4か国すべてで −39% 以上 |

### 合成データでの動作確認

以下は合成データ（正弦波+ノイズ）での基本動作の確認です。`scripts/generate_readme_figures.py` で再現できます。

### 手法別MAE比較

Hedge（Expert Advice）、等重み平均、最良単一Expertの3手法を比較:

![MAE比較](docs/figures/mae_comparison.png)

### 時系列予測比較

テスト期間における実測値と各手法の予測値:

![時系列比較](docs/figures/timeseries_comparison.png)

### 累積損失推移

時間経過に伴う累積MAEの推移。Hedgeは重みの適応的な更新により、等重み平均に対して累積損失を抑えます:

![累積損失](docs/figures/cumulative_loss.png)

### Expert重み推移

Hedgeアルゴリズムが各Expertにどのように重みを配分しているかの変遷:

![重み推移](docs/figures/weight_evolution.png)

## セットアップ

```bash
pip install -e ".[dev]"
```

または [uv](https://docs.astral.sh/uv/) を使う場合:

```bash
uv sync --all-extras
```

## 使い方

### データ取得

`data/raw/` は .gitignore されているため、clone 直後は空です。次のコマンドで全データセット（M4 / ETTh1 / OPSD / UCI）を取得できます。

```bash
uv run python scripts/download_data.py

# 一部だけ取得する場合
uv run python scripts/download_data.py --only m4 etth1
```

### 実験の実行

```bash
# 基本実行（5系列サンプル、light30 Expert群）
python -m src.run_experiment --data-path data/raw/ --experts light30 --series-sample 5

# Meta-η + 80 Expert群で全系列実行
python -m src.run_experiment --data-path data/raw/ --experts light80 --eta-mode meta_grid

# Fixed-Share（非定常データではこちらを推奨）
python -m src.run_experiment --data-path data/raw/ --aggregator fixed_share --alpha 0.01

# パラメータ不要の自動調整アルゴリズム
python -m src.run_experiment --data-path data/raw/ --aggregator adahedge

# 固定η + 損失スケーリング変更
python -m src.run_experiment --data-path data/raw/ --eta-mode fixed --etas 0.1 --scale-loss relative
```

### 比較実験とドキュメントサイトの再生成

```bash
uv run python scripts/experiments/exp0_uci_baseline.py   # 既定設定の再現
uv run python scripts/experiments/exp1_regime.py         # レジーム切替（合成データ）
uv run python scripts/experiments/exp2_m4.py             # M4 Hourly 414系列
uv run python scripts/experiments/exp3_etth1.py          # ETTh1
uv run python scripts/experiments/exp4_covid.py          # COVID-19期の電力需要

# docs/results/*.json から docs/*.html の表を再生成
uv run python scripts/build_site.py
```

### README図の再生成

```bash
uv run python scripts/generate_readme_figures.py
```

### テスト

```bash
pytest tests/ -v
```

## プロジェクト構成

```
expert-advise/
├── src/
│   ├── run_experiment.py        # 実験ループ・CLI
│   ├── report.py                # レポート・プロット生成
│   ├── data/
│   │   ├── load_uci.py          # UCIデータ読み込み
│   │   ├── preprocess.py        # 前処理（リサンプル・欠損補完・外れ値処理）
│   │   └── split.py             # Train/Valid/Test 時系列分割
│   ├── ensemble/
│   │   ├── hedge.py             # Hedge アルゴリズム
│   │   ├── meta_eta.py          # Meta-η Hedge（二段Hedge）
│   │   ├── fixed_share.py       # Fixed-Share（最良Expert追随）
│   │   ├── adahedge.py          # AdaHedge（学習率の自動調整）
│   │   ├── ml_poly.py           # ML-Poly（regretベース集約）
│   │   ├── runner.py            # 予測行列上でのオンライン実行・事後ベンチマーク
│   │   ├── loss.py              # 損失関数（MAE, sMAPE, RMSE）
│   │   └── scaling.py           # 損失スケーリング
│   └── experts/
│       ├── factory.py           # Expert一括生成（light30/light80）
│       ├── vectorized.py        # 予測行列の一括生成（高速版・ホライズン対応）
│       ├── naive.py             # LastValue, SeasonalNaive, Drift
│       ├── moving_avg.py        # SMA, Median
│       ├── smoothing.py         # EMA
│       ├── regression.py        # RidgeLag, HuberLag, KNNLag
│       └── seasonal_profile.py  # STLSeasonalMean
├── scripts/
│   ├── download_data.py         # データセット取得
│   ├── build_site.py            # docs/*.html の表を results から再生成
│   ├── generate_readme_figures.py  # README図生成
│   └── experiments/             # 比較実験（exp0〜exp4）
├── tests/                       # テストスイート
├── data/                        # データセット（.gitignore）
├── docs/                        # GitHub Pages サイト・仕様書・図表
│   ├── index.html               # トップ
│   ├── algorithm.html           # アルゴリズム解説
│   ├── experiment.html          # 実験レポート
│   ├── replication.html         # 追試結果
│   ├── figures/                 # 図
│   └── results/                 # 実験結果 JSON（サイトの数値の出典）
└── reports/                     # 実験出力
```

## 参考文献

1. **Cesa-Bianchi, N. & Lugosi, G.** (2006). *Prediction, Learning, and Games*. Cambridge University Press.
   - Expert Advice の理論的基盤
2. **Freund, Y. & Schapire, R.E.** (1997). A decision-theoretic generalization of on-line learning and an application to boosting. *Journal of Computer and System Sciences*, 55(1), 119-139.
   - Hedge アルゴリズムの原論文
3. **de Rooij, S., van Erven, T., Grünwald, P.D., & Koolen, W.M.** (2014). Follow the leader if you can, hedge if you must. *Journal of Machine Learning Research*, 15, 1281-1316.
   - Meta-learning rate と AdaHedge の理論
4. **Herbster, M. & Warmuth, M.K.** (1998). Tracking the best expert. *Machine Learning*, 32(2), 151-178.
   - Fixed-Share の原論文
5. **Gaillard, P., Stoltz, G., & van Erven, T.** (2014). A second-order bound with excess losses. *COLT*.
   - ML-Poly
6. **Devaine, M., Gaillard, P., Goude, Y., & Stoltz, G.** (2013). Forecasting electricity consumption by aggregating specialized experts. *Machine Learning*, 90(2), 231-260.
   - 電力需要予測への応用
7. **UCI Machine Learning Repository** — Electricity Load Diagrams 2011-2014.
   - 実験データソース。他に M4 Competition (hourly)、ETDataset (ETTh1)、
     Open Power System Data (欧州各国の実測需要) を使用
