# アナログ回路で作るニューラルネットワーク

## 目次

1. パーセプトロン
   1. NAND の実装
2. MLP
   1. XOR の実装
      1. NAND の組み合わせ(4 つで構築)
      2. 入力の組み合わせ
3. MNIST
   1. MLP による学習
   2. LTSPICE による回路シミュレーション 1 （上記全てを再現)
   3. 枝刈りによる計算量の削減
   4. LTSPICE による回路シミュレーション 2 (削減された回路)
   5. 回路への実装
4. 付録
   1. マイコンとの組み合わせ: DAC/ADC の利用による計算素子の削減

## Usage

### Setup

[uv](https://docs.astral.sh/uv/getting-started/installation/) のインストールを実施した後、Python
環境を構築してください。

次のコマンドを実行すると一括でできます。

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh # uv のインストール
uv sync && uv sync --check # パッケージをアップデート
uv pip install --upgrade  "tensorflow[and-cuda]" # CUDA 対応 tensorflow をインストール（任意。CUDAが必要な程モデルは大きくない）
uv tool install pre-commit
uv tool install flowmark # tool (コマンド) としてパッケージをインストール
pre-commit install # .git/hooks/pre-commit の設定
```

### LaTeX のコンパイル

```latex
cd latex
pdflatex main.tex
```

`main.pdf` が生成されます。

### pre-commit

このプロジェクトでは Git の `pre-commit` の設定に [pre-commit](https://pre-commit.com/) を利用しています。
インストールは [Setup](#setup) に記載しています。

`pre-commit` で設定されたコマンドを全て実行するには次のコマンドを実行してください。

```bash
pre-commit run --all
```

## Keras を用いた単純パーセプトロンによる NAND ゲートの実装

### Usage

```bash
cd model/nand
uv sync
uv run nand.py # Sigmoid, ReLU, Sigmoid-STE による NAND の実装
uv run nand_infer.py --weights_csv=./nand_weights_sigmoid --bias_csv=./nand_biases_sigmoid.csv # 外部からパラメータを与えた NAND の実装の評価
```

`nand_infer.py`は外部からパラメータを与えることができます。 デフォルトでは同じパラメータを与えた条件下で活性化関数が Sigmoid 関数と Step 関数の
2 条件で NAND 演算の精度を比較します。

詳細は `--help` オプションで確認できます。

### Sigmoid STE によるステップ関数を活性化関数とする NAND ゲートの実装

```bash
cd model/nand
uv run eval_nand_by_sigmoid_ste.py --runs 30 --epochs 50
```

2 入力 1 出力の単純パーセプトロンの活性関数に対し、

* 訓練：Sigmoid 関数
* 推論：Step 関数

とした場合の NAND ゲートとして推論精度を評価します。

#### 実行例

50 エポックで 100 回試行したときの混合行列は次の通りです。

| Sigmoid / Step | 0 | 1 |
| --- | --- | --- |
| 0 | 106 | 0 |
| 1 | 0 | 294 |

この実行結果では 1 つの入力の組に対しての出力が同一のパラメータを使用した場合、活性化関数を Sigmoid から Step
へ変更したとき推論結果が一致するかについて表現しています。

例えば、Sigmoid 関数を用いて入力を(1,1)で与えたとき 0 を出力した場合、Step 関数を用いて同様に入力を(1,1)で与えたとき 0
を出力した場合、Sigmoid=0, Step=0 として False-Negative にカウントしています。

一方で、Sigmoid 関数を用いて入力を(1,1)で与えたとき 1 を出力した場合（つまり学習に失敗している場合）、Step 関数を用いて同様に入力を(1,1)で与えたとき
1 を出力した場合、Sigmoid=1, Step=1 として True-Positive にカウントしています。

上記の結果より、 Sigmoid 関数を活性化関数として訓練して得られたモデルのパラメータは Step 関数を活性化関数として推論した場合、結果が全て一致していました。
これより、Sigmoid STE によるステップ関数を活性化関数とする NAND ゲートとしてパラメータの転用は効果があるといえます。

## Keras を用いた MLP の実装

### Usage

```bash
cd model/mlp_in_keras
uv sync
uv run mlp_in_keras.py # 通常の MNIST を 3 層 MLP で識別
uv run mlp_in_keras_bmnist.py # 7x7 に圧縮した MNIST (グレイスケールまたは2値）を MLP で識別
```
