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

## Keras を用いた MLP の実装

### Usage

```bash
cd model/mlp_in_keras
uv sync
uv run mlp_in_keras.py # 通常の MNIST を 3 層 MLP で識別
uv run mlp_in_keras_bmnist.py # 7x7 に圧縮した MNIST (グレイスケールまたは2値）を MLP で識別
```
