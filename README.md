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

### LaTeX のコンパイル

```latex
cd latex
pdflatex main.tex
```

`main.pdf` が生成されます。

## Keras による MLP の実装

### Setup

[uv](https://docs.astral.sh/uv/getting-started/installation/) のインストールを実施した後、Python
環境を構築してください。

次のコマンドを実行すると一括でできます。

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh # uv のインストール
uv sync # パッケージをアップデート
uv sync --check # 環境をチェック
```

### Usage

```bash
cd model/mlp_in_keras
uv sync
uv run mlp_in_keras.py # 通常の MNIST を 3 層 MLP で識別
uv run mlp_in_keras_bmnist.py # 7x7 に圧縮した MNIST (グレイスケールまたは2値）を MLP で識別
```
