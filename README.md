# アナログ回路で作るニューラルネットワーク

## 目次

1. パーセプトロン
    1. NAND の実装
1. MLP
    1. XOR の実装 
        1. NAND の組み合わせ(4つで構築)
        1. 入力の組み合わせ
1. MNIST
    1. MLP による学習
    1. LTSPICE による回路シミュレーション1 （上記全てを再現)
    1. 枝刈りによる計算量の削減
    1. LTSPICE による回路シミュレーション2 (削減された回路)
    1. 回路への実装
1. 付録
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

`uv` のインストールを実施してください。

### Usage

```bash
cd model/mlp_in_keras
uv sync
uv run mlp_in_keras.py # 通常の MNIST を 3 層 MLP で識別
uv run mlp_in_keras_bmnist.py # 7x7 に圧縮した MNIST (グレイスケールまたは2値）を MLP で識別
```
