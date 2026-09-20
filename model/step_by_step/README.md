# step_by_step

NumPy のみで実装した MLP を、小さな題材から段階的に試すためのディレクトリです。

| ディレクトリ | 内容 |
| --- | --- |
| [`nn_core/`](nn_core/) | 共通のニューラルネットワーク実装(`Neuron` / `Layer` / `VLayer` / `NeuralNetwork`) |
| [`xor/`](xor/) | XOR 問題。L2 正則化・活性化関数による収束性の違いを比較する([README](xor/README.md)) |
| [`mnist/`](mnist/) | 7x7 にプーリング+二値化した MNIST の分類(デフォルトは 0/1 の 2 クラス)([README](mnist/README.md)) |

## nn_core の構成

- [`neuron.py`](nn_core/neuron.py): 1 ニューロンの重み・バイアス・活性化(`relu` / `sigmoid`)・重み更新。
- [`layer.py`](nn_core/layer.py): `Neuron` をまとめたスカラー版の層。1 サンプルずつ処理する(参照実装)。
- [`vlayer.py`](nn_core/vlayer.py): 重み行列 `W` とバイアス `b`
  を配列で持つベクトル版の層。ミニバッチを行列演算でまとめて処理する。
- [`neuralnetwork.py`](nn_core/neuralnetwork.py): 層を束ねて学習・推論を行う。`batch_size == 1`
  ならスカラー版、`2` 以上なら `VLayer` を使う。 隠れ層と出力層で活性化関数を分けられる(`activation` /
  `output_activation`)。

各サブディレクトリのスクリプトは、親ディレクトリを `sys.path` に追加して `from nn_core import NeuralNetwork` で読み込みます。
実行はそれぞれのディレクトリ内で行ってください。
