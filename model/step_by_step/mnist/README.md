# MNIST MLP (step by step, NumPy only)

[`xor`](../xor/README.md) と同じ自作 MLP([`nn_core`](../nn_core/))で、縮減した MNIST を分類します。 MNIST
は 28x28 のまま使わず、**4x4 平均プーリングで 7x7 (49 画素)にし、二値化**してから入力します (`mlp_in_keras/converter.py`
の `Converter` を再利用)。アナログ回路への実装を見据えた縮減です。

## 1. 使い方 (Usage)

```bash
# 依存パッケージ(tensorflow は初回のデータ取得時のみ使用。データは ./data/mnist.npz にキャッシュされ、2 回目以降は読み込まない)
pip install numpy matplotlib tensorflow

# デフォルト: 0/1 の2クラス、層構造 [49, 2, 2]
python3 main.py

# ミニバッチ(ベクトル化)で高速に学習
python3 main.py --epochs 100 --batch-size 32 --learning-rate 0.05

# 3クラス(0/1/2)、隠れ層 8 ユニット → 層構造 [49, 8, 3]
python3 main.py --classes 0 1 2 --hidden 8 --epochs 100 --batch-size 32 --learning-rate 0.05
```

主なコマンドライン引数:

| 引数 | 説明 | デフォルト |
| --- | --- | --- |
| `--classes` | 分類する数字(0-9、2 個以上、重複不可) | `0 1` |
| `--hidden` | 隠れ層のユニット数(複数指定可、例 `--hidden 8 4`) | `2` |
| `--activation` | 隠れ層の活性化関数(`relu` / `sigmoid`) | `relu` |
| `--output-activation` | 出力層の活性化関数(`relu` / `sigmoid`) | `relu` |
| `--epochs` | 学習エポック数 | `20` |
| `--learning-rate` | 学習率 | `0.01` |
| `--batch-size` | `1`: 1 サンプルずつの SGD(スカラー版)、`2` 以上: ミニバッチ(ベクトル版) | `1` |
| `--l2` / `--l2-lambda` | L2 正則化の有効化フラグ / 強さ | 無効 / `0.001` |
| `--seed` | 重み初期化・シャッフルの乱数シード | `0` |

層構造は `[49, *hidden, クラス数]` になります。入力層は画素数(49)、出力層はクラス数で固定です。

実行すると、標準出力に学習経過(MSE)・ train/test accuracy ・混同行列が表示され、次のファイルが保存されます (`<run_name>` は
`クラス_層構造`、例 `01_49-2-2`)。

- `weights_<run_name>/`: 重み。`mlp_in_keras/trainer.py` の `save_model_weights` と同じ形式です。
  - `layer_{i}_weights.csv`: 形状 (入力数, ニューロン数)。行が入力、列がニューロン(`i` は 0 始まり)
  - `layer_{i}_bias.csv`: 1 列、ニューロン数の行
  - `model_weights.npz`: 上記と同じ `layer_{i}_weights` / `layer_{i}_bias` に加えて `layers`,
    `classes`
- `loss_<run_name>.png`: 学習曲線(MSE, 対数軸)

## 2. 設計上のポイント

- **出力層は ReLU、損失は MSE(softmax は不使用)**: 正解クラスの出力が 1、他が 0 になる one-hot を目標とし、 推論は出力の
  `argmax` です。最も活性化したクラスが答えになるので、3 クラス以上にもそのまま拡張できます。 回路実装では「ReLU 出力 → 最大値選択」となり、softmax
  より単純です。 softmax はニューロン単位でなく層全体に依存するため、`Layer` を大きく変える必要があり採用しませんでした。
- **同点は不正解**: 全出力が 0(同点)のサンプルは、`argmax` が先頭クラスを返してしまうため「未決定」として不正解に数えます。
  混同行列の最後の列が未決定の数です。
- **入力の正規化**: 二値化後の 0/255 を 0/1 に直して入力します。
- **ミニバッチ**: 勾配はバッチ内で平均するため、学習率の意味はバッチサイズに依存しません。
  スカラー版は「更新後の重みで誤差を逆伝播」、ベクトル版は「更新前の重み」を使うため、`--batch-size 1` と `2` 以上の結果は厳密には一致しません。
- **データの縮減**: 対象クラスの画像だけを取り出してから学習します(`Converter.extract_labels`)。

## 3. 実行結果のサンプル

| 条件 | 学習時間 | test accuracy |
| --- | --- | --- |
| 0/1、`--batch-size 1`、20 epoch | 5.5s | 98.2% |
| 0/1、`--batch-size 32`、100 epoch、lr 0.05 | 1.4s | 98.0% |
| 0/1/2、`--hidden 8`、`--batch-size 32`、100 epoch、lr 0.05 | - | 89.9% |

(いずれも `--seed 0`、1 回の実行結果。)同じ epoch 数に換算すると、ミニバッチ版はスカラー版の約 20 倍速です。

### 0/1、層構造 [49, 2, 2]

![loss 01](img/loss_01_49-2-2.png)

### 0/1/2、層構造 [49, 8, 3]

![loss 012](img/loss_012_49-8-3.png)

> **注意**: ReLU 出力は、正解クラスの出力が 0 に張り付くと勾配が 0 になり学習が進まなくなる(dying ReLU)ことがあります。
> シードによる差が出やすいので、複数の `--seed` で確認してください。うまくいかない場合は `--output-activation sigmoid` や
> より小さい学習率を試してください。隠れ層 2 ユニットは小さく、クラス数を増やすときは `--hidden` を大きくする必要があります。

## 4. 各コードの解説

### [`main.py`](main.py)

`argparse` で引数を受け取り、データ読み込み(`load_data`)→ 学習(`NeuralNetwork.train`)→ 評価(`evaluate`)→
重み・図の保存を行います。

- `load_data`: `tensorflow.keras.datasets.mnist` を読み込み、`Converter.pooling_4x4` →
  `binarize` → `extract_labels` の順に縮減し、 ラベルを `--classes` 内のインデックスに振り直します。
- `evaluate`: `NeuralNetwork.predict_batch` で全サンプルを一括推論し、accuracy と混同行列を返します。

### [`../nn_core/`](../nn_core/)

`Neuron` / `Layer`(スカラー版)、`VLayer`(ベクトル版)、`NeuralNetwork` の解説は
[`xor/README.md`](../xor/README.md#3-各コードの解説) と [`step_by_step/README.md`](../README.md)
を参照してください。
