# 課題 2: 誤差逆伝播法と勾配降下法による MLP の学習

## 1. 目的

- 課題 1 では重みを手で与えました。この課題では、**データから重みを自動で決める(学習する)** 仕組みを実装します。
- 誤差逆伝播法(バックプロパゲーション)で損失の勾配を求め、勾配降下法で重みを更新する流れを、自分でコードに書いて理解する。
- 「学習 = 損失を最小にする **最適化問題**」「勾配降下法 = その問題を解く **最適化手法** の 1 つ」という見方を身につける。
- 活性化関数(シグモイド・ ReLU ・ステップ)や L2 正則化の違いが、学習にどう影響するかを確かめる。
- 自分でテストを書き、実装が正しいことを **数値微分との比較** や **再現性** で確かめる方法を学ぶ。

## 2. 背景

### 2.1 損失関数

1 サンプルの入力 $x$ に対するネットワークの出力を $\hat{y}$、正解を $y$ とします。 この課題では、ずれの大きさを次の **二乗誤差** で測ります。

$$ L = \frac{1}{2} \sum_k (\hat{y}_k - y_k)^2, \qquad \frac{\partial L}{\partial
\hat{y}_k} = \hat{y}_k - y_k $$

学習曲線に記録する「エポックの損失」は、そのエポックで使った全サンプルの $L$ の平均です。

### 2.2 1 つのニューロンの逆伝播

ニューロンの順伝播は $u = \sum_i w_i x_i + b$、$y = f(u)$ でした($f$ は活性化関数)。
逆伝播では、後ろ(出力側)から「損失をこのニューロンの出力で偏微分した値」 $e = \partial L / \partial y$ を受け取り、連鎖律で次を求めます。

$$ \delta = e \cdot f'(u) $$

$$ \frac{\partial L}{\partial w_i} = \delta \, x_i, \qquad \frac{\partial L}{\partial b}
= \delta, \qquad \frac{\partial L}{\partial x_i} = \delta \, w_i $$

最後の $\partial L / \partial x_i$ が、1 つ前の層に渡す値です。 前の層のあるニューロンの出力は、この層の **全ニューロン**
に入力されているので、層としては全ニューロンの寄与を足し合わせます(行列で書くと $W^\top \delta$)。

### 2.3 誤差逆伝播法(ネットワーク全体)

1. 入力から出力へ順伝播し、各ニューロンの $u$ と入力を覚えておく。
2. 出力層で $\partial L / \partial \hat{y} = \hat{y} - y$ を求める。
3. 出力層 → 入力層の順に、各層の逆伝播を行う。ある層が返した値が、1 つ手前の層が受け取る $e$ になる。

逆伝播の間は重みを変更しません。すべての勾配を求め終わってから、最適化手法でまとめて更新します。

### 2.4 最適化問題と最適化手法

| 役割 | 決めること | この課題での担当 |
| --- | --- | --- |
| 最適化問題 | 何を最小にするか(目的関数 $J$)と、その勾配 | `MLP`(`compute_gradients`)と `loss.py` |
| 最適化手法 | 勾配を使ってパラメータをどう動かすか | `Optimizer` を継承したクラス(`GradientDescent`) |

**勾配降下法** は、パラメータを勾配と逆の向きに、学習率 $\eta$ に比例した分だけ動かします。

$$ w \leftarrow w - \eta \frac{\partial J}{\partial w}, \qquad b \leftarrow b - \eta
\frac{\partial J}{\partial b} $$

この課題では 1 サンプルごとに更新します(このやり方は **確率的勾配降下法(SGD)** とも呼ばれます)。 サンプルは毎エポック、先頭から同じ順番で使います。

最適化手法は `Optimizer` の `step` を実装したクラスとして作るので、`MLP` を変えずに別の手法(例: モーメンタム法)に差し替えられます。

### 2.5 L2 正則化

重みが大きくなりすぎるのを防ぐため、目的関数に重みの二乗和を加えます(バイアスは含めません)。

$$ J = L + \frac{\lambda}{2} \sum_{\text{全ニューロン}} \lVert w \rVert^2, \qquad
\frac{\partial J}{\partial w} = \frac{\partial L}{\partial w} + \lambda w $$

これは「最小にするもの」を変えるので、**最適化問題の側** の変更です。そのため `MLP.compute_gradients` で勾配に $\lambda w$
を加え、`GradientDescent` は変えません。 学習曲線には正則化項を含めない $L$ を記録します。

### 2.6 活性化関数と導関数

| 名前 | $f(u)$ | $f'(u)$ |
| --- | --- | --- |
| シグモイド | $\sigma(u) = 1 / (1 + e^{-u})$ | $\sigma(u)\,(1 - \sigma(u))$ |
| ReLU | $\max(u, 0)$ | $u > 0$ なら 1、それ以外(0 を含む)は 0 |
| ステップ | $u > 0$ なら 1、それ以外は 0 | **シグモイドの導関数で代用**(下記) |

導関数は、どれも $u$(活性化関数に入る前の値)の関数として実装します。

### 2.7 ステップ関数と Straight-Through Estimator(STE)

ステップ関数の本当の導関数は、$u = 0$ 以外ではどこでも 0 です。これでは $\delta$ が常に 0 になり、重みが全く更新されません。

そこで、**順伝播ではステップ関数をそのまま使い、逆伝播のときだけ、形の似たシグモイド関数の導関数を代わりに使います**。 この方法を Straight-Through
Estimator(STE)と呼びます。重みは 1 組だけで、順伝播と逆伝播の両方で同じ重みを使います。 この課題では `step_derivative`
をシグモイドの導関数にすることで実現します(`Neuron` 側に特別な処理は要りません)。

## 3. 課題の内容と進め方

`src/nn_core/` の `raise NotImplementedError` の部分と、`tests/` の本体が `pass`
になっているテスト関数を実装します。 課題 1 で実装した部分(`step`、`forward`、`set_weights` など)は、完成した状態で配布しています。

| ファイル | 内容 |
| --- | --- |
| `activation.py` | 活性化関数と導関数。`get_activation(name)` で (関数, 導関数) の組を取り出す |
| `loss.py` | 損失関数 `squared_error` とその勾配 `squared_error_grad` |
| `neuron.py` | `Neuron`。`backward` で勾配を求める |
| `layer.py` | `Layer`。`backward` で全ニューロンの逆伝播を行う |
| `mlp.py` | `MLP`。`compute_gradients`(勾配を求める)・`train_step`・`train` |
| `optimizer.py` | `Optimizer`(基底クラス)と `GradientDescent`(勾配降下法) |

次の 4 段階で進めてください。各段階のテストには `stage1`〜`stage4` のマーカーが付いているので、`pytest -m stage1`
のようにその段階のテストだけを実行できます。 ソースコードのヒントにも【段階 N】と書いてあります。

### 段階 1: シグモイド関数と学習

| 実装する場所 | 穴埋めのテスト |
| --- | --- |
| `activation.py`: `sigmoid`, `sigmoid_derivative` | `test_activation.py`: `test_sigmoid_derivative_matches_numerical_derivative` |
| `loss.py`: `squared_error`, `squared_error_grad` | `test_loss.py`: `test_squared_error_grad_matches_numerical_gradient` |
| `neuron.py`: `Neuron.backward` | `test_neuron.py`: `test_backward_gradients_match_hand_calculation` ほか計 3 つ |
| `layer.py`: `Layer.backward` | `test_layer.py`: `test_backward_returns_sum_over_neurons` ほか計 2 つ |
| `optimizer.py`: `GradientDescent.step` | `test_optimizer.py`: `test_step_moves_parameters_against_gradient` ほか計 2 つ |
| `mlp.py`: `compute_gradients`(L2 以外)、`train_step`、`train` | `test_backprop.py`: `numerical_gradients`、`test_gradient_check_sigmoid`、`test_train_step_*` |
|  | `test_training.py`: 再現性・参照値・損失の減少・ NAND / XOR の学習(シグモイド) |

```bash
pytest -m stage1 -v
python src/main.py --gate xor --activation sigmoid
```

### 段階 2: L2 正則化

| 実装する場所 | 穴埋めのテスト |
| --- | --- |
| `mlp.py`: `compute_gradients` の【段階 2】の部分 | `test_backprop.py`: `test_gradient_check_with_l2`、`test_l2_adds_lambda_times_weights_to_gradient` |
|  | `test_training.py`: `test_l2_learns_xor_with_smaller_weights` |

```bash
pytest -m "stage1 or stage2" -v
python src/main.py --gate xor --activation sigmoid --l2
```

L2 ありとなしで、学習後の重みの大きさ(`Trained weights`)と決定境界を比べてみましょう。

### 段階 3: ReLU 関数

| 実装する場所 | 穴埋めのテスト |
| --- | --- |
| `activation.py`: `relu`, `relu_derivative` | `test_activation.py`: `test_relu_derivative` |
|  | `test_neuron.py`: `test_backward_relu_inactive_gives_zero_gradient` |
|  | `test_backprop.py`: `test_gradient_check_relu` |
|  | `test_training.py`: `test_learns_xor_with_relu_and_l2` |

```bash
pytest -m "stage1 or stage2 or stage3" -v
python src/main.py --gate xor --activation relu --l2
```

ReLU は出力に上限がなく、$u \le 0$ では勾配が 0 になります。 そのため、どの入力に対しても $u \le 0$
になったニューロンは、それ以降まったく更新されなくなります(dying ReLU)。 段階 2 を先に実装しておくと、ReLU で L2 あり・なしを比べられます。
$\lambda$ を大きくしすぎる(例: `--l2-lambda 0.1`)と、重みが 0 に押しつぶされて学習できなくなる様子も観察できます(発展課題 3)。

### 段階 4: ステップ関数(STE)

| 実装する場所 | 穴埋めのテスト |
| --- | --- |
| `activation.py`: `step_derivative` | `test_activation.py`: `test_step_derivative_is_sigmoid_derivative` |
|  | `test_neuron.py`: `test_backward_step_uses_sigmoid_derivative` |
|  | `test_training.py`: `test_learns_nand_with_step`、`test_learns_xor_with_step` |

```bash
pytest -v                          # すべてのテスト
python src/main.py --gate xor --activation step
```

学習後の重みで課題 1 の `main.py --layer ...` を実行すると、同じ真理値表が出るはずです(試してみましょう)。

## 4. 要件(API 仕様)

テストはこの仕様に従って書かれています。クラス名・メソッド名・引数・属性名は変えないでください。

### `activation.py`

| 名前 | 内容 |
| --- | --- |
| `step(u)`, `sigmoid(u)`, `relu(u)` | 活性化関数。スカラーでも配列でも受け取り、配列なら同じ shape で返す |
| `sigmoid_derivative(u)`, `relu_derivative(u)`, `step_derivative(u)` | 各活性化関数の導関数(`u` の関数)。`step_derivative` は STE |
| `get_activation(name)` | `"sigmoid"` / `"relu"` / `"step"` から `(f, df)` を返す。未知の名前は `ValueError` |

`sigmoid` は `u` が非常に大きい・小さい(例: ±1000)ときもオーバーフローしないこと。

### `loss.py`

| 名前 | 内容 |
| --- | --- |
| `squared_error(pred, target)` | $\frac{1}{2}\sum_k (\text{pred}_k - \text{target}_k)^2$ を `float` で返す |
| `squared_error_grad(pred, target)` | `squared_error` を `pred` で偏微分した値。shape は `pred` と同じ |

### `Neuron(num_inputs, rng=None, activation="sigmoid")`

| 名前 | 内容 |
| --- | --- |
| `weights`, `bias` | 重み(shape `(num_inputs,)`)とバイアス。初期化は課題 1 と同じ |
| `activation_fn`, `activation_derivative` | 活性化関数とその導関数(`get_activation` で取り出したもの) |
| `inputs`, `u`, `output` | 直前の `forward` の入力・重み付き和 + バイアス・出力 |
| `grad_weights`, `grad_bias` | 直前の `backward` で求めた勾配。生成直後は 0 |
| `forward(inputs)` | 出力 $f(u)$ を `float` で返し、`inputs`・`u`・`output` を保存する(実装済み) |
| `backward(error)` | `error` $= \partial L/\partial y$ から勾配を求めて保存し、$\delta w$ を返す。重みは変更しない |

### `Layer(num_neurons, num_inputs, rng=None, activation="sigmoid")`

| 名前 | 内容 |
| --- | --- |
| `forward(inputs)` | shape `(num_neurons,)` の配列を返す(実装済み) |
| `backward(errors)` | `errors`(shape `(num_neurons,)`)を各ニューロンに渡し、戻り値の合計(shape `(num_inputs,)`)を返す |

### `MLP(layer_sizes, seed=None, activation="sigmoid", output_activation=None)`

| 名前 | 内容 |
| --- | --- |
| `activations` | 各層の活性化関数の名前。隠れ層は `activation`、出力層は `output_activation`(`None` なら `activation`) |
| `neurons()` | 入力に近い層から順に、全ニューロンを返す(実装済み) |
| `predict(inputs)` | shape `(N, 入力数)` を受け取り shape `(N, 出力数)` を返す(実装済み) |
| `compute_gradients(x, y, l2_lambda=0.0)` | 1 サンプルについて目的関数 $J$ の勾配を全ニューロンに求め、損失 $L$ を返す。重みは変更しない |
| `train_step(x, y, optimizer, l2_lambda=0.0)` | 勾配を求めて `optimizer.step(self)` で 1 回更新し、更新前の損失 $L$ を返す |
| `train(inputs, targets, epochs, optimizer, l2_lambda=0.0)` | 全サンプルを先頭から順に `train_step` することを `epochs` 回繰り返し、エポックの損失のリストを返す。 同じリストを `loss_history` の後ろにも追加する |

### `optimizer.py`

| 名前 | 内容 |
| --- | --- |
| `Optimizer` | 最適化手法の基底クラス(抽象クラス)。`step(mlp)` を実装したクラスだけがインスタンスを作れる |
| `GradientDescent(learning_rate)` | 勾配降下法。`step(mlp)` で全ニューロンの `weights` / `bias` を更新する |

### 乱数と再現性

`MLP(..., seed=S)` は、課題 1 と同じく `np.random.default_rng(S)` で乱数生成器を 1 つ作り、全ニューロンの初期化に使います。
学習中に乱数は使わない(サンプルの順番も固定)ので、**同じシード・同じ設定なら、学習曲線も学習後の重みも毎回同じ** になります。 テストや `main.py`
では、学習が成功することを確認済みのシードを固定で指定しています。

### 使ってよいライブラリ

- 使ってよい: Python 標準ライブラリ、NumPy(図の保存に matplotlib、テストに pytest)
- 使ってはいけない: Keras、TensorFlow、PyTorch、scikit-learn などの機械学習フレームワーク

## 5. テストについて

### 完成済みのテストと穴埋めのテスト

`tests/` には、完成済みのテストと、**本体が `pass` の穴埋めのテスト** が混ざっています。 穴埋めのテストは、docstring
に「条件」と「期待」(何を確かめるか、許容誤差)、コメントに「ヒント」が書いてあるので、それに従って本体を書いてください。

| ファイル | 内容 | 完成 / 穴埋め |
| --- | --- | --- |
| `test_activation.py` | 活性化関数の値・導関数(数値微分との比較) | 6 / 3 |
| `test_loss.py` | 損失関数とその勾配 | 1 / 1 |
| `test_neuron.py` | ニューロンの順伝播・逆伝播(手計算との比較) | 3 / 5 |
| `test_layer.py` | 層の構成・逆伝播 | 2 / 2 |
| `test_mlp.py` | MLP の構成・順伝播(課題 1 からの引き継ぎ) | 5 / 0 |
| `test_optimizer.py` | 最適化手法 | 3 / 2 |
| `test_backprop.py` | 勾配チェック(逆伝播と数値微分の比較)・ 1 回の更新 | 0 / 6 + ヘルパー 1 |
| `test_training.py` | 学習の再現性・参照値・ NAND / XOR の学習 | 2 / 9 |

> **注意: 本体が `pass` のテストは、何も確かめずに「合格」になります。** 全テストが合格していても、穴埋めのテストを書き終えていなければ課題は終わっていません。

書いたテストが本当に役に立つか確かめるには、**実装をわざと間違えて、テストが失敗するか** を見るのが確実です。 例えば `GradientDescent.step` の
`-` を `+` に変える、`Neuron.backward` で $f'(u)$ を掛け忘れる、などを試してみましょう(確かめたら元に戻すこと)。

### 数値微分による勾配チェック

関数 $J(w)$ の $w$ での傾きは、小さな $h$ を使って次のように近似できます(中心差分)。

$$ \frac{\partial J}{\partial w} \approx \frac{J(w + h) - J(w - h)}{2h} $$

逆伝播で求めた勾配がこの近似値と一致すれば、逆伝播の実装はほぼ確実に正しいと言えます。 `test_backprop.py` では $h = 10^{-6}$
とし、許容誤差を相対 $10^{-5}$・絶対 $10^{-8}$ としています。

### 許容誤差

浮動小数点数の計算には丸め誤差があるので、実数の比較には `pytest.approx` を使い、**許容誤差を必ず明記** します。

| 比べるもの | 許容誤差 | 理由 |
| --- | --- | --- |
| 同じ計算を 2 回した結果(再現性) | rel=1e-12, abs=1e-12 | 同じ手順なら、ほぼ完全に一致するはず |
| 参照値・手計算の値 | rel=1e-6, abs=1e-9 など | 計算の順番の違いによる、ごく小さな差を許す |
| 数値微分との比較 | rel=1e-5, abs=1e-8 | 数値微分そのものが近似なので、少し緩める |

## 6. 実行方法

### 準備

```bash
pip install numpy matplotlib pytest
```

### テスト

`ex2/` ディレクトリで実行します。

```bash
cd ex2
pytest -v                          # すべて
pytest -m stage1 -v                # 段階 1 のテストだけ
pytest -v tests/test_backprop.py   # 1 つのファイルだけ
```

### 動作確認用スクリプト `main.py`

NAND / XOR を学習させ、学習経過・予測・学習後の重みを表示して、決定境界と学習曲線の図を PNG に保存します。 `ex2/` ディレクトリで実行します。

```bash
python src/main.py --gate xor --activation sigmoid          # → result_xor_sigmoid_no_l2_seed2.png
python src/main.py --gate xor --activation relu --l2        # → result_xor_relu_l2_0.001_seed0.png
python src/main.py --gate nand --activation step
```

| 引数 | 説明 | デフォルト |
| --- | --- | --- |
| `--gate` | 学習させるゲート(`nand` / `xor`) | `xor` |
| `--activation` | 隠れ層の活性化関数(`sigmoid` / `relu` / `step`) | `sigmoid` |
| `--output-activation` | 出力層の活性化関数 | `--activation` と同じ |
| `--hidden` | 隠れ層のニューロン数(例: `--hidden 3`、`--hidden 4 4`)。`--hidden` だけなら隠れ層なし | nand: なし、xor: 2 |
| `--l2` | L2 正則化を有効にする | 無効 |
| `--l2-lambda` | L2 正則化の強さ $\lambda$(`--l2` 指定時のみ有効) | `0.001` |
| `--seed` / `--learning-rate` / `--epochs` | シード・学習率・エポック数 | 下の推奨設定 |

`--seed`・`--learning-rate`・`--epochs` を省略すると、次の **推奨設定**(既定の隠れ層で、L2 あり・なしのどちらでも正解率 100%
になることを確認済み)を使います。

| `--gate` | `--activation` | シード | 学習率 | エポック数 |
| --- | --- | --- | --- | --- |
| nand | sigmoid | 0 | 0.5 | 500 |
| nand | relu | 0 | 0.05 | 300 |
| nand | step | 0 | 0.5 | 50 |
| xor | sigmoid | 2 | 0.5 | 3000 |
| xor | relu | 0 | 0.05 | 1000 |
| xor | step | 4 | 0.5 | 300 |

シードを変えると、学習に失敗する(正解率が 100% にならない)こともあります。 初期値によって結果が変わることも観察してみましょう。

## 7. NumPy と pytest の追加の API

課題 1 の README の解説に加えて、この課題では次の API を使います。

### NumPy

| API | 説明 | 例 |
| --- | --- | --- |
| `np.exp(a)` | 要素ごとの $e^a$ | `np.exp(0.0)` → `1.0` |
| `np.clip(a, lo, hi)` | 各要素を `lo` 以上 `hi` 以下に切り詰める | `np.clip([-900, 3], -500, 500)` → `[-500, 3]` |
| `np.maximum(a, b)` | 要素ごとの大きい方 | `np.maximum([-1, 2], 0)` → `[0, 2]` |
| `np.zeros_like(a)` | `a` と同じ shape ・型の、0 で埋めた配列 |  |
| `a.copy()` | 配列のコピー(元の配列を書き換えても影響を受けない) |  |
| `a.T`、`A @ v` | 転置、行列とベクトルの積 | `W.T @ delta` |
| `np.linalg.norm(a)` / `np.sqrt(np.sum(a**2))` | ベクトルの大きさ(2 乗和の平方根) |  |
| `np.errstate(over="raise")` | `with` の中で、オーバーフローを警告ではなく例外にする(テストで使用) |  |

配列の要素を `a[i] = ...` で書き換えると、その配列を持っている全員から変化が見えます。
勾配チェックでは、重みを少しずらして目的関数を計算した後、**必ず元の値に戻して** ください。

### pytest

| API | 説明 |
| --- | --- |
| `pytest.approx(expected, rel=..., abs=...)` | 実数や配列・リストを「ほぼ等しいか」で比べる。`rel` は相対誤差、`abs` は絶対誤差で、どちらかを満たせば等しいとみなす |
| `@pytest.mark.stage1` | テストに「印(マーカー)」を付ける。`pytest -m stage1` で印の付いたテストだけを実行できる |
| `pytest -m "stage1 or stage2"` | 複数のマーカーのどれかが付いたテストを実行する |
| `conftest.py` の `pytest_configure` | マーカーを pytest に登録する(登録しないと警告が出る) |

`pytest.approx` の例:

```python
assert 0.1 + 0.2 == pytest.approx(0.3, abs=1e-12)
assert [1.0, 2.0000001] == pytest.approx([1.0, 2.0], rel=1e-6)
```

## 8. 評価基準

1. `src/nn_core/` の穴埋めを実装し、`pytest` の全テストに合格すること。
2. `tests/` の穴埋めのテストを、docstring の条件・期待・許容誤差どおりに実装していること。 (完成済みのテストや、docstring
   の仕様は変更しないこと)
3. 書いたテストが、実装をわざと間違えたときに失敗すること(「5. テストについて」参照)。
4. 逆伝播で重みを変更せず、更新を `Optimizer` に任せる構成になっていること。
5. `main.py` の推奨設定で、NAND / XOR を正解率 100% で学習できること。

## 9. 発展課題(任意)

1. `Optimizer` を継承して **モーメンタム法**(前回の更新量の一部を今回の更新に足す)を実装し、`GradientDescent` と学習曲線を比べよう。
2. XOR をシグモイドで学習させるとき、隠れ層のニューロン数(`--hidden`)を 2 ・ 3 ・ 4 と変えて、学習に成功するシードの割合を調べよう。
3. `--l2 --l2-lambda` を 0.001、0.01、0.1 と大きくしていくと、学習後の重み・決定境界・学習曲線はどう変わるか、 シグモイドと ReLU
   で調べよう。
4. `--activation relu --output-activation sigmoid` のように、隠れ層と出力層で活性化関数を変えるとどうなるか試そう。
