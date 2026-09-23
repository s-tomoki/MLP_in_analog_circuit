# 課題 1: ニューロンと MLP の順伝播(Step 活性化)

## 1. 目的

- ニューロン 1 個が「入力 → 重み付き和 → バイアス → 活性化関数 → 出力」の順に計算することを、自分でコードに書いて理解する。
- ニューロンを並べて **層(Layer)**、層をつないで **多層パーセプトロン(MLP)** を組み立てる。
- 重みを手で設定するだけで NAND や XOR の論理ゲートが作れることを確かめる。 特に、XOR には **隠れ層が必要** であることを体感する。

この課題では学習(重みの自動調整)はまだ扱いません。重みはすべて手で与えます。

## 2. 背景

### ニューロンとステップ関数

入力 $x_1, \dots, x_n$、重み $w_1, \dots, w_n$、バイアス $b$ を持つニューロンの出力 $y$ は次のとおりです。

$$ u = \sum_{i=1}^{n} w_i x_i + b, \qquad y = \mathrm{step}(u) $$

$$ \mathrm{step}(u) = \begin{cases} 1 & (u > 0) \\
0 & (u \le 0) \end{cases} $$

$u = 0$ のときは **0** とします。

### NAND と XOR

- NAND は、ニューロン 1 個(直線 1 本での区切り)で実現できます。
- XOR は、平面上の 4 点を直線 1 本では正しく区切れません。 そのため、隠れ層を 1 つ挟み、**XOR = AND(OR, NAND)**
  のように分解して実現します。

## 3. 課題の内容

`src/nn_core/` 内の以下のファイルにある `raise NotImplementedError` の部分を実装してください。
各メソッドに日本語のヒントがコメントで書いてあります。

| ファイル | 実装するもの |
| --- | --- |
| `activation.py` | `step` |
| `neuron.py` | `Neuron`(`__init__`, `set_weights`, `forward`) |
| `layer.py` | `Layer`(`__init__`, `set_weights`, `get_weights`, `forward`) |
| `mlp.py` | `MLP`(`__init__`, `set_weights`, `get_weights`, `forward`) |

おすすめの順番は `activation.py` → `neuron.py` → `layer.py` → `mlp.py` です。 1
つ実装するごとに、そのファイルに対応するテストを実行して確かめましょう。

`src/nn_core/__init__.py`、`src/main.py`、`tests/` は完成済みです。**変更しないでください。**

## 4. 要件(API 仕様)

テストはこの仕様に従って書かれています。クラス名・メソッド名・引数・属性名は変えないでください。

### `step(x)`

- スカラーでも NumPy 配列でも受け取れること。配列の場合は要素ごとに適用し、同じ shape で返す。
- `x > 0` なら 1、それ以外(`x == 0` を含む)は 0。

### `Neuron(num_inputs, rng=None)`

| 名前 | 内容 |
| --- | --- |
| `weights` 属性 | shape `(num_inputs,)` の NumPy 配列。初期値は標準正規分布の乱数 |
| `bias` 属性 | 実数。初期値は [0, 1) の一様乱数 |
| `set_weights(weights, bias)` | 重み・バイアスを置き換える。`weights` の長さが違えば `ValueError` |
| `forward(inputs)` | 重み付き和 → バイアスを加える → `step` の順に計算し、0 か 1 を返す |

乱数は引数 `rng`(`np.random.Generator`)から取り出してください。`None` なら新しく作ります。

### `Layer(num_neurons, num_inputs, rng=None)`

| 名前 | 内容 |
| --- | --- |
| `neurons` 属性 | `Neuron` のリスト(長さ `num_neurons`) |
| `set_weights(W, b)` | `W`: shape `(num_neurons, num_inputs)`、`b`: shape `(num_neurons,)`。shape が違えば `ValueError` |
| `get_weights()` | `(W, b)` を返す(形は `set_weights` と同じ) |
| `forward(inputs)` | 全ニューロンの出力を並べた、shape `(num_neurons,)` の NumPy 配列を返す |

### `MLP(layer_sizes, seed=None)`

| 名前 | 内容 |
| --- | --- |
| `layer_sizes` 属性 | 入力層を含む各層のユニット数。例: `[2, 2, 1]`。要素数が 2 未満なら `ValueError` |
| `layers` 属性 | `Layer` のリスト(入力層は含まないので、長さは `len(layer_sizes) - 1`) |
| `set_weights(params)` | `params = [(W1, b1), (W2, b2), ...]`。層の数が違えば `ValueError` |
| `get_weights()` | `[(W1, b1), (W2, b2), ...]` を返す |
| `forward(inputs)` | 各層の出力を次の層の入力にして出力層まで計算し、shape `(layer_sizes[-1],)` の NumPy 配列を返す |

`seed` から乱数生成器を 1 つ作り、すべての層・ニューロンで共有してください。

### 使ってよいライブラリ

- 使ってよい: Python 標準ライブラリ、NumPy
- 使ってはいけない: Keras、TensorFlow、PyTorch、scikit-learn などの機械学習フレームワーク

## 5. 実行方法

### 準備

```bash
pip install numpy matplotlib pytest
```

### テスト

`ex1/` ディレクトリで実行します。

```bash
cd ex1
pytest -v                          # すべてのテスト
pytest -v tests/test_neuron.py     # Neuron のテストだけ
```

| テストファイル | 確認する内容 |
| --- | --- |
| `test_activation.py` | `step` の値(0 ちょうどの扱いを含む)と配列入力 |
| `test_neuron.py` | 初期化、重みの設定、順伝播、ニューロン 1 個での NAND |
| `test_layer.py` | 層の構成、重みの設定と取得、順伝播 |
| `test_mlp.py` | ネットワークの構成、シードによる再現性、重みの設定と取得、層から層への受け渡し |
| `test_logic_gates.py` | 決められた重みの MLP で、NAND ・ XOR の真理値表が正しく出ること |

テストで使う重みは次のとおりです。

| ゲート | 構成 | 重み | バイアス |
| --- | --- | --- | --- |
| NAND | `[2, 1]` | `(-0.5, -0.5)` | `0.8` |
| XOR 隠れ層 | `[2, 2, 1]` の 1 層目 | `(0.8, 0.8)`, `(-0.8, -0.8)` | `-0.5`, `0.9` |
| XOR 出力層 | `[2, 2, 1]` の 2 層目 | `(0.8, 0.8)` | `-1.2` |

### 動作確認用スクリプト `main.py`

実装した MLP に重みを与え、真理値表を表示して、決定境界の図を PNG に保存します(学習はしません)。 `ex1/` ディレクトリで実行します。

```bash
python src/main.py --gate xor     # XOR のプリセット重みで実行 → result_xor_preset.png
python src/main.py --gate nand    # NAND のプリセット重みで実行 → result_nand_preset.png
```

| 引数 | 説明 | デフォルト |
| --- | --- | --- |
| `--gate` | 目標のゲート(`nand` / `xor`)。プリセット重みの選択と、正解率の計算に使う | `xor` |
| `--layer` | 1 層分の重みを直接指定する(プリセットを上書き)。書き方は下記 | なし |

#### `--layer` で重みを指定する

- `--layer` を **1 回書くと 1 層** になります(入力層は数えません)。書いた順に並び、**最後の `--layer` が出力層** です。
- 1 つのニューロンは「**重み(入力の個数ぶん)→ バイアス**」の順に並べます。複数のニューロンは続けて並べます。
- 最初の層の入力は 2 個です。2 層目以降の入力の個数は、前の層のニューロン数になります。
- 出力層のニューロンは 1 個にしてください。

例 1: XOR(2 層。テストと同じ重み)

```bash
python src/main.py --gate xor \
  --layer 0.8 0.8 -0.5  -0.8 -0.8 0.9 \
  --layer 0.8 0.8 -1.2
```

| `--layer` | ニューロン | 重み | バイアス | 役割 |
| --- | --- | --- | --- | --- |
| 1 回目(隠れ層) | 1 個目 | `0.8 0.8` | `-0.5` | OR |
|  | 2 個目 | `-0.8 -0.8` | `0.9` | NAND |
| 2 回目(出力層) | 1 個目 | `0.8 0.8` | `-1.2` | AND |

例 2: 3 層(2 層目は入力をそのまま次へ渡すだけなので、結果は XOR のまま)

```bash
python src/main.py --gate xor \
  --layer 0.8 0.8 -0.5  -0.8 -0.8 0.9 \
  --layer 1 0 -0.5  0 1 -0.5 \
  --layer 0.8 0.8 -1.2
```

保存される画像は `result_<gate>_custom.png` です。背景の色が MLP の出力、点の色が正解(赤 = 0、青 = 1)です。
点と背景の色が一致していれば、その入力に対して正しく出力できています。

## 6. NumPy の基本(この課題で使う API)

NumPy は、数値の配列(ベクトル・行列)を扱うライブラリです。慣習として `np` という名前で読み込みます。

```python
import numpy as np
```

### 配列を作る・変換する

| API | 説明 | 例 |
| --- | --- | --- |
| `np.array(x)` | リストなどから新しい配列を作る(常にコピーする) | `np.array([1, 2, 3])` |
| `np.asarray(x)` | 配列に変換する。元から配列ならコピーせずそのまま返す | `np.asarray([1, 2])` |
| `dtype=float` | 要素の型を浮動小数点数にする(上の 2 つの引数) | `np.array([1, 2], dtype=float)` → `[1., 2.]` |
| `np.zeros(shape)` | 0 で埋めた配列を作る | `np.zeros(3)`、`np.zeros((2, 3))` |
| `a.reshape(shape)` | 要素の並びはそのままで形だけ変える。`-1` は「残りから自動で決める」 | `np.arange(6).reshape(-1, 3)` → 2 行 3 列 |
| `a.tolist()` | Python のリストに戻す(表示用) |  |

リストのリストから `np.array` を作ると 2 次元配列になります。 `np.array([a, b])`(`a`, `b` は同じ長さの 1
次元配列)とすると、`a` と `b` を行として積み重ねた行列になります。

### 形(shape)と次元

| API | 説明 | 例 |
| --- | --- | --- |
| `a.shape` | 各次元の長さのタプル | `np.zeros(3).shape` → `(3,)`、`np.zeros((2, 3)).shape` → `(2, 3)` |
| `np.ndim(a)` | 次元の数。スカラーは 0 | `np.ndim(5.0)` → `0` |
| `len(a)` | 最初の次元の長さ | `len(np.zeros((2, 3)))` → `2` |

`(3,)` は「長さ 3 の 1 次元配列(ベクトル)」、`(2, 3)` は「2 行 3 列の 2 次元配列(行列)」です。

### 要素の取り出し

```python
W = np.array([[1, 2, 3],
              [4, 5, 6]])
W[0]        # 0 行目 → array([1, 2, 3])
W[1, 2]     # 1 行目 2 列目 → 6
W[:, 0]     # すべての行の 0 列目 → array([1, 4])
W[:, :-1]   # 最後の列を除く → array([[1, 2], [4, 5]])
for row in W:   # 1 行ずつ取り出す
    print(row)
```

### 計算

| API | 説明 | 例 |
| --- | --- | --- |
| `a + b`、`a * b` | 同じ shape の配列どうしなら **要素ごと** に計算する | `np.array([1, 2]) * np.array([3, 4])` → `[3, 8]` |
| `a + 1` | スカラーとの計算は全要素に適用される | `np.array([1, 2]) + 1` → `[2, 3]` |
| `a > 0` | 比較も要素ごと。結果は `True` / `False` の配列 | `np.array([-1, 0, 2]) > 0` → `[False, False, True]` |
| `np.sum(a)` | 全要素の合計 | `np.sum([1, 2, 3])` → `6` |
| `np.dot(a, b)` | 1 次元配列どうしなら内積(要素ごとの積の合計) | `np.dot([1, 2], [3, 4])` → `1*3 + 2*4 = 11` |
| `np.where(cond, x, y)` | `cond` が `True` の要素は `x`、`False` の要素は `y` を選ぶ | `np.where([True, False], 10, 20)` → `[10, 20]` |
| `np.unique(a)` | 重複を除いた要素(テストで使用) | `np.unique([1, 0, 1])` → `[0, 1]` |

NumPy の関数にスカラーを渡すと、戻り値が「0 次元の配列」や「NumPy のスカラー型」になることがあります。 普通の Python の数値にしたいときは
`int(v)` や `float(v)` で変換します。

### 乱数

```python
rng = np.random.default_rng(0)   # シード 0 の乱数生成器を作る(省略するとシードも毎回変わる)
rng.normal(size=3)               # 標準正規分布(平均 0、標準偏差 1)の乱数を 3 個 → shape (3,)
rng.uniform()                    # [0, 1) の一様乱数を 1 個
```

- 同じシードで作った乱数生成器からは、**同じ順番で同じ値** が出てきます(実験の再現に使います)。
- 1 つの乱数生成器から続けて取り出すと、毎回違う値になります。 ニューロンごとに `np.random.default_rng(0)`
  を作り直すと、全ニューロンが同じ初期値になってしまうので注意してください。

### テストで使う比較関数

浮動小数点数は計算誤差があるため、`==` ではなく「ほぼ等しいか」で比べることがあります。

| API | 説明 |
| --- | --- |
| `np.testing.assert_array_equal(a, b)` | `a` と `b` の shape と全要素が **完全に** 一致しなければテスト失敗 |
| `np.testing.assert_allclose(a, b)` | `a` と `b` の全要素が **ほぼ** 一致しなければテスト失敗 |
| `np.allclose(a, b)` | 全要素がほぼ一致すれば `True`(テスト失敗にはせず、真偽値を返すだけ) |

## 7. pytest の基本

pytest は、Python のテストを自動で見つけて実行するツールです。

### テストの見つけ方

- `pytest` を実行すると、`test_*.py` という名前のファイルの中から、`test_` で始まる関数を探してすべて実行します。
- 関数の中で `assert` などが失敗しなければ **合格(PASSED)**、失敗したり例外が起きたりすれば **不合格(FAILED)** です。
- `conftest.py` は、テストの前に pytest が自動で読み込む設定ファイルです。 この課題では `src/` を import パスに追加し、テストから
  `import nn_core` できるようにしています。
- 各テスト関数の docstring(関数の先頭の `"""..."""`)に、そのテストの **条件** と **期待する結果** を書いてあります。
  テストが失敗したら、まずそこを読んでください。

### テストの書き方で使っている API

| API | 説明 |
| --- | --- |
| `assert 条件` | 条件が `False` ならテスト失敗。Python の標準の文 |
| `with pytest.raises(ValueError):` | この `with` の中で `ValueError` が **起きること** を確かめる。起きなければテスト失敗 |
| `pytest.approx(x)` | 浮動小数点数を「ほぼ等しいか」で比べる。例: `assert 0.1 + 0.2 == pytest.approx(0.3)` |
| `@pytest.mark.parametrize("引数名", [値, ...])` | 同じテスト関数を、値を変えながら何回も実行する |

`parametrize` の例です。

```python
@pytest.mark.parametrize("x, expected", [((0, 0), 1), ((1, 1), 0)])
def test_nand(x, expected):
    ...
```

これは `test_nand(x=(0, 0), expected=1)` と `test_nand(x=(1, 1), expected=0)` の 2
つのテストとして実行されます。 結果には `test_nand[x0-1]` のように、どの値で実行したかが表示されます。

### よく使う実行方法

```bash
pytest                                        # すべて実行(結果は . = 合格、F = 失敗 の 1 文字ずつ)
pytest -v                                     # テスト名ごとに PASSED / FAILED を表示
pytest tests/test_neuron.py                   # 1 つのファイルだけ
pytest tests/test_neuron.py::test_forward_uses_bias   # 1 つのテスト関数だけ
pytest -k nand                                # 名前に "nand" を含むテストだけ
pytest -x                                     # 最初の失敗で止める
```

### 失敗したときの読み方

失敗すると、どの行の `assert` が失敗したか、そのときの値が表示されます。

```
    def test_zero_gives_zero(x):
        """入力がちょうど 0 のときは 0 を返す(境界値)。

        条件: 整数の 0 と浮動小数点数の 0.0 を与える。
        期待: 戻り値が 0 である(1 ではない)。
        """
>       assert step(x) == 0
E       assert array(1) == 0
E        +  where array(1) = step(0)

tests/test_activation.py:39: AssertionError
```

`>` が失敗した行、`E` の行が詳細です。この例では「`step(0)` が 1(`array(1)` は 0 次元の NumPy 配列)を返したが、 0
であるべき」と読めます。docstring もあわせて表示されるので、何を確かめるテストかが分かります。

まだ実装していないメソッドを呼ぶテストは、`NotImplementedError` で失敗します。 実装を進めるにつれて、この失敗が減っていきます。

## 8. 評価基準

1. `pytest` の全テストに合格すること(`tests/` は変更しないこと)。
2. `Neuron.forward` が、ヒントの順序(重み付き和 → バイアスを加える → 活性化関数)どおりに計算していること。
3. `Layer` は `Neuron` を、`MLP` は `Layer` を使って組み立てていること(下の層の処理を上の層に書き直さないこと)。
4. 変数名が分かりやすく、読みやすいコードであること。

## 9. 発展課題(任意)

1. AND ・ OR ・ NOR をニューロン 1 個で実現する重みとバイアスを自分で考え、`main.py` の `--layer` で確かめてみよう。 (`--gate`
   は正解率の計算に使うだけなので、目標と違うゲートを指定すると正解率は下がります)
2. XOR をニューロン 1 個(`--layer` を 1 回だけ)で実現できる重みはあるか、試してから理由を考えよう。
3. 例 1(AND(OR, NAND))とは違う分解で XOR を作ってみよう。 (ヒント: XOR は「x1 だけが 1」または「x2 だけが 1」のときに 1 になる)
