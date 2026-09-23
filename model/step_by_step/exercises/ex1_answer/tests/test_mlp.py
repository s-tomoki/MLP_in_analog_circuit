"""MLP クラスのテスト。

テスト対象: nn_core.MLP(__init__, set_weights, get_weights, forward)
"""

import numpy as np
import pytest
from nn_core import MLP, Layer


@pytest.mark.parametrize("layer_sizes", [[], [2]])
def test_rejects_too_few_layer_sizes(layer_sizes):
    """layer_sizes の要素数が 2 未満なら ValueError を送出する。

    条件: 空のリスト、または要素 1 個のリストで MLP を作ろうとする。
    期待: ValueError が送出される。
    """
    with pytest.raises(ValueError):
        MLP(layer_sizes)


def test_structure():
    """layer_sizes のとおりに層が構成される。

    条件: layer_sizes = [2, 3, 4, 1] で MLP を作る。
    期待:
        - layers が長さ 3 のリストで、各要素が Layer である(入力層は含まない)。
        - 各層のニューロン数が [3, 4, 1] である。
        - 各層の (W, b) の shape が ((3, 2), (3,)), ((4, 3), (4,)), ((1, 4), (1,)) である
          (各層の入力数 = 前の層のサイズ)。
    """
    mlp = MLP([2, 3, 4, 1], seed=0)
    assert len(mlp.layers) == 3
    assert all(isinstance(layer, Layer) for layer in mlp.layers)
    assert [len(layer.neurons) for layer in mlp.layers] == [3, 4, 1]

    shapes = [(W.shape, b.shape) for W, b in mlp.get_weights()]
    assert shapes == [((3, 2), (3,)), ((4, 3), (4,)), ((1, 4), (1,))]


def test_same_seed_gives_same_initial_weights():
    """同じ seed を指定すると、同じ初期値になる(再現性)。

    条件: seed=123 で同じ構成の MLP を 2 つ作る。
    期待: すべての層の W と b が完全に一致する。
    """
    a = MLP([2, 3, 1], seed=123).get_weights()
    b = MLP([2, 3, 1], seed=123).get_weights()
    for (Wa, ba), (Wb, bb) in zip(a, b):
        np.testing.assert_array_equal(Wa, Wb)
        np.testing.assert_array_equal(ba, bb)


def test_neurons_are_not_all_identical():
    """ニューロンごとに異なる初期値が入る。

    条件: seed=0 で MLP([2, 3, 1]) を作り、1 層目の 0 番目と 1 番目のニューロンの重みを比べる。
    期待: 2 つの重みが一致しない
        (ニューロンごとに同じシードの乱数生成器を作り直していると失敗する)。
    """
    W, _ = MLP([2, 3, 1], seed=0).get_weights()[0]
    assert not np.allclose(W[0], W[1])


def test_set_and_get_weights_roundtrip():
    """set_weights で設定した値が get_weights でそのまま取り出せる。

    条件: MLP([2, 3, 1]) に、2 層分の (W, b) を設定する。
    期待: get_weights が返す各層の (W, b) が、設定したものと一致する。
    """
    mlp = MLP([2, 3, 1], seed=0)
    params = [
        (np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]), np.array([0.7, 0.8, 0.9])),
        (np.array([[-1.0, -2.0, -3.0]]), np.array([-4.0])),
    ]
    mlp.set_weights(params)
    for (W, b), (W_got, b_got) in zip(params, mlp.get_weights()):
        np.testing.assert_allclose(W_got, W)
        np.testing.assert_allclose(b_got, b)


def test_set_weights_rejects_wrong_number_of_layers():
    """params の層数が MLP の層数と合わないときは ValueError を送出する。

    条件: 2 層ある MLP([2, 2, 1]) に、1 層分だけの params を設定しようとする。
    期待: ValueError が送出される。
    """
    mlp = MLP([2, 2, 1], seed=0)
    with pytest.raises(ValueError):
        mlp.set_weights([(np.zeros((2, 2)), np.zeros(2))])


def test_set_weights_rejects_wrong_shape():
    """いずれかの層で W の shape が合わないときは ValueError を送出する。

    条件: MLP([2, 2, 1]) の 2 層目に、shape (1, 2) であるべき W として shape (1, 3) を渡す。
    期待: ValueError が送出される。
    """
    mlp = MLP([2, 2, 1], seed=0)
    with pytest.raises(ValueError):
        mlp.set_weights([(np.zeros((2, 2)), np.zeros(2)), (np.zeros((1, 3)), np.zeros(1))])


@pytest.mark.parametrize("layer_sizes", [[2, 1], [2, 3, 1], [2, 3, 2], [3, 4, 5, 2]])
def test_forward_output_shape(layer_sizes):
    """順伝播の出力の shape は、出力層のニューロン数に一致する。

    条件: さまざまな layer_sizes の MLP に、長さ layer_sizes[0] のゼロ配列を入力する。
    期待: 戻り値が shape (layer_sizes[-1],) の NumPy 配列である。
    """
    mlp = MLP(layer_sizes, seed=0)
    out = mlp.forward(np.zeros(layer_sizes[0]))
    assert isinstance(out, np.ndarray)
    assert out.shape == (layer_sizes[-1],)


@pytest.mark.parametrize("x, expected", [([0], [1]), ([1], [0])])
def test_forward_feeds_each_layer_output_into_next_layer(x, expected):
    """ある層の出力が、次の層の入力として使われる。

    条件: MLP([1, 1, 1]) の 1 層目を「入力をそのまま通す」、2 層目を「反転する(NOT)」重みにし、
        0 と 1 を入力する。
    期待: 入力 0 なら [1]、入力 1 なら [0] を返す
        (1 層目の出力を 2 層目に渡していないと、この結果にならない)。
    """
    mlp = MLP([1, 1, 1], seed=0)
    mlp.set_weights([([[1.0]], [-0.5]), ([[-1.0]], [0.5])])
    np.testing.assert_array_equal(mlp.forward(x), expected)
