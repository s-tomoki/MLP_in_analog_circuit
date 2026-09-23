"""Layer クラスのテスト。

テスト対象: nn_core.Layer(__init__, set_weights, get_weights, forward)
"""

import numpy as np
import pytest
from nn_core import Layer, Neuron


def test_structure():
    """指定した個数・入力数のニューロンで層が構成される。

    条件: ニューロン数 3、入力数 2 の層を作る。
    期待: neurons が長さ 3 のリストで、各要素が Neuron であり、重みの shape が (2,) である。
    """
    layer = Layer(3, 2, rng=np.random.default_rng(0))
    assert len(layer.neurons) == 3
    for neuron in layer.neurons:
        assert isinstance(neuron, Neuron)
        assert neuron.weights.shape == (2,)


def test_set_and_get_weights_roundtrip():
    """set_weights で設定した値が get_weights でそのまま取り出せる。

    条件: ニューロン数 3、入力数 2 の層に、shape (3, 2) の W と shape (3,) の b を設定する。
    期待: get_weights が返す (W, b) の shape と値が、設定したものと一致する。
    """
    layer = Layer(3, 2, rng=np.random.default_rng(0))
    W = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    b = np.array([-0.1, 0.0, 0.1])
    layer.set_weights(W, b)

    W_got, b_got = layer.get_weights()
    assert W_got.shape == (3, 2)
    assert b_got.shape == (3,)
    np.testing.assert_allclose(W_got, W)
    np.testing.assert_allclose(b_got, b)


def test_set_weights_assigns_each_row_to_each_neuron():
    """W の i 行目と b の i 番目が、i 番目のニューロンに設定される。

    条件: ニューロン数 2、入力数 3 の層に、リストの W と b を設定する。
    期待: neurons[0] に W の 0 行目と b[0]、neurons[1] に W の 1 行目と b[1] が入っている。
    """
    layer = Layer(2, 3, rng=np.random.default_rng(0))
    W = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    b = [7.0, 8.0]
    layer.set_weights(W, b)
    np.testing.assert_allclose(layer.neurons[0].weights, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(layer.neurons[1].weights, [4.0, 5.0, 6.0])
    assert layer.neurons[0].bias == pytest.approx(7.0)
    assert layer.neurons[1].bias == pytest.approx(8.0)


@pytest.mark.parametrize(
    "W, b",
    [
        (np.zeros((2, 2)), np.zeros(3)),  # ニューロン数が足りない
        (np.zeros((3, 3)), np.zeros(3)),  # 入力数が合わない
        (np.zeros((3, 2)), np.zeros(2)),  # バイアスの個数が合わない
    ],
)
def test_set_weights_rejects_wrong_shape(W, b):
    """W または b の shape が層の構成と合わないときは ValueError を送出する。

    条件: ニューロン数 3、入力数 2 の層に、shape の誤った W / b を設定しようとする
        (どこが誤っているかは各行のコメント参照)。
    期待: ValueError が送出される。
    """
    layer = Layer(3, 2, rng=np.random.default_rng(0))
    with pytest.raises(ValueError):
        layer.set_weights(W, b)


def test_forward_returns_one_output_per_neuron():
    """順伝播の出力は、ニューロン 1 個につき 1 つの 0 / 1 を並べた配列になる。

    条件: ニューロン数 4、入力数 2 の層(重みは乱数のまま)に入力 [0, 1] を与える。
    期待: 戻り値が shape (4,) の NumPy 配列で、要素はすべて 0 または 1 である。
    """
    layer = Layer(4, 2, rng=np.random.default_rng(0))
    out = layer.forward([0, 1])
    assert isinstance(out, np.ndarray)
    assert out.shape == (4,)
    assert set(np.unique(out)) <= {0, 1}


@pytest.mark.parametrize(
    "inputs, expected",
    [
        ([0, 0], [0, 1]),
        ([0, 1], [1, 1]),
        ([1, 0], [1, 1]),
        ([1, 1], [1, 0]),
    ],
)
def test_forward_with_known_weights(inputs, expected):
    """既知の重みで、各ニューロンの出力が正しい順に並ぶ。

    条件: 1 番目のニューロンが OR、2 番目が NAND として働く重みを設定し、
        真理値表の 4 通りの入力を与える。
    期待: 出力が [OR の結果, NAND の結果] になる。
    """
    layer = Layer(2, 2, rng=np.random.default_rng(0))
    layer.set_weights([[0.8, 0.8], [-0.8, -0.8]], [-0.5, 0.9])
    np.testing.assert_array_equal(layer.forward(inputs), expected)
