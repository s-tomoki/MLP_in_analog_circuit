"""MLP クラスの構成・順伝播のテスト(ex1 からの引き継ぎ部分)。

テスト対象: nn_core.MLP(__init__, get_weights, neurons, forward, predict)
学習に関わるメソッドは test_backprop.py と test_training.py で確かめる。
"""

import numpy as np
import pytest
from nn_core import MLP, Layer


@pytest.mark.stage1
@pytest.mark.parametrize("layer_sizes", [[], [2]])
def test_rejects_too_few_layer_sizes(layer_sizes):
    """layer_sizes の要素数が 2 未満なら ValueError を送出する。

    条件: 空のリスト、または要素 1 個のリストで MLP を作ろうとする。
    期待: ValueError が送出される。
    """
    with pytest.raises(ValueError):
        MLP(layer_sizes)


@pytest.mark.stage1
def test_structure_and_activations():
    """layer_sizes と活性化関数の指定どおりに層が構成される。

    条件:
        - MLP([2, 3, 4, 1], activation="relu") を作る。
        - MLP([2, 3, 1], activation="relu", output_activation="sigmoid") を作る。
    期待:
        - 1 つ目: 層が 3 つ(すべて Layer)、ニューロン数 [3, 4, 1]、(W, b) の shape が
          ((3, 2), (3,)), ((4, 3), (4,)), ((1, 4), (1,))、全層の活性化関数が relu。
        - 2 つ目: 隠れ層は relu、出力層だけ sigmoid。
    """
    mlp = MLP([2, 3, 4, 1], seed=0, activation="relu")
    assert len(mlp.layers) == 3
    assert all(isinstance(layer, Layer) for layer in mlp.layers)
    assert [len(layer.neurons) for layer in mlp.layers] == [3, 4, 1]
    shapes = [(W.shape, b.shape) for W, b in mlp.get_weights()]
    assert shapes == [((3, 2), (3,)), ((4, 3), (4,)), ((1, 4), (1,))]
    assert mlp.activations == ["relu", "relu", "relu"]

    mlp = MLP([2, 3, 1], seed=0, activation="relu", output_activation="sigmoid")
    assert mlp.activations == ["relu", "sigmoid"]
    assert mlp.layers[0].neurons[0].activation == "relu"
    assert mlp.layers[1].neurons[0].activation == "sigmoid"


@pytest.mark.stage1
def test_same_seed_gives_same_initial_weights():
    """同じ seed なら同じ初期値、違う seed なら違う初期値になる。

    条件: seed=123 で MLP([2, 3, 1]) を 2 つ、seed=124 で 1 つ作る。
    期待: seed=123 どうしは全層の W, b が完全に一致し、seed=124 とは一致しない。
    """
    a = MLP([2, 3, 1], seed=123).get_weights()
    b = MLP([2, 3, 1], seed=123).get_weights()
    c = MLP([2, 3, 1], seed=124).get_weights()
    for (Wa, ba), (Wb, bb) in zip(a, b):
        np.testing.assert_array_equal(Wa, Wb)
        np.testing.assert_array_equal(ba, bb)
    assert not np.allclose(a[0][0], c[0][0])


@pytest.mark.stage1
def test_forward_and_predict_shapes():
    """forward は 1 サンプル、predict は複数サンプルをまとめて計算する。

    条件: MLP([2, 3, 2]) に、1 サンプル (0.0, 1.0) と、
        4 サンプルを並べた shape (4, 2) の配列を与える。
    期待: forward は shape (2,)、predict は shape (4, 2) の NumPy 配列を返し、
        predict の各行は、その行を forward した結果と一致する。
    """
    mlp = MLP([2, 3, 2], seed=0)
    out = mlp.forward([0.0, 1.0])
    assert isinstance(out, np.ndarray)
    assert out.shape == (2,)

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    pred = mlp.predict(X)
    assert pred.shape == (4, 2)
    for x, row in zip(X, pred):
        np.testing.assert_allclose(row, mlp.forward(x))


@pytest.mark.stage1
def test_neurons_iterates_all_neurons_in_order():
    """neurons() は、入力に近い層から順に全ニューロンを返す。

    条件: MLP([2, 3, 2]) の neurons() をリストにする。
    期待: 長さ 5 で、先頭 3 個が 1 層目、残り 2 個が 2 層目のニューロン(同じオブジェクト)。
    """
    mlp = MLP([2, 3, 2], seed=0)
    neurons = list(mlp.neurons())
    assert len(neurons) == 5
    assert neurons[:3] == mlp.layers[0].neurons
    assert neurons[3:] == mlp.layers[1].neurons
