"""Neuron クラスのテスト。

テスト対象: nn_core.Neuron(__init__, set_weights, forward)
"""

import numpy as np
import pytest
from nn_core import Neuron


def test_initial_weights_shape():
    """生成直後の重みとバイアスの形が正しい。

    条件: 入力数 3 のニューロンを作る。
    期待: weights が shape (3,) の NumPy 配列で、bias がスカラー(0 次元)である。
    """
    neuron = Neuron(3, rng=np.random.default_rng(0))
    assert isinstance(neuron.weights, np.ndarray)
    assert neuron.weights.shape == (3,)
    assert np.ndim(neuron.bias) == 0


def test_same_rng_seed_gives_same_initial_weights():
    """同じシードの乱数生成器を渡すと、同じ初期値になる(再現性)。

    条件: シード 42 の乱数生成器をそれぞれ新しく作り、2 つのニューロンに渡す。
    期待: 2 つのニューロンの weights と bias が完全に一致する。
    """
    a = Neuron(4, rng=np.random.default_rng(42))
    b = Neuron(4, rng=np.random.default_rng(42))
    np.testing.assert_array_equal(a.weights, b.weights)
    assert a.bias == b.bias


def test_can_be_created_without_rng():
    """rng を省略しても生成できる。

    条件: rng を渡さずに入力数 2 のニューロンを作る。
    期待: エラーにならず、weights の shape が (2,) である。
    """
    neuron = Neuron(2)
    assert neuron.weights.shape == (2,)


def test_set_weights_replaces_values():
    """set_weights で重みとバイアスが置き換わる。

    条件: 入力数 2 のニューロンに、リストの重み [1.0, -2.0] とバイアス 0.5 を設定する。
    期待: weights が [1.0, -2.0]、bias が 0.5 になっている。
    """
    neuron = Neuron(2, rng=np.random.default_rng(0))
    neuron.set_weights([1.0, -2.0], 0.5)
    np.testing.assert_allclose(neuron.weights, [1.0, -2.0])
    assert neuron.bias == pytest.approx(0.5)


@pytest.mark.parametrize("bad_weights", [[1.0], [1.0, 2.0, 3.0]])
def test_set_weights_rejects_wrong_length(bad_weights):
    """重みの個数が入力数と合わないときは ValueError を送出する。

    条件: 入力数 2 のニューロンに、長さ 1 または 3 の重みを設定しようとする。
    期待: ValueError が送出される。
    """
    neuron = Neuron(2, rng=np.random.default_rng(0))
    with pytest.raises(ValueError):
        neuron.set_weights(bad_weights, 0.0)


@pytest.mark.parametrize(
    "inputs, expected",
    [
        ([1, 1], 1),  # 1 + 1 - 1.5 = 0.5  > 0
        ([1, 0], 0),  # 1 + 0 - 1.5 = -0.5 <= 0
        ([0, 0], 0),  # 0 + 0 - 1.5 = -1.5 <= 0
    ],
)
def test_forward_with_known_weights(inputs, expected):
    """既知の重みで、重み付き和 → バイアス → step の結果が出力される。

    条件: 重み [1.0, 1.0]、バイアス -1.5 を設定し、各入力を与える。
    期待: 重み付き和 + バイアスが正なら 1、0 以下なら 0 を返す(計算は各行のコメント参照)。
    """
    neuron = Neuron(2, rng=np.random.default_rng(0))
    neuron.set_weights([1.0, 1.0], -1.5)
    assert neuron.forward(inputs) == expected


def test_forward_uses_bias():
    """バイアスだけで出力が決まる場合も正しく計算する。

    条件: 重みを [0.0, 0.0] にして入力 [0, 0] を与え、バイアスを 0.1 / -0.1 と変える。
    期待: バイアス 0.1 のとき 1、-0.1 のとき 0 を返す。
    """
    neuron = Neuron(2, rng=np.random.default_rng(0))
    neuron.set_weights([0.0, 0.0], 0.1)
    assert neuron.forward([0, 0]) == 1
    neuron.set_weights([0.0, 0.0], -0.1)
    assert neuron.forward([0, 0]) == 0


def test_forward_gives_zero_when_sum_is_exactly_zero():
    """重み付き和 + バイアスがちょうど 0 のときは 0 を返す(境界値)。

    条件: 重み [1.0, -1.0]、バイアス 0.0 で入力 [1, 1] を与える(1 - 1 + 0 = 0)。
    期待: 0 を返す。
    """
    neuron = Neuron(2, rng=np.random.default_rng(0))
    neuron.set_weights([1.0, -1.0], 0.0)
    assert neuron.forward([1, 1]) == 0


def test_forward_accepts_list_and_ndarray():
    """入力はリストでも NumPy 配列でもよい。

    条件: 同じ値の入力を、リストと NumPy 配列の 2 通りで与える。
    期待: どちらでも同じ出力になる。
    """
    neuron = Neuron(3, rng=np.random.default_rng(0))
    neuron.set_weights([0.5, -1.0, 2.0], -0.2)
    assert neuron.forward([1, 0, 1]) == neuron.forward(np.array([1, 0, 1]))


@pytest.mark.parametrize(
    "inputs, expected",
    [([0, 0], 1), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)],
)
def test_single_neuron_nand(inputs, expected):
    """ニューロン 1 個で NAND ゲートが実現できる。

    条件: 重み (-0.5, -0.5)、バイアス 0.8 を設定し、真理値表の 4 通りの入力を与える。
    期待: NAND の真理値表どおり、(1, 1) のときだけ 0、それ以外は 1 を返す。
    """
    neuron = Neuron(2, rng=np.random.default_rng(0))
    neuron.set_weights([-0.5, -0.5], 0.8)
    assert neuron.forward(inputs) == expected
