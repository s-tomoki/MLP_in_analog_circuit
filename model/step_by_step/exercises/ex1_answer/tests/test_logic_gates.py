"""重みを手動で設定した MLP が論理ゲートとして動作するかのテスト。

テスト対象: nn_core.MLP を使った NAND / XOR ゲート

使用する重み:
    NAND: MLP([2, 1])
        出力層: 重み (-0.5, -0.5), バイアス 0.8
    XOR: MLP([2, 2, 1])
        隠れ層: 重み (0.8, 0.8), (-0.8, -0.8), バイアス -0.5, 0.9  (OR と NAND)
        出力層: 重み (0.8, 0.8), バイアス -1.2                      (AND)
"""

import numpy as np
import pytest
from nn_core import MLP

NAND_PARAMS = [
    (np.array([[-0.5, -0.5]]), np.array([0.8])),
]

XOR_PARAMS = [
    (np.array([[0.8, 0.8], [-0.8, -0.8]]), np.array([-0.5, 0.9])),  # 隠れ層
    (np.array([[0.8, 0.8]]), np.array([-1.2])),  # 出力層
]

NAND_TABLE = [((0, 0), 1), ((0, 1), 1), ((1, 0), 1), ((1, 1), 0)]
XOR_TABLE = [((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 0)]


@pytest.mark.parametrize("x, expected", NAND_TABLE)
def test_nand(x, expected):
    """1 層の MLP が NAND ゲートとして動作する。

    条件: MLP([2, 1]) に NAND の重みを設定し、真理値表の 4 通りの入力を与える。
    期待: 出力が shape (1,) で、値が NAND の真理値表と一致する。
    """
    mlp = MLP([2, 1])
    mlp.set_weights(NAND_PARAMS)
    out = mlp.forward(np.array(x))
    assert out.shape == (1,)
    assert out[0] == expected


@pytest.mark.parametrize("x, expected", XOR_TABLE)
def test_xor(x, expected):
    """隠れ層を持つ MLP が XOR ゲートとして動作する。

    条件: MLP([2, 2, 1]) に XOR の重みを設定し、真理値表の 4 通りの入力を与える。
    期待: 出力が shape (1,) で、値が XOR の真理値表と一致する。
    """
    mlp = MLP([2, 2, 1])
    mlp.set_weights(XOR_PARAMS)
    out = mlp.forward(np.array(x))
    assert out.shape == (1,)
    assert out[0] == expected


@pytest.mark.parametrize(
    "x, expected_hidden",
    [((0, 0), [0, 1]), ((0, 1), [1, 1]), ((1, 0), [1, 1]), ((1, 1), [1, 0])],
)
def test_xor_hidden_layer_computes_or_and_nand(x, expected_hidden):
    """XOR の隠れ層が OR と NAND を計算している。

    条件: MLP([2, 2, 1]) に XOR の重みを設定し、隠れ層(layers[0])だけに入力を与える。
    期待: 隠れ層の出力が [OR の結果, NAND の結果] になる
        (XOR(x1, x2) = AND(OR(x1, x2), NAND(x1, x2)) という分解の確認)。
    """
    mlp = MLP([2, 2, 1])
    mlp.set_weights(XOR_PARAMS)
    np.testing.assert_array_equal(mlp.layers[0].forward(np.array(x)), expected_hidden)
