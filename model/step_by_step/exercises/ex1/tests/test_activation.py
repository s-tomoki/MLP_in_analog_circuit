"""step 関数のテスト。

テスト対象: nn_core.step
"""

import numpy as np
import pytest
from nn_core import step


@pytest.mark.parametrize("x", [0.3, 1, 5.0, 1e-9])
def test_positive_input_gives_one(x):
    """正の入力に対して 1 を返す。

    条件: 正のスカラー(ごく小さい正の値 1e-9 を含む)を与える。
    期待: 戻り値が 1 である。
    """
    assert step(x) == 1


@pytest.mark.parametrize("x", [-0.3, -1, -5.0, -1e-9])
def test_negative_input_gives_zero(x):
    """負の入力に対して 0 を返す。

    条件: 負のスカラー(ごく小さい負の値 -1e-9 を含む)を与える。
    期待: 戻り値が 0 である。
    """
    assert step(x) == 0


@pytest.mark.parametrize("x", [0, 0.0])
def test_zero_gives_zero(x):
    """入力がちょうど 0 のときは 0 を返す(境界値)。

    条件: 整数の 0 と浮動小数点数の 0.0 を与える。
    期待: 戻り値が 0 である(1 ではない)。
    """
    assert step(x) == 0


def test_1d_array_is_applied_elementwise():
    """1 次元配列を与えると、要素ごとに適用した配列を返す。

    条件: 負・0・正の値が混ざった shape (5,) の配列を与える。
    期待: 戻り値の shape が (5,) で、各要素が 0 / 1 に正しく変換されている。
    """
    x = np.array([-2.0, -0.1, 0.0, 0.1, 2.0])
    out = step(x)
    assert np.shape(out) == x.shape
    np.testing.assert_array_equal(out, [0, 0, 0, 1, 1])


def test_2d_array_keeps_shape():
    """2 次元配列を与えても shape を保ったまま要素ごとに適用する。

    条件: shape (2, 3) の配列を与える。
    期待: 戻り値の shape が (2, 3) で、各要素が 0 / 1 に正しく変換されている。
    """
    x = np.array([[1.0, -1.0, 0.0], [-0.5, 0.5, 3.0]])
    out = step(x)
    assert np.shape(out) == x.shape
    np.testing.assert_array_equal(out, [[1, 0, 0], [0, 1, 1]])
