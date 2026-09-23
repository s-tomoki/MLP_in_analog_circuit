"""活性化関数とその導関数のテスト。

テスト対象: nn_core.activation
"""

import numpy as np
import pytest
from nn_core import (
    get_activation,
    relu,
    relu_derivative,
    sigmoid,
    sigmoid_derivative,
    step,
    step_derivative,
)


@pytest.mark.stage1
def test_sigmoid_values():
    """シグモイド関数の代表的な値と、配列入力での shape。

    条件: u = 0, ±2, ±50 と、それらを並べた配列を与える。
    期待:
        - σ(0) = 0.5、σ(2) ≈ 0.880797、σ(-2) ≈ 0.119203、σ(50) ≈ 1、σ(-50) ≈ 0
          (許容誤差 abs=1e-6)。
        - 配列を与えると同じ shape の配列が返る。
    """
    assert sigmoid(0.0) == pytest.approx(0.5, abs=1e-6)
    assert sigmoid(2.0) == pytest.approx(0.880797, abs=1e-6)
    assert sigmoid(-2.0) == pytest.approx(0.119203, abs=1e-6)
    assert sigmoid(50.0) == pytest.approx(1.0, abs=1e-6)
    assert sigmoid(-50.0) == pytest.approx(0.0, abs=1e-6)

    u = np.array([[0.0, 2.0, -2.0], [50.0, -50.0, 0.0]])
    out = sigmoid(u)
    assert np.shape(out) == u.shape
    np.testing.assert_allclose(out[0], [0.5, 0.880797, 0.119203], atol=1e-6)


@pytest.mark.stage1
def test_sigmoid_does_not_overflow():
    """極端に大きな |u| でもオーバーフローしない。

    条件: NumPy のオーバーフローを例外にする設定で、u = ±1000 を与える。
    期待: 例外が起きず、σ(1000) ≈ 1、σ(-1000) ≈ 0(許容誤差 abs=1e-12)。
    """
    with np.errstate(over="raise"):
        assert sigmoid(1000.0) == pytest.approx(1.0, abs=1e-12)
        assert sigmoid(-1000.0) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.stage1
@pytest.mark.parametrize("u", [-3.0, -0.5, 0.0, 0.5, 3.0])
def test_sigmoid_derivative_matches_numerical_derivative(u):
    """sigmoid_derivative が、シグモイド関数の数値微分と一致する。

    条件: いくつかの u で、中心差分 (σ(u+h) - σ(u-h)) / (2h)(h = 1e-6)を計算する。
    期待: sigmoid_derivative(u) と一致する(許容誤差 rel=1e-6, abs=1e-9)。
    """
    # ヒント:
    # - h を小さな値(1e-6)にして、σ(u + h) と σ(u - h) の差を 2h で割ると、u での傾きの近似になる
    # - pytest.approx の rel(相対誤差)と abs(絶対誤差)を指定して比べる
    h = 1e-6
    numerical = (sigmoid(u + h) - sigmoid(u - h)) / (2 * h)
    assert sigmoid_derivative(u) == pytest.approx(numerical, rel=1e-6, abs=1e-9)


@pytest.mark.stage3
def test_relu_values():
    """ReLU 関数の値。

    条件: 負・0・正の値を並べた配列を与える。
    期待: 負と 0 は 0、正はそのままの値になる。shape は変わらない。
    """
    u = np.array([-2.0, -0.1, 0.0, 0.1, 3.0])
    out = relu(u)
    assert np.shape(out) == u.shape
    np.testing.assert_allclose(out, [0.0, 0.0, 0.0, 0.1, 3.0])
    assert relu(-1.0) == 0.0
    assert relu(2.5) == pytest.approx(2.5)


@pytest.mark.stage3
def test_relu_derivative():
    """ReLU 関数の導関数。

    条件: 負・0・正の値を並べた配列と、スカラーを与える。
    期待: u > 0 では 1、u <= 0(0 ちょうどを含む)では 0。配列の shape は変わらない。
    """
    # ヒント:
    # - u = 0 では本来微分できないが、この課題では 0 とする約束(README 参照)
    # - 配列を 1 つ作って relu_derivative に渡し、期待する 0 / 1 の並びと比べる
    # - スカラーでも確かめる
    u = np.array([-2.0, -0.1, 0.0, 0.1, 3.0])
    out = relu_derivative(u)
    assert np.shape(out) == u.shape
    np.testing.assert_array_equal(out, [0.0, 0.0, 0.0, 1.0, 1.0])
    assert relu_derivative(5.0) == 1.0
    assert relu_derivative(-5.0) == 0.0


@pytest.mark.stage4
def test_step_values():
    """ステップ関数の値(ex1 と同じ仕様)。

    条件: 負・0・正の値を並べた配列を与える。
    期待: u > 0 では 1、u <= 0 では 0。shape は変わらない。
    """
    u = np.array([-2.0, 0.0, 1e-9, 3.0])
    out = step(u)
    assert np.shape(out) == u.shape
    np.testing.assert_array_equal(out, [0.0, 0.0, 1.0, 1.0])


@pytest.mark.stage4
def test_step_derivative_is_sigmoid_derivative():
    """ステップ関数の導関数として、シグモイド関数の導関数を代用している(STE)。

    条件: いくつかの u を並べた配列を与える。
    期待: step_derivative(u) が sigmoid_derivative(u) と一致する(許容誤差 rel=1e-12)。
        特に、u = 0 以外でも 0 にならない(本当のステップ関数の導関数とは違う)。
    """
    # ヒント:
    # - いくつかの u(負・0・正)で step_derivative と sigmoid_derivative を計算して比べる
    # - 「0 ではない」ことも assert で確かめると、STE の意味がはっきりする
    u = np.array([-3.0, -0.5, 0.0, 0.5, 3.0])
    np.testing.assert_allclose(step_derivative(u), sigmoid_derivative(u), rtol=1e-12)
    assert np.all(step_derivative(u) > 0)


@pytest.mark.stage1
@pytest.mark.parametrize(
    "name, expected",
    [
        ("sigmoid", (sigmoid, sigmoid_derivative)),
        ("relu", (relu, relu_derivative)),
        ("step", (step, step_derivative)),
    ],
)
def test_get_activation_returns_function_pairs(name, expected):
    """名前から (活性化関数, 導関数) の組が取り出せる。

    条件: "sigmoid"、"relu"、"step" を指定する。
    期待: 対応する関数の組が返る(step の導関数は STE 用の step_derivative)。
    """
    assert get_activation(name) == expected


@pytest.mark.stage1
def test_get_activation_rejects_unknown_name():
    """未知の名前を指定すると ValueError を送出する。

    条件: "tanh" を指定する(この課題では用意していない)。
    期待: ValueError が送出される。
    """
    with pytest.raises(ValueError):
        get_activation("tanh")
