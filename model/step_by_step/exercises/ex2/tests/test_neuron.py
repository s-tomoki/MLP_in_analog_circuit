"""Neuron クラスのテスト。

テスト対象: nn_core.Neuron(forward, backward)

手計算で使うニューロン(このファイルの多くのテストで共通):
    重み (0.5, -0.3)、バイアス 0.1、入力 (1.0, 2.0)
    → u = 0.5×1.0 + (-0.3)×2.0 + 0.1 = 0.0
"""

import numpy as np
import pytest
from nn_core import Neuron, sigmoid


def make_neuron(activation="sigmoid"):
    """手計算用のニューロン(重み (0.5, -0.3)、バイアス 0.1)を作る。"""
    neuron = Neuron(2, rng=np.random.default_rng(0), activation=activation)
    neuron.set_weights([0.5, -0.3], 0.1)
    return neuron


@pytest.mark.stage1
def test_forward_stores_intermediate_values():
    """forward が、逆伝播で使う値(入力・u・出力)を保存する。

    条件: 重み (0.4, 0.2)、バイアス -0.1 のシグモイドニューロンに入力 (1.0, 0.5) を与える。
    期待: inputs = (1.0, 0.5)、u = 0.4、output = σ(0.4) で、戻り値も σ(0.4)
        (許容誤差 abs=1e-12)。
    """
    neuron = Neuron(2, rng=np.random.default_rng(0))
    neuron.set_weights([0.4, 0.2], -0.1)
    out = neuron.forward([1.0, 0.5])
    np.testing.assert_allclose(neuron.inputs, [1.0, 0.5])
    assert neuron.u == pytest.approx(0.4, abs=1e-12)
    assert neuron.output == pytest.approx(float(sigmoid(0.4)), abs=1e-12)
    assert out == pytest.approx(float(sigmoid(0.4)), abs=1e-12)


@pytest.mark.stage1
def test_unknown_activation_raises():
    """未知の活性化関数の名前を指定すると ValueError を送出する。

    条件: activation="tanh" で Neuron を作ろうとする。
    期待: ValueError が送出される。
    """
    with pytest.raises(ValueError):
        Neuron(2, activation="tanh")


@pytest.mark.stage1
def test_initial_gradients_are_zero():
    """生成直後の勾配は 0 で初期化されている。

    条件: 入力数 3 のニューロンを作る。
    期待: grad_weights が shape (3,) のゼロ配列、grad_bias が 0。
    """
    neuron = Neuron(3, rng=np.random.default_rng(0))
    np.testing.assert_array_equal(neuron.grad_weights, np.zeros(3))
    assert neuron.grad_bias == 0.0


@pytest.mark.stage1
def test_backward_gradients_match_hand_calculation():
    """backward で求めた勾配が手計算と一致する。

    条件: 手計算用ニューロン(u = 0)に forward した後、error = 0.8 で backward する。
    期待: σ'(0) = 0.25 なので δ = 0.8 × 0.25 = 0.2。
        grad_weights = δ × 入力 = (0.2, 0.4)、grad_bias = δ = 0.2(許容誤差 abs=1e-12)。
    """
    # ヒント:
    # - make_neuron() でニューロンを作り、forward([1.0, 2.0]) してから backward(0.8) を呼ぶ
    # - grad_weights と grad_bias を、docstring の手計算の値と pytest.approx で比べる
    pass


@pytest.mark.stage1
def test_backward_returns_error_for_inputs():
    """backward の戻り値は、損失を各入力で偏微分した値 δ × 重み になる。

    条件: 手計算用ニューロンに forward した後、error = 0.8 で backward する(δ = 0.2)。
    期待: 戻り値が δ × 重み = (0.1, -0.06)(許容誤差 abs=1e-12)。
    """
    # ヒント:
    # - backward の戻り値を変数に受け取って、手計算の値と比べる
    # - 戻り値は「前の層のニューロンの出力」に対する勾配になる(入力に対する勾配ではなく重みを掛ける)
    pass


@pytest.mark.stage1
def test_backward_does_not_change_weights():
    """backward は勾配を求めるだけで、重み・バイアスは変更しない。

    条件: 手計算用ニューロンに forward → backward(1.0) を行う。
    期待: weights が (0.5, -0.3)、bias が 0.1 のまま(更新は Optimizer の役目)。
    """
    # ヒント:
    # - backward の前後で weights と bias を比べる
    # - 「値が同じ」ことを確かめるので、np.testing.assert_array_equal が使える
    pass


@pytest.mark.stage3
def test_backward_relu_inactive_gives_zero_gradient():
    """ReLU で u <= 0 のとき、勾配はすべて 0 になる。

    条件: 手計算用ニューロンを ReLU にし、入力 (0.0, 1.0)(u = -0.2)で forward → backward(1.0)。
    期待: grad_weights = (0, 0)、grad_bias = 0、戻り値も (0, 0)。
    """
    # ヒント:
    # - make_neuron("relu") で ReLU のニューロンを作る
    # - u <= 0 なら ReLU の導関数は 0 なので、δ も 0 になるはず
    pass


@pytest.mark.stage4
def test_backward_step_uses_sigmoid_derivative():
    """ステップ関数のニューロンは、出力は 0 / 1 だが、勾配はシグモイドの導関数で計算する(STE)。

    条件: 手計算用ニューロンを step にし、入力 (1.0, 1.0)(u = 0.3)で forward → backward(1.0)。
    期待: 出力は 1.0。δ = σ'(0.3) なので grad_weights = (σ'(0.3), σ'(0.3))、grad_bias = σ'(0.3)
        (許容誤差 rel=1e-12)。
    """
    # ヒント:
    # - make_neuron("step") で step のニューロンを作る
    # - 期待値は sigmoid_derivative(0.3) を使って計算する
    pass
