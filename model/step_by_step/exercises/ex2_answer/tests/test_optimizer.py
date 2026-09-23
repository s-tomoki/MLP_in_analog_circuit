"""最適化手法(Optimizer)のテスト。

テスト対象: nn_core.Optimizer, nn_core.GradientDescent
"""

import numpy as np
import pytest
from nn_core import MLP, GradientDescent, Optimizer


def make_mlp_with_gradients():
    """重みと勾配を手で設定した MLP([2, 1]) を作る。

    重み (0.5, -0.3)、バイアス 0.1、grad_weights (0.2, -0.4)、grad_bias 0.5。
    """
    mlp = MLP([2, 1], seed=0)
    mlp.set_weights([([[0.5, -0.3]], [0.1])])
    neuron = mlp.layers[0].neurons[0]
    neuron.grad_weights = np.array([0.2, -0.4])
    neuron.grad_bias = 0.5
    return mlp


@pytest.mark.stage1
def test_optimizer_base_class_cannot_be_used_directly():
    """Optimizer は基底クラスで、step を実装しないと使えない。

    条件: Optimizer() と、step を実装していない派生クラスのインスタンスを作ろうとする。
    期待: どちらも TypeError が送出される(抽象メソッド step が未実装のため)。
    """

    class NoStep(Optimizer):
        pass

    with pytest.raises(TypeError):
        Optimizer()  # type: ignore[abstract]
    with pytest.raises(TypeError):
        NoStep()  # type: ignore[abstract]


@pytest.mark.stage1
def test_custom_optimizer_can_be_plugged_in():
    """Optimizer を継承したクラスなら、MLP を変えずに最適化手法を差し替えられる。

    条件: step が呼ばれた回数を数えるだけの最適化手法を作り、
        MLP.train で 4 サンプル × 3 エポック学習する。
    期待: step がちょうど 12 回(1 サンプルにつき 1 回)呼ばれる。
    """

    class CountingOptimizer(Optimizer):
        def __init__(self):
            self.calls = 0

        def step(self, mlp):
            self.calls += 1

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    Y = np.array([[0], [1], [1], [0]], dtype=float)
    optimizer = CountingOptimizer()
    MLP([2, 2, 1], seed=0).train(X, Y, epochs=3, optimizer=optimizer)
    assert optimizer.calls == 12


@pytest.mark.stage1
def test_zero_gradient_keeps_parameters():
    """勾配が 0 なら、GradientDescent.step でパラメータは変わらない。

    条件: 重み (0.5, -0.3)、バイアス 0.1 の MLP([2, 1]) の勾配を 0 にして、学習率 0.1 で step する。
    期待: 重みとバイアスが元のまま(許容誤差 abs=1e-12)。
    """
    mlp = make_mlp_with_gradients()
    neuron = mlp.layers[0].neurons[0]
    neuron.grad_weights = np.zeros(2)
    neuron.grad_bias = 0.0
    GradientDescent(0.1).step(mlp)
    assert neuron.weights == pytest.approx([0.5, -0.3], abs=1e-12)
    assert neuron.bias == pytest.approx(0.1, abs=1e-12)


@pytest.mark.stage1
def test_step_moves_parameters_against_gradient():
    """GradientDescent.step は、パラメータを「学習率 × 勾配」だけ勾配と逆向きに動かす。

    条件: make_mlp_with_gradients() の MLP を、学習率 0.1 で 1 回 step する。
    期待: 重み = (0.5 - 0.1×0.2, -0.3 - 0.1×(-0.4)) = (0.48, -0.26)、
        バイアス = 0.1 - 0.1×0.5 = 0.05(許容誤差 abs=1e-12)。
    """
    # ヒント:
    # - GradientDescent(0.1) を作って step(mlp) を 1 回呼ぶ
    # - 更新後の重み・バイアスは mlp.layers[0].neurons[0] から取り出せる
    mlp = make_mlp_with_gradients()
    GradientDescent(0.1).step(mlp)
    neuron = mlp.layers[0].neurons[0]
    assert neuron.weights == pytest.approx([0.48, -0.26], abs=1e-12)
    assert neuron.bias == pytest.approx(0.05, abs=1e-12)


@pytest.mark.stage1
def test_update_amount_is_proportional_to_learning_rate():
    """学習率を 2 倍にすると、1 回の更新での変化量も 2 倍になる。

    条件: 同じ重み・勾配の MLP を 2 つ用意し、学習率 0.1 と 0.2 でそれぞれ 1 回 step する。
    期待: (更新後の重み - 更新前の重み) が、学習率 0.2 の方でちょうど 2 倍
        (許容誤差 rel=1e-12, abs=1e-12)。バイアスも同様。
    """
    # ヒント:
    # - make_mlp_with_gradients() を 2 回呼んで、同じ状態の MLP を 2 つ作る
    # - 変化量 = 更新後 - 更新前。更新前の値はコピーして残しておく(配列は .copy())
    small = make_mlp_with_gradients()
    large = make_mlp_with_gradients()
    w0 = small.layers[0].neurons[0].weights.copy()
    b0 = small.layers[0].neurons[0].bias

    GradientDescent(0.1).step(small)
    GradientDescent(0.2).step(large)

    dw_small = small.layers[0].neurons[0].weights - w0
    dw_large = large.layers[0].neurons[0].weights - w0
    db_small = small.layers[0].neurons[0].bias - b0
    db_large = large.layers[0].neurons[0].bias - b0
    assert dw_large == pytest.approx(2 * dw_small, rel=1e-12, abs=1e-12)
    assert db_large == pytest.approx(2 * db_small, rel=1e-12, abs=1e-12)
