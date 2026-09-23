"""誤差逆伝播法で求めた勾配の正しさのテスト。

テスト対象: nn_core.MLP(compute_gradients, train_step)

勾配チェック: 誤差逆伝播法で求めた勾配(解析的な勾配)を、目的関数を実際に少しだけ
動かして測った傾き(数値微分)と比べる。両者が一致すれば、逆伝播の実装が正しいと判断できる。
"""

import numpy as np
import pytest
from nn_core import MLP, GradientDescent, squared_error

EPS = 1e-6  # 数値微分でパラメータをずらす幅
GRAD_REL = 1e-5  # 勾配チェックの許容誤差(相対)
GRAD_ABS = 1e-8  # 勾配チェックの許容誤差(絶対)

SAMPLES = [
    (np.array([0.0, 1.0]), np.array([1.0])),
    (np.array([1.0, 1.0]), np.array([0.0])),
    (np.array([0.3, -0.8]), np.array([0.5])),
]


def objective(mlp, x, y, l2_lambda=0.0):
    """目的関数 J = L + (l2_lambda / 2) × (全ニューロンの重みの二乗和) を、順伝播だけで計算する。"""
    loss = squared_error(mlp.forward(x), y)
    penalty = sum(np.sum(neuron.weights**2) for neuron in mlp.neurons())
    return loss + 0.5 * l2_lambda * penalty


def numerical_gradients(mlp, x, y, l2_lambda=0.0):
    """全ニューロンについて、目的関数の重み・バイアスに関する勾配を中心差分で求める。

    Returns:
        ニューロンごとの (重みの勾配, バイアスの勾配) のリスト。順番は mlp.neurons() と同じ。
    """
    # ヒント:
    # - 各ニューロンの weights[i] を +EPS / -EPS だけずらして objective を計算し、差を 2×EPS で割る
    # - ずらした値は、計算が終わったら必ず元に戻す
    # - バイアスについても同じことをする
    grads = []
    for neuron in mlp.neurons():
        grad_w = np.zeros_like(neuron.weights)
        for i in range(len(neuron.weights)):
            original = neuron.weights[i]
            neuron.weights[i] = original + EPS
            plus = objective(mlp, x, y, l2_lambda)
            neuron.weights[i] = original - EPS
            minus = objective(mlp, x, y, l2_lambda)
            neuron.weights[i] = original
            grad_w[i] = (plus - minus) / (2 * EPS)

        original = neuron.bias
        neuron.bias = original + EPS
        plus = objective(mlp, x, y, l2_lambda)
        neuron.bias = original - EPS
        minus = objective(mlp, x, y, l2_lambda)
        neuron.bias = original
        grad_b = (plus - minus) / (2 * EPS)
        grads.append((grad_w, grad_b))
    return grads


@pytest.mark.stage1
@pytest.mark.parametrize("x, y", SAMPLES)
def test_gradient_check_sigmoid(x, y):
    """シグモイドの MLP で、逆伝播の勾配が数値微分と一致する。

    条件: MLP([2, 3, 1], seed=0, activation="sigmoid") で、
        各サンプルについて compute_gradients を呼ぶ。
    期待: 全ニューロンの grad_weights / grad_bias が numerical_gradients の結果と一致する
        (許容誤差 rel=GRAD_REL, abs=GRAD_ABS)。
    """
    # ヒント:
    # - 先に compute_gradients(x, y) を呼んで、各ニューロンに解析的な勾配を求めさせる
    # - numerical_gradients(mlp, x, y) と、mlp.neurons() の順に 1 つずつ比べる
    mlp = MLP([2, 3, 1], seed=0, activation="sigmoid")
    mlp.compute_gradients(x, y)
    for neuron, (grad_w, grad_b) in zip(mlp.neurons(), numerical_gradients(mlp, x, y)):
        assert neuron.grad_weights == pytest.approx(grad_w, rel=GRAD_REL, abs=GRAD_ABS)
        assert neuron.grad_bias == pytest.approx(grad_b, rel=GRAD_REL, abs=GRAD_ABS)


@pytest.mark.stage2
@pytest.mark.parametrize("x, y", SAMPLES)
def test_gradient_check_with_l2(x, y):
    """L2 正則化ありでも、逆伝播の勾配が目的関数 J の数値微分と一致する。

    条件: MLP([2, 3, 1], seed=0) で、l2_lambda=0.1 として compute_gradients を呼ぶ。
    期待: 全ニューロンの grad_weights / grad_bias が
        numerical_gradients(..., l2_lambda=0.1) と一致する
        (許容誤差 rel=GRAD_REL, abs=GRAD_ABS)。
    """
    # ヒント:
    # - test_gradient_check_sigmoid とほぼ同じ。compute_gradients と numerical_gradients の両方に
    #   同じ l2_lambda を渡す
    l2_lambda = 0.1
    mlp = MLP([2, 3, 1], seed=0)
    mlp.compute_gradients(x, y, l2_lambda)
    numerical = numerical_gradients(mlp, x, y, l2_lambda)
    for neuron, (grad_w, grad_b) in zip(mlp.neurons(), numerical):
        assert neuron.grad_weights == pytest.approx(grad_w, rel=GRAD_REL, abs=GRAD_ABS)
        assert neuron.grad_bias == pytest.approx(grad_b, rel=GRAD_REL, abs=GRAD_ABS)


@pytest.mark.stage2
def test_l2_adds_lambda_times_weights_to_gradient():
    """L2 正則化は、重みの勾配にだけ λ × 重み を加える(バイアスは対象外)。

    条件: 重み (0.5, -0.3)、バイアス 0.1 の MLP([2, 1]) に入力 (1.0, 2.0) を与えると、
        出力は σ(0) = 0.5。正解も 0.5 にすると損失の勾配は 0 になる。
        この状態で l2_lambda=0.1 として compute_gradients を呼ぶ。
    期待: grad_weights = 0.1 × (0.5, -0.3) = (0.05, -0.03)、grad_bias = 0(許容誤差 abs=1e-12)。
    """
    # ヒント:
    # - 損失の勾配が 0 になる状況を作ると、正則化の分だけが grad_weights に残る
    # - 重みは mlp.set_weights([([[0.5, -0.3]], [0.1])]) で設定できる
    mlp = MLP([2, 1], seed=0)
    mlp.set_weights([([[0.5, -0.3]], [0.1])])
    mlp.compute_gradients(np.array([1.0, 2.0]), np.array([0.5]), l2_lambda=0.1)
    neuron = mlp.layers[0].neurons[0]
    assert neuron.grad_weights == pytest.approx([0.05, -0.03], abs=1e-12)
    assert neuron.grad_bias == pytest.approx(0.0, abs=1e-12)


@pytest.mark.stage3
@pytest.mark.parametrize("x, y", SAMPLES)
def test_gradient_check_relu(x, y):
    """ReLU の MLP で、逆伝播の勾配が数値微分と一致する。

    条件: MLP([2, 3, 1], seed=1, activation="relu") で、
        各サンプルについて compute_gradients を呼ぶ。
    期待: 全ニューロンの grad_weights / grad_bias が numerical_gradients の結果と一致する
        (許容誤差 rel=GRAD_REL, abs=GRAD_ABS)。
    """
    # ヒント:
    # - test_gradient_check_sigmoid と同じ手順で、activation だけ変える
    # - ReLU は u = 0 で折れ曲がっているので、u が 0 に極端に近いと数値微分がずれる。
    #   seed=1 ではそうならないことを確認済み
    mlp = MLP([2, 3, 1], seed=1, activation="relu")
    mlp.compute_gradients(x, y)
    for neuron, (grad_w, grad_b) in zip(mlp.neurons(), numerical_gradients(mlp, x, y)):
        assert neuron.grad_weights == pytest.approx(grad_w, rel=GRAD_REL, abs=GRAD_ABS)
        assert neuron.grad_bias == pytest.approx(grad_b, rel=GRAD_REL, abs=GRAD_ABS)


@pytest.mark.stage1
def test_train_step_matches_hand_calculation():
    """train_step 1 回(順伝播 → 逆伝播 → 更新)の結果が手計算と一致する。

    条件: 重み (0.5, -0.3)、バイアス 0.1 のシグモイドの MLP([2, 1]) を、入力 (1.0, 2.0)、正解 1.0、
        学習率 0.1 の GradientDescent で 1 回 train_step する。
    期待(許容誤差 abs=1e-12):
        - u = 0、出力 = 0.5 なので、戻り値(更新前の損失)は 1/2 × (0.5 - 1)^2 = 0.125
        - δ = (0.5 - 1) × σ'(0) = -0.125
        - 重み = (0.5, -0.3) - 0.1 × δ × (1.0, 2.0) = (0.5125, -0.275)
        - バイアス = 0.1 - 0.1 × δ = 0.1125
    """
    # ヒント:
    # - mlp.set_weights で重みを設定し、GradientDescent(0.1) を渡して train_step を呼ぶ
    # - 戻り値と、更新後の weights / bias を docstring の値と比べる
    mlp = MLP([2, 1], seed=0)
    mlp.set_weights([([[0.5, -0.3]], [0.1])])
    loss = mlp.train_step(np.array([1.0, 2.0]), np.array([1.0]), GradientDescent(0.1))
    neuron = mlp.layers[0].neurons[0]
    assert loss == pytest.approx(0.125, abs=1e-12)
    assert neuron.weights == pytest.approx([0.5125, -0.275], abs=1e-12)
    assert neuron.bias == pytest.approx(0.1125, abs=1e-12)


@pytest.mark.stage1
def test_train_step_reduces_loss_on_same_sample():
    """十分小さな学習率で 1 回更新すると、同じサンプルに対する損失は減る。

    条件: MLP([2, 3, 1], seed=0) で、サンプル (0.3, -0.8) → 0.5 について、
        学習率 0.1 で train_step した前後の損失を比べる。
    期待: 更新後の損失 < 更新前の損失。
    """
    # ヒント:
    # - 更新前の損失は train_step の戻り値でも、objective(mlp, x, y) でも求められる
    # - 更新後の損失は objective(mlp, x, y) で求める
    x, y = SAMPLES[2]
    mlp = MLP([2, 3, 1], seed=0)
    before = mlp.train_step(x, y, GradientDescent(0.1))
    after = objective(mlp, x, y)
    assert after < before
