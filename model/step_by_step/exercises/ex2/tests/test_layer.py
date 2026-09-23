"""Layer クラスのテスト。

テスト対象: nn_core.Layer(forward, backward)
"""

import numpy as np
import pytest
from nn_core import Layer


@pytest.mark.stage1
@pytest.mark.parametrize("activation", ["sigmoid", "relu", "step"])
def test_structure_and_activation(activation):
    """指定した個数のニューロンが、指定した活性化関数で作られる。

    条件: ニューロン数 3、入力数 2、各 activation で層を作る。
    期待: neurons の長さが 3 で、全ニューロンの activation が指定した名前、重みの shape が (2,)。
    """
    layer = Layer(3, 2, rng=np.random.default_rng(0), activation=activation)
    assert len(layer.neurons) == 3
    for neuron in layer.neurons:
        assert neuron.activation == activation
        assert neuron.weights.shape == (2,)


@pytest.mark.stage1
def test_forward_shape():
    """順伝播の出力は、ニューロン 1 個につき 1 つの値を並べた配列になる。

    条件: ニューロン数 4、入力数 2 のシグモイドの層に入力 (0.3, -0.7) を与える。
    期待: shape (4,) の NumPy 配列で、全要素が 0 より大きく 1 より小さい。
    """
    layer = Layer(4, 2, rng=np.random.default_rng(0))
    out = layer.forward([0.3, -0.7])
    assert isinstance(out, np.ndarray)
    assert out.shape == (4,)
    assert np.all((out > 0) & (out < 1))


@pytest.mark.stage1
def test_backward_returns_sum_over_neurons():
    """層の backward の戻り値は、各ニューロンの寄与 δ_i × w_i の合計(= Wᵀδ)になる。

    条件: W = [[0.2, -0.4, 0.1], [0.5, 0.3, -0.2]]、b = [0.1, -0.1] のシグモイドの層に
        入力 (1.0, 0.5, -1.0) を与えて forward し、errors = (0.6, -0.3) で backward する。
    期待: δ_i = errors_i × σ'(u_i) として、戻り値が Σ_i δ_i × W[i] と一致する
        (許容誤差 rel=1e-12, abs=1e-12)。戻り値の shape は (3,)。
    """
    # ヒント:
    # - forward の後、各ニューロンの u は layer.neurons[i].u で取り出せる
    # - 期待値は、δ_i を計算して δ_i × W[i] を足し合わせて作る(for 文でも、W.T @ δ でもよい)
    pass


@pytest.mark.stage1
def test_backward_sets_gradients_of_every_neuron():
    """層の backward で、層内の全ニューロンの勾配が求まる。

    条件: ニューロン数 3、入力数 2 のシグモイドの層に入力 (1.0, -1.0) を与えて forward し、
        errors = (1.0, 1.0, 1.0) で backward する。
    期待: どのニューロンも grad_weights が 0 でない値になっている(backward 前は 0)。
    """
    # ヒント:
    # - 生成直後の grad_weights はゼロ配列(test_neuron.py 参照)
    # - backward 後に、各ニューロンについて「0 ではない要素がある」ことを確かめる(np.any が使える)
    pass
