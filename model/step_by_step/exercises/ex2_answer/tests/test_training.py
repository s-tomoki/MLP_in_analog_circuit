"""学習全体(MLP.train)のテスト: 再現性と、論理ゲートを学習できること。

テスト対象: nn_core.MLP(train, predict), nn_core.GradientDescent

学習の設定(シード・学習率・エポック数)は、学習が成功することを確認済みの値を定数で固定している。
シードを固定しているので、正しく実装できていれば何度実行しても同じ結果になる。
"""

import numpy as np
import pytest
from nn_core import MLP, GradientDescent

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
Y_NAND = np.array([[1], [1], [1], [0]], dtype=float)
Y_XOR = np.array([[0], [1], [1], [0]], dtype=float)

# 学習の設定: (シード, 学習率, エポック数)
NAND_SIGMOID = (0, 0.5, 500)
XOR_SIGMOID = (2, 0.5, 3000)
XOR_RELU = (0, 0.05, 1000)
NAND_STEP = (0, 0.5, 50)
XOR_STEP = (4, 0.5, 300)
L2_LAMBDA = 1e-3

# 再現性テストの許容誤差: 同じ計算を 2 回するだけなので、ほぼ完全に一致するはず
SAME_REL = 1e-12
SAME_ABS = 1e-12

# XOR_SIGMOID の設定で 5 エポック学習したときの損失の履歴(解答の実装で計算した参照値)
REFERENCE_HISTORY = [0.1427734660, 0.1402741329, 0.1384636301, 0.1371694892, 0.1362465310]


def accuracy(mlp, inputs, targets):
    """出力を 0.5 で 0 / 1 に丸めたときの正解率。"""
    predicted = mlp.predict(inputs) > 0.5
    return float(np.mean(predicted == (targets > 0.5)))


def weight_norm(mlp):
    """全ニューロンの重み(バイアスを除く)の二乗和の平方根。"""
    return float(np.sqrt(sum(np.sum(neuron.weights**2) for neuron in mlp.neurons())))


def train(layer_sizes, activation, targets, setting, l2_lambda=0.0):
    """設定 (シード, 学習率, エポック数) で MLP を作って学習し、(MLP, 損失の履歴) を返す。"""
    seed, learning_rate, epochs = setting
    mlp = MLP(layer_sizes, seed=seed, activation=activation)
    history = mlp.train(X, targets, epochs, GradientDescent(learning_rate), l2_lambda)
    return mlp, history


@pytest.mark.stage1
def test_same_seed_gives_same_history_and_weights():
    """同じシード・同じ設定で学習すると、学習曲線も最終的な重みも毎回同じになる。

    条件: XOR_SIGMOID のシード・学習率で、MLP([2, 2, 1]) を 200 エポック学習することを 2 回行う。
    期待: 2 回の損失の履歴と、学習後の全層の W, b が一致する
        (許容誤差 rel=SAME_REL, abs=SAME_ABS)。
    """
    # ヒント:
    # - 同じ設定で MLP を 2 つ作り、それぞれ train する(train() ヘルパーは epochs が固定なので、
    #   ここでは MLP と GradientDescent を直接作るとよい)
    # - 損失の履歴(リスト)どうしは pytest.approx でまとめて比べられる
    # - 重みは get_weights() の (W, b) を層ごとに比べる
    seed, learning_rate, _ = XOR_SIGMOID
    runs = []
    for _ in range(2):
        mlp = MLP([2, 2, 1], seed=seed)
        history = mlp.train(X, Y_XOR, 200, GradientDescent(learning_rate))
        runs.append((history, mlp.get_weights()))

    (history_a, weights_a), (history_b, weights_b) = runs
    assert history_a == pytest.approx(history_b, rel=SAME_REL, abs=SAME_ABS)
    for (Wa, ba), (Wb, bb) in zip(weights_a, weights_b):
        assert Wa == pytest.approx(Wb, rel=SAME_REL, abs=SAME_ABS)
        assert ba == pytest.approx(bb, rel=SAME_REL, abs=SAME_ABS)


@pytest.mark.stage1
def test_history_matches_reference_values():
    """仕様どおりに実装していれば、学習曲線が参照値と一致する。

    条件: XOR_SIGMOID のシード・学習率で、MLP([2, 2, 1]) を 5 エポック学習する。
    期待: 損失の履歴が REFERENCE_HISTORY と一致する(許容誤差 rel=1e-6, abs=1e-9)。
        一致しない場合は、初期化の順番・損失の定義(1/2 を掛ける、エポック内の平均)・
        サンプルの順番・更新のタイミングのどれかが仕様と違う。
    """
    # ヒント:
    # - 5 エポックだけ学習して、戻り値の履歴を REFERENCE_HISTORY と比べる
    seed, learning_rate, _ = XOR_SIGMOID
    mlp = MLP([2, 2, 1], seed=seed)
    history = mlp.train(X, Y_XOR, 5, GradientDescent(learning_rate))
    assert history == pytest.approx(REFERENCE_HISTORY, rel=1e-6, abs=1e-9)


@pytest.mark.stage1
def test_different_seeds_give_different_histories():
    """シードが違えば初期値が違うので、学習曲線も違う。

    条件: シード 0 と 1 で、MLP([2, 2, 1]) を学習率 0.5 で 5 エポック学習する。
    期待: 損失の履歴が一致しない。
    """
    histories = [
        MLP([2, 2, 1], seed=seed).train(X, Y_XOR, 5, GradientDescent(0.5)) for seed in (0, 1)
    ]
    assert histories[0] != pytest.approx(histories[1], rel=1e-6)


@pytest.mark.stage1
def test_history_length_and_loss_decreases():
    """損失の履歴はエポック数と同じ長さで、学習が進むと損失が下がる。

    条件: NAND_SIGMOID の設定で MLP([2, 1]) を学習する。
    期待:
        - 戻り値の長さ = エポック数、mlp.loss_history も同じ内容。
        - 最後のエポックの損失 < 最初のエポックの損失の半分。
    """
    # ヒント:
    # - train() ヘルパーを使うと (MLP, 履歴) が得られる
    # - エポック数は NAND_SIGMOID[2] で取り出せる
    mlp, history = train([2, 1], "sigmoid", Y_NAND, NAND_SIGMOID)
    assert len(history) == NAND_SIGMOID[2]
    assert mlp.loss_history == history
    assert history[-1] < 0.5 * history[0]


@pytest.mark.stage1
def test_learns_nand_with_sigmoid():
    """シグモイドの MLP([2, 1]) が NAND を学習できる。

    条件: NAND_SIGMOID の設定で学習する。
    期待: 正解率が 100%。
    """
    # ヒント:
    # - train() ヘルパーで学習し、accuracy() ヘルパーで正解率を求める
    mlp, _ = train([2, 1], "sigmoid", Y_NAND, NAND_SIGMOID)
    assert accuracy(mlp, X, Y_NAND) == 1.0


@pytest.mark.stage1
def test_learns_xor_with_sigmoid():
    """シグモイドの MLP([2, 2, 1]) が XOR を学習できる。

    条件: XOR_SIGMOID の設定で学習する。
    期待: 正解率が 100% で、最後のエポックの損失が 0.01 未満。
    """
    # ヒント:
    # - test_learns_nand_with_sigmoid と同じ流れ。損失の履歴の最後の値も確かめる
    mlp, history = train([2, 2, 1], "sigmoid", Y_XOR, XOR_SIGMOID)
    assert accuracy(mlp, X, Y_XOR) == 1.0
    assert history[-1] < 0.01


@pytest.mark.stage2
def test_l2_learns_xor_with_smaller_weights():
    """L2 正則化ありでも XOR を学習でき、重みは正則化なしより小さく抑えられる。

    条件: XOR_SIGMOID の設定で、l2_lambda=0(なし)と L2_LAMBDA(あり)の 2 通り学習する。
    期待:
        - どちらも正解率 100%。
        - 学習後の重みの大きさ(weight_norm)が、L2 ありの方が小さい。
    """
    # ヒント:
    # - train() ヘルパーの最後の引数で l2_lambda を指定できる
    # - weight_norm() ヘルパーで重みの大きさを比べる
    plain, _ = train([2, 2, 1], "sigmoid", Y_XOR, XOR_SIGMOID)
    regularized, _ = train([2, 2, 1], "sigmoid", Y_XOR, XOR_SIGMOID, L2_LAMBDA)
    assert accuracy(plain, X, Y_XOR) == 1.0
    assert accuracy(regularized, X, Y_XOR) == 1.0
    assert weight_norm(regularized) < weight_norm(plain)


@pytest.mark.stage3
def test_learns_xor_with_relu_and_l2():
    """ReLU の MLP([2, 2, 1]) が、L2 正則化ありで XOR を学習できる。

    条件: XOR_RELU の設定、l2_lambda=L2_LAMBDA で学習する(出力層も ReLU)。
    期待: 正解率が 100%。
    """
    # ヒント:
    # - activation に "relu" を指定する以外は、シグモイドのテストと同じ
    mlp, _ = train([2, 2, 1], "relu", Y_XOR, XOR_RELU, L2_LAMBDA)
    assert accuracy(mlp, X, Y_XOR) == 1.0


@pytest.mark.stage4
def test_learns_nand_with_step():
    """ステップ関数の MLP([2, 1]) が、STE を使って NAND を学習できる。

    条件: NAND_STEP の設定で学習する。
    期待: 正解率が 100% で、最後のエポックの損失が 0(出力が 0 / 1 なので、全問正解なら誤差は 0)。
    """
    # ヒント:
    # - activation に "step" を指定する
    # - 損失がちょうど 0 になることも pytest.approx(0.0, abs=1e-12) で確かめる
    mlp, history = train([2, 1], "step", Y_NAND, NAND_STEP)
    assert accuracy(mlp, X, Y_NAND) == 1.0
    assert history[-1] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.stage4
def test_learns_xor_with_step():
    """ステップ関数の MLP([2, 2, 1]) が、STE を使って XOR を学習できる。

    条件: XOR_STEP の設定で学習する。
    期待: 正解率が 100%。
    """
    # ヒント:
    # - test_learns_nand_with_step と同じ流れで、層の構成と設定を XOR 用に変える
    mlp, _ = train([2, 2, 1], "step", Y_XOR, XOR_STEP)
    assert accuracy(mlp, X, Y_XOR) == 1.0


@pytest.mark.stage4
def test_step_network_outputs_are_binary():
    """学習後も、ステップ関数の MLP の出力は 0 か 1 だけになる(学習は STE、出力は step)。

    条件: XOR_STEP の設定で学習し、4 通りの入力で predict する。
    期待: 全出力が 0.0 または 1.0 で、XOR の真理値表と一致する。
    """
    mlp, _ = train([2, 2, 1], "step", Y_XOR, XOR_STEP)
    pred = mlp.predict(X)
    assert set(np.unique(pred)) <= {0.0, 1.0}
    np.testing.assert_array_equal(pred, Y_XOR)
