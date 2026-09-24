"""MNIST 識別器のユニットテスト。

テスト対象: mnist_data(preprocess, select_classes, one_hot)、classifier.MnistClassifier

実際の MNIST は使わず、乱数で作った少数の「画像」だけで、shape・型・値の範囲・再現性を確かめる。
学習がうまくいくか(精度)は問わない。ネットワーク接続は不要で、数秒以内に終わる。
"""

import numpy as np
import pytest
from classifier import UNDECIDED, MnistClassifier
from mnist_data import one_hot, preprocess, select_classes
from nn_core import GradientDescent


def make_images(n, seed=0):
    """MNIST と同じ形式(shape (n, 28, 28)、uint8、0〜255)の乱数画像を作る。"""
    return np.random.default_rng(seed).integers(0, 256, size=(n, 28, 28), dtype=np.uint8)


def make_dataset(n, seed=0):
    """既定の前処理(7x7・二値化)をした乱数画像と、0 / 1 の乱数ラベルを作る。"""
    x = preprocess(make_images(n, seed))
    y = np.random.default_rng(seed + 100).integers(0, 2, size=n)
    return x, y


@pytest.mark.parametrize(
    "size, binarize, length",
    [(7, True, 49), (7, False, 49), (28, True, 784), (28, False, 784)],
)
def test_preprocess_shape_and_values(size, binarize, length):
    """前処理の出力の shape と値の範囲。

    条件: 乱数画像 5 枚を、size と binarize の各組み合わせで前処理する。
    期待: shape が (5, size×size)、値は 0 以上 1 以下。binarize=True なら値は 0 と 1 だけ。
    """
    x = preprocess(make_images(5), size=size, binarize=binarize)
    assert x.shape == (5, length)
    assert np.all((x >= 0) & (x <= 1))
    if binarize:
        assert set(np.unique(x)) <= {0.0, 1.0}


def test_preprocess_pooling_averages_4x4_blocks():
    """7x7 への縮小では、4x4 画素ずつの平均をとる。

    条件: 左上の 4x4 画素が 255、その右隣の 4x4 画素が 51(= 0.2 × 255)、
        残りが 0 の画像を前処理する。
    期待: 二値化なしでは先頭 2 つの値が 1.0 と 0.2、残りは 0(許容誤差 abs=1e-12)。
        二値化ありでは 1 と 0 になる(0.2 は 0.5 以下なので 0)。
    """
    image = np.zeros((1, 28, 28), dtype=np.uint8)
    image[0, 0:4, 0:4] = 255
    image[0, 0:4, 4:8] = 51
    expected = np.zeros(49)
    expected[0] = 1.0
    expected[1] = 0.2
    assert preprocess(image, binarize=False)[0] == pytest.approx(expected, abs=1e-12)
    assert preprocess(image, binarize=True)[0][:2] == pytest.approx([1.0, 0.0], abs=1e-12)


def test_select_classes_remaps_labels():
    """指定した数字だけを取り出し、ラベルを classes 内の位置に付け替える。

    条件: ラベル [3, 7, 1, 7, 3] の 5 枚から classes=[7, 3] を取り出す。
    期待: 1 以外の 4 枚が元の順番で残り、ラベルは 7 → 0、3 → 1 で [1, 0, 0, 1] になる。
    """
    images = np.arange(5).reshape(5, 1)
    x, y = select_classes(images, np.array([3, 7, 1, 7, 3]), [7, 3])
    np.testing.assert_array_equal(x.ravel(), [0, 1, 3, 4])
    np.testing.assert_array_equal(y, [1, 0, 0, 1])


def test_one_hot():
    """ラベルを one-hot ベクトルに変換する。

    条件: 2 クラスのラベル [1, 0, 1] と、10 クラスのラベル [0, 3, 9] を変換する。
    期待:
        - 2 クラス: [[0, 1], [1, 0], [0, 1]]
        - 10 クラス: shape (3, 10)、各行の和が 1、各行で 1 が立っている位置が元のラベル
    """
    np.testing.assert_array_equal(one_hot([1, 0, 1], 2), [[0, 1], [1, 0], [0, 1]])
    t = one_hot(np.array([0, 3, 9]), 10)
    assert t.shape == (3, 10)
    np.testing.assert_array_equal(t.sum(axis=1), [1, 1, 1])
    np.testing.assert_array_equal(np.argmax(t, axis=1), [0, 3, 9])


def test_default_structure():
    """既定の識別器は [49, 2, 2] の ReLU ネットワーク。

    条件: 引数を省略して MnistClassifier を作る。
    期待: 層構成が [49, 2, 2]、活性化関数が隠れ層・出力層とも relu。
    """
    clf = MnistClassifier(seed=0)
    assert clf.mlp.layer_sizes == [49, 2, 2]
    assert clf.mlp.activations == ["relu", "relu"]


def test_predict_scores_shape_and_range():
    """スコア(出力層の出力)の shape と値の範囲。

    条件: 既定の識別器に、前処理済みの乱数画像 5 枚を与える。
    期待: shape (5, 2) の NumPy 配列で、値はすべて 0 以上(出力層が ReLU のため)。
    """
    x, _ = make_dataset(5)
    scores = MnistClassifier(seed=0).predict_scores(x)
    assert isinstance(scores, np.ndarray)
    assert scores.shape == (5, 2)
    assert np.all(scores >= 0)


def test_backward_gradient_shapes_and_types():
    """1 サンプルの逆伝播で求まる勾配の shape と型。

    条件: 隠れ層 4 の識別器で、乱数画像 1 枚と one-hot の正解を使って mlp.compute_gradients を呼ぶ。
    期待:
        - 戻り値(損失)が float で 0 以上。
        - 全ニューロンについて、grad_weights が weights と同じ shape の float 配列で、
          値が有限(nan や inf でない)。
        - grad_bias が有限の実数。
    """
    # ヒント:
    # - MnistClassifier(hidden=(4,), seed=0) を作り、make_dataset(1) で 1 サンプル用意する
    # - 正解は one_hot(y, 2)[0] のように 1 行取り出して渡す
    # - clf.mlp.neurons() で全ニューロンを順に取り出せる。np.isfinite と np.all が使える
    clf = MnistClassifier(hidden=(4,), seed=0)
    x, y = make_dataset(1)
    loss = clf.mlp.compute_gradients(x[0], one_hot(y, 2)[0])
    assert isinstance(loss, float)
    assert loss >= 0
    for neuron in clf.mlp.neurons():
        assert neuron.grad_weights.shape == neuron.weights.shape
        assert neuron.grad_weights.dtype == np.float64
        assert np.all(np.isfinite(neuron.grad_weights))
        assert np.isfinite(neuron.grad_bias)


def test_predict_returns_labels_and_undecided():
    """predict はスコア最大のクラスを返し、全クラスのスコアが同じなら UNDECIDED(-1)を返す。

    条件:
        1. 既定の識別器(seed=0)に、前処理済みの乱数画像 20 枚を与える。
        2. 全層の重み・バイアスを 0 にして(スコアがすべて 0 になる)、同じ画像を与える。
    期待:
        1. 戻り値が shape (20,) の整数の配列。
           スコアに差があるサンプルでは、スコアの argmax と一致する。
           スコアがすべて同じサンプルは -1。(スコアに差があるサンプルが 1 つ以上あることも確かめる)
        2. 全サンプルが -1。
    """
    # ヒント:
    # - スコアは clf.predict_scores(x) で求められる
    # - 「全クラスのスコアが同じ」かどうかは、各行の全要素が先頭の要素と等しいかで判定できる
    # - 整数型かどうかは np.issubdtype(配列.dtype, np.integer) で調べられる
    # - 重みを 0 にするには clf.mlp.set_weights([(W1, b1), (W2, b2)]) に np.zeros(...) を渡す
    #   (W1: (2, 49)、b1: (2,)、W2: (2, 2)、b2: (2,))
    clf = MnistClassifier(seed=0)
    x, _ = make_dataset(20)
    scores = clf.predict_scores(x)
    predicted = clf.predict(x)
    assert predicted.shape == (20,)
    assert np.issubdtype(predicted.dtype, np.integer)
    decided = ~np.all(scores == scores[:, :1], axis=1)
    assert decided.any()
    np.testing.assert_array_equal(predicted[decided], np.argmax(scores[decided], axis=1))
    assert np.all(predicted[~decided] == UNDECIDED)

    clf.mlp.set_weights([(np.zeros((2, 49)), np.zeros(2)), (np.zeros((2, 2)), np.zeros(2))])
    assert np.all(clf.predict(x) == UNDECIDED)


def test_accuracy_and_confusion_matrix():
    """accuracy と confusion_matrix は、predict の結果から計算される(UNDECIDED は不正解)。

    条件: predict が常に [0, 1, -1, 1] を返すようにした識別器で、正解 [0, 0, 1, 1] と比べる。
    期待:
        - 正解率は 2/4 = 0.5。
        - 混同行列は [[1, 1, 0], [0, 1, 1]](行 = 正解、列 = 予測、最後の列 = 未決定)。
    """
    clf = MnistClassifier(seed=0)
    clf.predict = lambda x: np.array([0, 1, UNDECIDED, 1])  # type: ignore[method-assign]
    x = np.zeros((4, 49))
    labels = np.array([0, 0, 1, 1])
    assert clf.accuracy(x, labels) == pytest.approx(0.5)
    np.testing.assert_array_equal(clf.confusion_matrix(x, labels), [[1, 1, 0], [0, 1, 1]])


def test_fit_is_reproducible():
    """同じシードなら、fit の学習の履歴(損失・検証正解率)と学習後の重みが毎回同じになる。

    条件: 前処理済みの乱数画像 20 枚(検証用に別の 10 枚)で、seed=3 の識別器を
        学習率 0.05・λ=0.01 で 2 エポック学習することを 2 回行う。
    期待:
        - 2 回の loss と val_accuracy の履歴、学習後の全層の W, b が一致する
          (許容誤差 rel=1e-12, abs=1e-12)。
        - loss と val_accuracy の長さはエポック数(2)。
          loss は 0 以上、val_accuracy は 0 以上 1 以下。
    """
    # ヒント:
    # - 同じ設定で MnistClassifier を 2 つ作り、それぞれ fit する
    #   (シャッフルの順番もシードで決まるので、同じになるはず)
    # - 検証データは fit の x_val / y_val に渡す
    # - 重みは clf.mlp.get_weights() で取り出して、層ごとに比べる
    x, y = make_dataset(20)
    x_val, y_val = make_dataset(10, seed=1)
    runs = []
    for _ in range(2):
        clf = MnistClassifier(seed=3)
        history = clf.fit(x, y, 2, GradientDescent(0.05), 0.01, x_val=x_val, y_val=y_val)
        runs.append((history, clf.mlp.get_weights()))

    (history_a, weights_a), (history_b, weights_b) = runs
    for key in ("loss", "val_accuracy"):
        assert len(history_a[key]) == 2
        assert history_a[key] == pytest.approx(history_b[key], rel=1e-12, abs=1e-12)
    assert all(v >= 0 for v in history_a["loss"])
    assert all(0 <= v <= 1 for v in history_a["val_accuracy"])
    for (Wa, ba), (Wb, bb) in zip(weights_a, weights_b):
        assert Wa == pytest.approx(Wb, rel=1e-12, abs=1e-12)
        assert ba == pytest.approx(bb, rel=1e-12, abs=1e-12)


def test_fit_shuffles_samples_with_rng():
    """fit は self.rng を使ってサンプルの順番を入れ替えている。

    条件: 同じ seed で作った識別器を 2 つ用意し、片方だけ rng を別のシードのものに取り替えて、
        同じデータで 1 エポック学習する(重みの初期値は同じで、順番の入れ替え方だけが違う)。
    期待: 2 つの損失の履歴が一致しない(順番を入れ替えずに先頭から使っていると一致してしまう)。
    """
    x, y = make_dataset(20)
    a = MnistClassifier(seed=3)
    b = MnistClassifier(seed=3)
    b.rng = np.random.default_rng(999)
    history_a = a.fit(x, y, 1, GradientDescent(0.05))
    history_b = b.fit(x, y, 1, GradientDescent(0.05))
    assert history_a["loss"] != pytest.approx(history_b["loss"], rel=1e-9)


def test_fit_loss_is_mean_over_samples():
    """履歴の loss は、そのエポックで使った全サンプルの損失の平均になっている。

    条件: 学習率を極端に小さく(1e-12)して、重みがほぼ変わらない状態で 1 エポック学習する。
    期待: loss が、学習前のスコアと one-hot の正解から計算した「1/2 × 二乗誤差の和」の
        サンプル平均と一致する(許容誤差 rel=1e-6)。
    """
    x, y = make_dataset(20)
    clf = MnistClassifier(hidden=(4,), seed=0)
    expected = np.mean(0.5 * np.sum((clf.predict_scores(x) - one_hot(y, 2)) ** 2, axis=1))
    history = clf.fit(x, y, 1, GradientDescent(1e-12))
    assert history["loss"][0] == pytest.approx(expected, rel=1e-6)


def test_fit_calls_log_once_per_epoch():
    """fit は各エポックの後に log(エポック番号, 履歴) を 1 回ずつ呼ぶ。

    検証データを与えなければ、val_accuracy は空のリストになる。


    条件: 乱数画像 10 枚で、検証データなしで 3 エポック学習し、log に渡された値を記録する。
    期待: log がエポック番号 1, 2, 3 の順に呼ばれ、そのときの loss の長さがエポック番号と等しい。
        戻り値の val_accuracy は空のリスト。
    """
    x, y = make_dataset(10)
    calls = []
    history = MnistClassifier(seed=0).fit(
        x, y, 3, GradientDescent(0.05), log=lambda epoch, h: calls.append((epoch, len(h["loss"])))
    )
    assert calls == [(1, 1), (2, 2), (3, 3)]
    assert history["val_accuracy"] == []
