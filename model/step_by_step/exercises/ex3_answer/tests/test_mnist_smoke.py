"""実際の MNIST を使うスモークテスト。

テスト対象: mnist_data.load_mnist と、MnistClassifier の学習・推論の一連の流れ

「煙が出ないか(大きく壊れていないか)」だけを手早く確かめるテスト。
0 と 1 の画像を数百件だけ使って数エポック学習し、でたらめに答える場合(2 クラスなので 50%)より
明らかに高い正解率が出ることを確かめる。全データでの学習や、最高の精度は求めない。

データは ex3/data/mnist.npz にキャッシュする。手元に無く、ダウンロードもできない
(ネットワークに接続していない等)ときは、このファイルのテストをスキップする。
"""

import zipfile

import numpy as np
import pytest
from classifier import MnistClassifier
from mnist_data import load_mnist, preprocess, select_classes
from nn_core import GradientDescent

pytestmark = pytest.mark.smoke

CLASSES = [0, 1]
TRAIN_SIZE = 500
TEST_SIZE = 500
SEED = 0
EPOCHS = 5
LEARNING_RATE = 0.05
L2_LAMBDA = 0.01
RANDOM_BASELINE = 1 / len(CLASSES)  # でたらめに答えたときの正解率(50%)
MIN_ACCURACY = 0.9  # 合格ライン: ランダムより明らかに高い


@pytest.fixture(scope="module")
def mnist_raw():
    """MNIST 全体を読み込む(このファイルのテストで 1 回だけ)。取得できなければスキップする。"""
    try:
        return load_mnist()
    except (OSError, zipfile.BadZipFile) as error:
        pytest.skip(f"MNIST を取得できないためスキップします: {error}")


@pytest.fixture(scope="module")
def mnist_subset(mnist_raw):
    """0 と 1 の画像を、訓練・テストとも先頭から数百件だけ取り出し、既定の前処理をしたもの。"""
    (x_train, y_train), (x_test, y_test) = mnist_raw
    x_train, y_train = select_classes(x_train, y_train, CLASSES)
    x_test, y_test = select_classes(x_test, y_test, CLASSES)
    return (
        preprocess(x_train[:TRAIN_SIZE]),
        y_train[:TRAIN_SIZE],
        preprocess(x_test[:TEST_SIZE]),
        y_test[:TEST_SIZE],
    )


def train_small_model(subset):
    """スモークテスト用の設定で既定の識別器を学習し、(識別器, 履歴) を返す。"""
    x_train, y_train, _, _ = subset
    clf = MnistClassifier(seed=SEED)
    history = clf.fit(x_train, y_train, EPOCHS, GradientDescent(LEARNING_RATE), L2_LAMBDA)
    return clf, history


def test_smoke_dataset_shapes(mnist_raw):
    """読み込んだ MNIST の件数・形・ラベルの範囲。

    条件: load_mnist() の戻り値を調べる。
    期待: 訓練 60,000 件・テスト 10,000 件、画像は 28x28 の uint8、ラベルは 0〜9。
    """
    (x_train, y_train), (x_test, y_test) = mnist_raw
    assert x_train.shape == (60000, 28, 28)
    assert x_test.shape == (10000, 28, 28)
    assert x_train.dtype == np.uint8
    assert y_train.shape == (60000,)
    assert y_test.shape == (10000,)
    assert set(np.unique(y_train)) == set(range(10))


def test_smoke_subset_training_beats_random_baseline(mnist_subset):
    """数百件・数エポックの学習で、ランダムより明らかに高い正解率になる。

    条件: 訓練 TRAIN_SIZE 件の 0 / 1 画像で、SEED・EPOCHS・LEARNING_RATE・L2_LAMBDA の設定で
        既定の識別器([49, 2, 2]、ReLU)を学習し、テスト TEST_SIZE 件で正解率を測る。
    期待: 正解率が MIN_ACCURACY(90%)以上。ランダム(RANDOM_BASELINE = 50%)より明らかに高い。
    """
    # ヒント:
    # - train_small_model(mnist_subset) で学習済みの識別器が得られる
    # - テストデータは mnist_subset の 3 番目と 4 番目の要素
    # - 失敗したときに原因を調べやすいよう、assert のメッセージに正解率を入れておくとよい
    #   (例: assert 条件, f"accuracy={accuracy:.3f}")
    _, _, x_test, y_test = mnist_subset
    clf, _ = train_small_model(mnist_subset)
    accuracy = clf.accuracy(x_test, y_test)
    assert accuracy > RANDOM_BASELINE
    assert accuracy >= MIN_ACCURACY, f"accuracy={accuracy:.3f}"


def test_smoke_loss_decreases(mnist_subset):
    """学習が進むと損失が下がる。

    条件: test_smoke_subset_training_beats_random_baseline と同じ設定で学習する。
    期待: 最後のエポックの損失が、最初のエポックの損失より小さい。
    """
    _, history = train_small_model(mnist_subset)
    assert history["loss"][-1] < history["loss"][0]
