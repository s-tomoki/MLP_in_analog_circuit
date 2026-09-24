"""MNIST データセットの取得と前処理。"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np

URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
DEFAULT_CACHE = Path(__file__).resolve().parent.parent / "data" / "mnist.npz"


def load_mnist(cache_path=DEFAULT_CACHE, url: str = URL):
    """MNIST を読み込む。手元に無ければ URL からダウンロードして cache_path に保存する。

    Args:
        cache_path: 保存先(npz ファイル)のパス。
        url: ダウンロード元の URL。

    Returns:
        ((x_train, y_train), (x_test, y_test))。
        x_* は shape (N, 28, 28) の uint8 画像(0〜255)、y_* は shape (N,) のラベル(0〜9)。
    """
    cache_path = Path(cache_path)
    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        partial = cache_path.with_name(cache_path.name + ".part")
        try:
            urllib.request.urlretrieve(url, partial)
            partial.replace(cache_path)
        finally:
            partial.unlink(missing_ok=True)
    with np.load(cache_path) as f:
        return (f["x_train"], f["y_train"]), (f["x_test"], f["y_test"])


def preprocess(images, size: int = 7, binarize: bool = True) -> np.ndarray:
    """画像をネットワークの入力ベクトルに変換する。

    1. 画素値を 0〜1 の実数にする。
    2. size=7 なら、4x4 画素ずつの平均をとって 28x28 → 7x7 に縮小する(平均プーリング)。
    3. binarize=True なら、0.5 より大きい画素を 1、それ以外を 0 にする(二値化)。
    4. 1 枚の画像を 1 列に並べる。

    Args:
        images: shape (N, 28, 28) の画像。
        size: 7(縮小する)または 28(そのまま)。
        binarize: 二値化するかどうか。

    Returns:
        shape (N, size * size) の NumPy 配列。
    """
    x = np.asarray(images, dtype=float) / 255.0
    n = len(x)
    if size == 7:
        x = x.reshape(n, 7, 4, 7, 4).mean(axis=(2, 4))
    elif size != 28:
        raise ValueError(f"size must be 7 or 28, got {size}")
    x = x.reshape(n, size * size)
    if binarize:
        x = (x > 0.5).astype(float)
    return x


def select_classes(images, labels, classes):
    """指定した数字の画像だけを取り出し、ラベルを 0, 1, 2, ... に付け替える。

    Args:
        images: shape (N, ...) の画像。
        labels: shape (N,) のラベル(0〜9)。
        classes: 取り出す数字のリスト。例: [0, 1]。ラベルは classes 内の位置に付け替える
            (classes = [7, 3] なら 7 → 0、3 → 1)。

    Returns:
        (取り出した画像, 付け替えたラベル)。元の順番を保つ。
    """
    labels = np.asarray(labels)
    mask = np.isin(labels, classes)
    lookup = {c: i for i, c in enumerate(classes)}
    new_labels = np.array([lookup[label] for label in labels[mask]], dtype=int)
    return np.asarray(images)[mask], new_labels


def one_hot(labels, num_classes: int) -> np.ndarray:
    """ラベルを one-hot ベクトル(正解のクラスだけ 1、他は 0)に変換する。

    Args:
        labels: shape (N,) の整数ラベル(0〜num_classes-1)。
        num_classes: クラス数。

    Returns:
        shape (N, num_classes) の float の NumPy 配列。例: ラベル 1、クラス数 3 → [0, 1, 0]。
    """
    return np.eye(num_classes)[np.asarray(labels, dtype=int)]
