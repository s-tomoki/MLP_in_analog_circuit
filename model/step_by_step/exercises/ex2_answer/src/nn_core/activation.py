"""活性化関数と、その導関数を集めたモジュール。

どの関数も、活性化関数に入る前の値 u(重み付き和 + バイアス)を引数に取る。
導関数も「u での傾き」として u の関数で定義する。
スカラーでも NumPy 配列でも受け取れ、配列なら要素ごとに計算して同じ shape で返す。
"""

import numpy as np


def step(u):
    """ステップ関数。u > 0 なら 1、それ以外(0 を含む)は 0。"""
    return np.where(np.asarray(u) > 0, 1.0, 0.0)


def sigmoid(u):
    """シグモイド関数 σ(u) = 1 / (1 + exp(-u))。出力は 0 より大きく 1 より小さい。"""
    return 1.0 / (1.0 + np.exp(-np.clip(u, -500, 500)))


def sigmoid_derivative(u):
    """シグモイド関数の導関数 σ'(u)。"""
    s = sigmoid(u)
    return s * (1.0 - s)


def relu(u):
    """ReLU 関数。u > 0 なら u、それ以外は 0。"""
    return np.maximum(u, 0.0)


def relu_derivative(u):
    """ReLU 関数の導関数。u > 0 なら 1、それ以外(0 を含む)は 0 とする。"""
    return np.where(np.asarray(u) > 0, 1.0, 0.0)


def step_derivative(u):
    """ステップ関数の「代わりの」導関数(Straight-Through Estimator)。

    ステップ関数の本当の導関数は、u = 0 以外ではどこでも 0 になり、学習に使えない。
    そこで逆伝播のときだけ、形の似たシグモイド関数の導関数で代用する。
    """
    return sigmoid_derivative(u)


_ACTIVATIONS = {
    "sigmoid": (sigmoid, sigmoid_derivative),
    "relu": (relu, relu_derivative),
    "step": (step, step_derivative),
}


def get_activation(name: str):
    """名前から (活性化関数, その導関数) の組を取り出す。

    Args:
        name: "sigmoid"、"relu"、"step" のいずれか。

    Returns:
        (f, df) のタプル。f(u) が活性化関数、df(u) がその導関数。

    Raises:
        ValueError: 未知の名前が指定されたとき。
    """
    if name not in _ACTIVATIONS:
        raise ValueError(f"Unknown activation: {name!r} (choose from {sorted(_ACTIVATIONS)})")
    return _ACTIVATIONS[name]
