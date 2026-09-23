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
    # ヒント【段階 1】:
    # - σ(u) = 1 / (1 + exp(-u)) を NumPy の関数で計算する(配列にもそのまま使える)
    # - u が大きな負の値(例: -1000)だと exp(-u) がオーバーフローする。
    #   計算の前に u を -500〜500 の範囲に収めておく(NumPy に「値を範囲内に切り詰める」関数がある)
    raise NotImplementedError


def sigmoid_derivative(u):
    """シグモイド関数の導関数 σ'(u)。"""
    # ヒント【段階 1】:
    # - シグモイド関数の導関数は、シグモイド関数自身の値を使って表せる(README の数式を参照)
    # - 上で実装した sigmoid を呼び出して使う
    raise NotImplementedError


def relu(u):
    """ReLU 関数。u > 0 なら u、それ以外は 0。"""
    # ヒント【段階 3】:
    # - 「u と 0 の大きい方」を要素ごとに返せばよい(NumPy に要素ごとの最大値を取る関数がある)
    raise NotImplementedError


def relu_derivative(u):
    """ReLU 関数の導関数。u > 0 なら 1、それ以外(0 を含む)は 0 とする。"""
    # ヒント【段階 3】:
    # - u > 0 の要素は 1、それ以外(0 を含む)は 0 にする
    # - 実装済みの step 関数の書き方が参考になる
    raise NotImplementedError


def step_derivative(u):
    """ステップ関数の「代わりの」導関数(Straight-Through Estimator)。

    ステップ関数の本当の導関数は、u = 0 以外ではどこでも 0 になり、学習に使えない。
    そこで逆伝播のときだけ、形の似たシグモイド関数の導関数で代用する。
    """
    # ヒント【段階 4】:
    # - ステップ関数の本当の導関数は使わず、形の似た関数の導関数で代用する(docstring 参照)
    # - すでに実装した関数を呼び出すだけでよい
    raise NotImplementedError


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
