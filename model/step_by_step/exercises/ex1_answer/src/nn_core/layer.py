"""層(同じ入力を受け取るニューロンの集まり)。"""

from __future__ import annotations

import numpy as np

from .neuron import Neuron


class Layer:
    """同じ入力を共有するニューロンを並べた「層」。

    Attributes:
        neurons: この層に属する Neuron のリスト。
    """

    def __init__(self, num_neurons: int, num_inputs: int, rng: np.random.Generator | None = None):
        """
        Args:
            num_neurons: この層のニューロン数(= この層の出力の個数)。
            num_inputs: 各ニューロンが受け取る入力の個数。
            rng: 乱数生成器。各ニューロンの初期化に使う。
        """
        self.num_neurons = num_neurons
        self.num_inputs = num_inputs
        self.neurons = [Neuron(num_inputs, rng=rng) for _ in range(num_neurons)]

    def set_weights(self, W, b) -> None:
        """層内の全ニューロンの重みとバイアスをまとめて設定する。

        Args:
            W: shape (num_neurons, num_inputs) の配列。i 行目が i 番目のニューロンの重み。
            b: shape (num_neurons,) の配列。i 番目が i 番目のニューロンのバイアス。

        Raises:
            ValueError: W または b の shape が層の構成と合わないとき。
        """
        W = np.asarray(W, dtype=float)
        b = np.asarray(b, dtype=float)
        if W.shape != (self.num_neurons, self.num_inputs):
            raise ValueError(
                f"W must have shape ({self.num_neurons}, {self.num_inputs}), got {W.shape}"
            )
        if b.shape != (self.num_neurons,):
            raise ValueError(f"b must have shape ({self.num_neurons},), got {b.shape}")
        for neuron, w_row, b_i in zip(self.neurons, W, b):
            neuron.set_weights(w_row, b_i)

    def get_weights(self) -> tuple[np.ndarray, np.ndarray]:
        """層内の全ニューロンの重みとバイアスを取り出す。

        Returns:
            (W, b) のタプル。形は set_weights の引数と同じ。
        """
        W = np.array([neuron.weights for neuron in self.neurons])
        b = np.array([neuron.bias for neuron in self.neurons])
        return W, b

    def forward(self, inputs) -> np.ndarray:
        """層の出力を計算する(順伝播)。

        Args:
            inputs: 長さ num_inputs の配列。

        Returns:
            shape (num_neurons,) の NumPy 配列。i 番目が i 番目のニューロンの出力。
        """
        return np.array([neuron.forward(inputs) for neuron in self.neurons])
