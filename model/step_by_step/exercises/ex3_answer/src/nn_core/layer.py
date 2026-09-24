"""層(同じ入力を受け取るニューロンの集まり)。"""

from __future__ import annotations

import numpy as np

from .neuron import Neuron


class Layer:
    """同じ入力を共有するニューロンを並べた「層」。

    Attributes:
        neurons: この層に属する Neuron のリスト。
    """

    def __init__(
        self,
        num_neurons: int,
        num_inputs: int,
        rng: np.random.Generator | None = None,
        activation: str = "sigmoid",
    ):
        """
        Args:
            num_neurons: この層のニューロン数(= この層の出力の個数)。
            num_inputs: 各ニューロンが受け取る入力の個数。
            rng: 乱数生成器。各ニューロンの初期化に使う。
            activation: この層の全ニューロンが使う活性化関数の名前。
        """
        self.num_neurons = num_neurons
        self.num_inputs = num_inputs
        self.neurons = [
            Neuron(num_inputs, rng=rng, activation=activation) for _ in range(num_neurons)
        ]

    def set_weights(self, W, b) -> None:
        """層内の全ニューロンの重みとバイアスをまとめて設定する。

        Args:
            W: shape (num_neurons, num_inputs) の配列。
            b: shape (num_neurons,) の配列。

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
        """層内の全ニューロンの (W, b) を取り出す。"""
        W = np.array([neuron.weights for neuron in self.neurons])
        b = np.array([neuron.bias for neuron in self.neurons])
        return W, b

    def forward(self, inputs) -> np.ndarray:
        """層の出力を計算する(順伝播)。shape (num_neurons,) の NumPy 配列を返す。"""
        return np.array([neuron.forward(inputs) for neuron in self.neurons])

    def backward(self, errors) -> np.ndarray:
        """層内の全ニューロンについて逆伝播を行う。重みはまだ変更しない。

        Args:
            errors: 損失をこの層の各出力で偏微分した値。shape (num_neurons,)。

        Returns:
            損失をこの層の各入力で偏微分した値。shape (num_inputs,)。
        """
        prev_errors = np.zeros(self.num_inputs)
        for neuron, error in zip(self.neurons, errors):
            prev_errors += neuron.backward(error)
        return prev_errors
