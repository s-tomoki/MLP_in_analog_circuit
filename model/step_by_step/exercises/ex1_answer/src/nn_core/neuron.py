"""ニューロン(ネットワークの最小単位)。"""

from __future__ import annotations

import numpy as np

from .activation import step


class Neuron:
    """複数の入力を受け取り、0 か 1 を 1 つ出力するニューロン。

    Attributes:
        weights: 各入力に対応する重み。shape (num_inputs,) の NumPy 配列。
        bias: バイアス(実数 1 つ)。
    """

    def __init__(self, num_inputs: int, rng: np.random.Generator | None = None):
        """
        Args:
            num_inputs: このニューロンが受け取る入力の個数。
            rng: 乱数生成器。None のときは新しく作る。
        """
        rng = rng if rng is not None else np.random.default_rng()
        self.num_inputs = num_inputs
        self.weights = rng.normal(size=num_inputs)
        self.bias = float(rng.uniform())

    def set_weights(self, weights, bias) -> None:
        """重みとバイアスを、外から与えた値で置き換える。

        Args:
            weights: 長さ num_inputs の配列(リストでもよい)。
            bias: 実数。

        Raises:
            ValueError: weights の長さが入力の個数と一致しないとき。
        """
        weights = np.array(weights, dtype=float)
        if weights.shape != (self.num_inputs,):
            raise ValueError(f"weights must have shape ({self.num_inputs},), got {weights.shape}")
        self.weights = weights
        self.bias = float(bias)

    def forward(self, inputs) -> int:
        """入力に対するニューロンの出力を計算する(順伝播)。

        Args:
            inputs: 長さ num_inputs の配列(リストでもよい)。

        Returns:
            0 または 1。
        """
        weighted_sum = np.dot(np.asarray(inputs, dtype=float), self.weights)
        return int(step(weighted_sum + self.bias))
