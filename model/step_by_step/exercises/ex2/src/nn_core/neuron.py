"""ニューロン(ネットワークの最小単位)。"""

from __future__ import annotations

import numpy as np

from .activation import get_activation


class Neuron:
    """複数の入力を受け取り、実数を 1 つ出力するニューロン。

    Attributes:
        weights: 各入力に対応する重み。shape (num_inputs,) の NumPy 配列。
        bias: バイアス(実数)。
        activation: 活性化関数の名前。
        activation_fn: 活性化関数 f(u)。
        activation_derivative: 活性化関数の導関数 f'(u)。
        inputs: 直前の forward で受け取った入力。
        u: 直前の forward で計算した「重み付き和 + バイアス」。
        output: 直前の forward の出力 f(u)。
        grad_weights: 直前の backward で求めた、損失の weights に関する勾配。shape (num_inputs,)。
        grad_bias: 直前の backward で求めた、損失の bias に関する勾配(実数)。
    """

    def __init__(
        self,
        num_inputs: int,
        rng: np.random.Generator | None = None,
        activation: str = "sigmoid",
    ):
        """
        Args:
            num_inputs: このニューロンが受け取る入力の個数。
            rng: 乱数生成器。None のときは新しく作る。
            activation: 活性化関数の名前("sigmoid"、"relu"、"step")。

        Raises:
            ValueError: 未知の活性化関数の名前が指定されたとき。
        """
        self.activation = activation
        self.activation_fn, self.activation_derivative = get_activation(activation)

        rng = rng if rng is not None else np.random.default_rng()
        self.num_inputs = num_inputs
        self.weights = rng.normal(size=num_inputs)
        self.bias = float(rng.uniform())

        self.inputs = np.zeros(num_inputs)
        self.u = 0.0
        self.output = 0.0
        self.grad_weights = np.zeros(num_inputs)
        self.grad_bias = 0.0

    def set_weights(self, weights, bias) -> None:
        """重みとバイアスを、外から与えた値で置き換える。

        Raises:
            ValueError: weights の長さが入力の個数と一致しないとき。
        """
        weights = np.array(weights, dtype=float)
        if weights.shape != (self.num_inputs,):
            raise ValueError(f"weights must have shape ({self.num_inputs},), got {weights.shape}")
        self.weights = weights
        self.bias = float(bias)

    def forward(self, inputs) -> float:
        """入力に対するニューロンの出力を計算する(順伝播)。

        逆伝播で使うため、入力・u・出力を属性に保存しておく。

        Args:
            inputs: 長さ num_inputs の配列。

        Returns:
            出力 f(u)(実数)。
        """
        self.inputs = np.asarray(inputs, dtype=float)
        self.u = float(np.dot(self.inputs, self.weights) + self.bias)
        self.output = float(self.activation_fn(self.u))
        return self.output

    def backward(self, error: float) -> np.ndarray:
        """損失の勾配を計算する(逆伝播)。重みはまだ変更しない。

        Args:
            error: 損失をこのニューロンの出力で偏微分した値 ∂L/∂(出力)。

        Returns:
            損失をこのニューロンの各入力で偏微分した値。shape (num_inputs,)。
            (前の層のニューロンにとっての「出力に関する勾配」の一部になる)
        """
        # ヒント【段階 1】(計算の順序):
        # 1. δ = error × f'(u) を計算する。
        #    f' は self.activation_derivative、u は forward で保存した self.u
        # 2. 重みの勾配 = δ × (forward で保存した入力)、バイアスの勾配 = δ。
        #    それぞれ self.grad_weights、self.grad_bias に保存する
        # 3. 前の層へ返す値として、δ × (このニューロンの重み) を返す
        # - 重み・バイアスはここでは変更しない(更新は Optimizer の役目)
        # - 【段階 3・4】活性化関数ごとの違いは self.activation_derivative が吸収するので、
        #   ここを書き換える必要はない
        raise NotImplementedError
