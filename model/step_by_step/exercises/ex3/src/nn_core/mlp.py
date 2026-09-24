"""多層パーセプトロン(MLP)。"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np

from .layer import Layer
from .loss import squared_error, squared_error_grad
from .neuron import Neuron

if TYPE_CHECKING:
    from .optimizer import Optimizer


class MLP:
    """層を順につないだ多層パーセプトロン。

    Attributes:
        layer_sizes: 各層のユニット数のリスト(入力層を含む)。例: [2, 2, 1]
        activations: 各層(入力層を除く)の活性化関数の名前のリスト。
        layers: Layer のリスト(入力層は含まない)。
        loss_history: train で記録したエポックごとの損失(train を呼ぶたびに後ろに追加される)。
    """

    def __init__(
        self,
        layer_sizes: list[int],
        seed: int | None = None,
        activation: str = "sigmoid",
        output_activation: str | None = None,
    ):
        """
        Args:
            layer_sizes: 例 [2, 2, 1] → 入力 2 個、隠れ層 2 ニューロン、出力層 1 ニューロン。
            seed: 重みを乱数で初期化するときのシード。
            activation: 隠れ層の活性化関数の名前。
            output_activation: 出力層の活性化関数の名前。None なら activation と同じ。

        Raises:
            ValueError: layer_sizes の要素数が 2 未満のとき、または未知の活性化関数のとき。
        """
        if len(layer_sizes) < 2:
            raise ValueError(
                f"layer_sizes needs at least 2 entries (input and output), got {layer_sizes}"
            )
        if output_activation is None:
            output_activation = activation
        self.layer_sizes = list(layer_sizes)
        num_layers = len(self.layer_sizes) - 1
        self.activations = [activation] * (num_layers - 1) + [output_activation]

        rng = np.random.default_rng(seed)
        self.layers = [
            Layer(n_out, n_in, rng=rng, activation=act)
            for n_in, n_out, act in zip(
                self.layer_sizes[:-1], self.layer_sizes[1:], self.activations
            )
        ]
        self.loss_history: list[float] = []

    def set_weights(self, params) -> None:
        """全層の重みとバイアスをまとめて設定する。params は層ごとの (W, b) のリスト。"""
        if len(params) != len(self.layers):
            raise ValueError(f"params must have {len(self.layers)} entries, got {len(params)}")
        for layer, (W, b) in zip(self.layers, params):
            layer.set_weights(W, b)

    def get_weights(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """全層の (W, b) をリストで取り出す。"""
        return [layer.get_weights() for layer in self.layers]

    def neurons(self) -> Iterator[Neuron]:
        """入力に近い層から順に、全ニューロンを 1 個ずつ返す。"""
        for layer in self.layers:
            yield from layer.neurons

    def forward(self, inputs) -> np.ndarray:
        """1 サンプル分の出力を計算する(順伝播)。shape (layer_sizes[-1],) の配列を返す。"""
        a = np.asarray(inputs, dtype=float)
        for layer in self.layers:
            a = layer.forward(a)
        return a

    def predict(self, inputs) -> np.ndarray:
        """複数サンプルの出力をまとめて計算する。

        Args:
            inputs: shape (N, layer_sizes[0]) の配列。

        Returns:
            shape (N, layer_sizes[-1]) の NumPy 配列。
        """
        return np.array([self.forward(x) for x in np.asarray(inputs, dtype=float)])

    def compute_gradients(self, x, y, l2_lambda: float = 0.0) -> float:
        """1 サンプルについて、目的関数の勾配を全ニューロンに求める。重みはまだ変更しない。

        目的関数は J = L + (l2_lambda / 2) × (全ニューロンの重みの二乗和)。
        L は squared_error。バイアスは正則化の対象外。

        Args:
            x: 入力。shape (layer_sizes[0],)。
            y: 正解。shape (layer_sizes[-1],)。
            l2_lambda: L2 正則化の強さ λ。0 なら正則化なし。

        Returns:
            このサンプルの損失 L(正則化項は含めない)。
        """
        pred = self.forward(x)
        loss = squared_error(pred, y)

        errors = squared_error_grad(pred, y)
        for layer in reversed(self.layers):
            errors = layer.backward(errors)

        for neuron in self.neurons():
            neuron.grad_weights = neuron.grad_weights + l2_lambda * neuron.weights
        return loss

    def train_step(self, x, y, optimizer: Optimizer, l2_lambda: float = 0.0) -> float:
        """1 サンプルで勾配を求め、optimizer でパラメータを 1 回更新する。

        Returns:
            更新前のパラメータでのこのサンプルの損失 L。
        """
        loss = self.compute_gradients(x, y, l2_lambda)
        optimizer.step(self)
        return loss

    def train(
        self, inputs, targets, epochs: int, optimizer: Optimizer, l2_lambda: float = 0.0
    ) -> list[float]:
        """全サンプルを先頭から順に 1 つずつ使って更新することを、epochs 回繰り返す。

        Args:
            inputs: shape (N, layer_sizes[0]) の配列。
            targets: shape (N, layer_sizes[-1]) の配列。
            epochs: 繰り返す回数。
            optimizer: パラメータの更新に使う最適化手法。
            l2_lambda: L2 正則化の強さ λ。

        Returns:
            エポックごとの損失(そのエポックで train_step が返した損失の平均)のリスト。長さ epochs。
        """
        inputs = np.asarray(inputs, dtype=float)
        targets = np.asarray(targets, dtype=float)
        history = []
        for _ in range(epochs):
            total = 0.0
            for x, y in zip(inputs, targets):
                total += self.train_step(x, y, optimizer, l2_lambda)
            history.append(total / len(inputs))
        self.loss_history.extend(history)
        return history
