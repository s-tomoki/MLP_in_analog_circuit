"""多層パーセプトロン(MLP)。"""

from __future__ import annotations

import numpy as np

from .layer import Layer


class MLP:
    """層を順につないだ多層パーセプトロン。

    Attributes:
        layer_sizes: 各層のユニット数のリスト(入力層を含む)。例: [2, 2, 1]
        layers: Layer のリスト(入力層は含まない)。
            layers[0] が入力の直後の層、layers[-1] が出力層。
    """

    def __init__(self, layer_sizes: list[int], seed: int | None = None):
        """
        Args:
            layer_sizes: 例 [2, 2, 1] → 入力 2 個、隠れ層 2 ニューロン、出力層 1 ニューロン。
            seed: 重みを乱数で初期化するときのシード。

        Raises:
            ValueError: layer_sizes の要素数が 2 未満のとき。
        """
        if len(layer_sizes) < 2:
            raise ValueError(
                f"layer_sizes needs at least 2 entries (input and output), got {layer_sizes}"
            )
        self.layer_sizes = list(layer_sizes)
        rng = np.random.default_rng(seed)
        self.layers = [
            Layer(n_out, n_in, rng=rng)
            for n_in, n_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:])
        ]

    def set_weights(self, params) -> None:
        """全層の重みとバイアスをまとめて設定する。

        Args:
            params: 層ごとの (W, b) のリスト。長さは len(self.layers)。
                各 (W, b) の形は Layer.set_weights の引数と同じ。

        Raises:
            ValueError: params の長さが層の数と一致しないとき、
                またはいずれかの層で W, b の shape が合わないとき。
        """
        if len(params) != len(self.layers):
            raise ValueError(f"params must have {len(self.layers)} entries, got {len(params)}")
        for layer, (W, b) in zip(self.layers, params):
            layer.set_weights(W, b)

    def get_weights(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """全層の重みとバイアスを取り出す。

        Returns:
            層ごとの (W, b) のリスト。
        """
        return [layer.get_weights() for layer in self.layers]

    def forward(self, inputs) -> np.ndarray:
        """ネットワーク全体の出力を計算する(順伝播)。

        Args:
            inputs: 長さ layer_sizes[0] の配列。

        Returns:
            shape (layer_sizes[-1],) の NumPy 配列。
        """
        a = np.asarray(inputs, dtype=float)
        for layer in self.layers:
            a = layer.forward(a)
        return a
