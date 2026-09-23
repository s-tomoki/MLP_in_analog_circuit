"""多層パーセプトロン(MLP)。"""

from __future__ import annotations

import numpy as np


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
        # ヒント:
        # - layer_sizes の要素数が 2 未満なら ValueError を送出する
        #   (入力層と出力層の 2 つは最低限必要)
        # - seed から乱数生成器を 1 つだけ作り、すべての層で共有する
        # - layer_sizes の「隣り合う 2 つの値」の組ごとに Layer を 1 つ作り、
        #   順に self.layers に入れる
        #   前の値がその層の入力数、後ろの値がその層のニューロン数になる
        #   例: [2, 3, 1] → 入力 2・ニューロン 3 の層、入力 3・ニューロン 1 の層 の 2 つ
        raise NotImplementedError

    def set_weights(self, params) -> None:
        """全層の重みとバイアスをまとめて設定する。

        Args:
            params: 層ごとの (W, b) のリスト。長さは len(self.layers)。
                各 (W, b) の形は Layer.set_weights の引数と同じ。

        Raises:
            ValueError: params の長さが層の数と一致しないとき、
                またはいずれかの層で W, b の shape が合わないとき。
        """
        # ヒント:
        # - params の長さと層の数が違えば ValueError を送出する
        # - 各層に、対応する (W, b) を渡して設定する(Layer.set_weights を利用する)
        # - 各層の shape の確認は Layer.set_weights に任せてよい
        raise NotImplementedError

    def get_weights(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """全層の重みとバイアスを取り出す。

        Returns:
            層ごとの (W, b) のリスト。
        """
        # ヒント: 各層の get_weights の結果を、層の順にリストにまとめる
        raise NotImplementedError

    def forward(self, inputs) -> np.ndarray:
        """ネットワーク全体の出力を計算する(順伝播)。

        Args:
            inputs: 長さ layer_sizes[0] の配列。

        Returns:
            shape (layer_sizes[-1],) の NumPy 配列。
        """
        # ヒント(計算の順序):
        # 1. 入力を最初の層に渡し、その層の出力を得る
        # 2. ある層の出力を、次の層の入力として渡す。これを出力層まで繰り返す
        # 3. 出力層の出力をそのまま返す
        # - どの層の中でも、各ニューロンは
        #   「重み付き和 → バイアスを加える → 活性化関数」
        #   の順に計算している
        raise NotImplementedError
