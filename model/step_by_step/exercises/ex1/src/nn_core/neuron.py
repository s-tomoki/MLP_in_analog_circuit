"""ニューロン(ネットワークの最小単位)。"""

from __future__ import annotations

import numpy as np


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
        # ヒント:
        # - rng が None なら、NumPy の乱数生成器を新しく用意する
        # - 乱数は必ずその rng から取り出す
        #   (同じ rng を渡せば同じ初期値になり、実験を再現できる)
        # - self.weights: 入力の個数と同じ長さの NumPy 配列。標準正規分布の乱数で初期化する
        # - self.bias: 0 以上 1 未満の一様乱数 1 つで初期化する
        # - set_weights で長さを確認するときのために、入力の個数も属性として覚えておく
        raise NotImplementedError

    def set_weights(self, weights, bias) -> None:
        """重みとバイアスを、外から与えた値で置き換える。

        Args:
            weights: 長さ num_inputs の配列(リストでもよい)。
            bias: 実数。

        Raises:
            ValueError: weights の長さが入力の個数と一致しないとき。
        """
        # ヒント:
        # - weights は浮動小数点の NumPy 配列に変換してから保存する
        #   (呼び出し元が後で元の配列を書き換えても影響を受けないよう、コピーにしておくと安全)
        # - 長さが入力の個数と違えば ValueError を送出する
        #   (エラーメッセージに「期待した長さ」と「実際の長さ」を入れておくとデバッグしやすい)
        # - bias は実数(float)として保存する
        raise NotImplementedError

    def forward(self, inputs) -> int:
        """入力に対するニューロンの出力を計算する(順伝播)。

        Args:
            inputs: 長さ num_inputs の配列(リストでもよい)。

        Returns:
            0 または 1。
        """
        # ヒント(計算の順序):
        # 1. 重み付き和: 入力の各要素と、それに対応する重みを掛け合わせ、すべて足し合わせる
        # 2. 1 の結果にバイアスを加える
        # 3. 2 の値を活性化関数 step に通し、その結果を出力とする
        # - 1 は for 文で書いても、NumPy の内積の関数を使ってもよい
        # - step の戻り値は NumPy の値なので、int に変換して返すとよい
        raise NotImplementedError
