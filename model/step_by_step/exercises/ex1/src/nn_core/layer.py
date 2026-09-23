"""層(同じ入力を受け取るニューロンの集まり)。"""

from __future__ import annotations

import numpy as np


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
        # ヒント:
        # - Neuron を num_neurons 個作り、リスト self.neurons に入れる
        # - どのニューロンも入力の個数は num_inputs
        # - rng は各ニューロンにそのまま渡す(層の中で 1 つの乱数生成器を使い回す)
        # - set_weights で shape を確認するときのために、ニューロン数と入力数も覚えておく
        raise NotImplementedError

    def set_weights(self, W, b) -> None:
        """層内の全ニューロンの重みとバイアスをまとめて設定する。

        Args:
            W: shape (num_neurons, num_inputs) の配列。i 行目が i 番目のニューロンの重み。
            b: shape (num_neurons,) の配列。i 番目が i 番目のニューロンのバイアス。

        Raises:
            ValueError: W または b の shape が層の構成と合わないとき。
        """
        # ヒント:
        # - W と b を NumPy 配列に変換し、shape が (ニューロン数, 入力数) と
        #    (ニューロン数,) か確かめる
        # - 合わなければ ValueError を送出する
        # - 各ニューロンに「W の自分の行」と「b の自分の要素」を渡して設定する
        #   (Neuron.set_weights を利用する)
        raise NotImplementedError

    def get_weights(self) -> tuple[np.ndarray, np.ndarray]:
        """層内の全ニューロンの重みとバイアスを取り出す。

        Returns:
            (W, b) のタプル。形は set_weights の引数と同じ。
        """
        # ヒント:
        # - 各ニューロンの重みを 1 行ずつ積み重ねると W になる
        # - 各ニューロンのバイアスを並べると b になる
        # - どちらも NumPy 配列にして返す
        raise NotImplementedError

    def forward(self, inputs) -> np.ndarray:
        """層の出力を計算する(順伝播)。

        Args:
            inputs: 長さ num_inputs の配列。

        Returns:
            shape (num_neurons,) の NumPy 配列。i 番目が i 番目のニューロンの出力。
        """
        # ヒント:
        # - 層の中のすべてのニューロンに「同じ入力」を渡す
        # - 各ニューロンの出力を順に並べ、NumPy 配列にして返す
        # - 各ニューロンの中の計算(重み付き和 → バイアスを加える → 活性化関数)は
        #   Neuron.forward に任せればよい
        raise NotImplementedError
