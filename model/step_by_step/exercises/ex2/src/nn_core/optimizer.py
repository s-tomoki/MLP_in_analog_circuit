"""最適化手法(求めた勾配を使ってパラメータをどう動かすか)。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .mlp import MLP


class Optimizer(ABC):
    """最適化手法の基底クラス。

    MLP が「何を最小化するか(損失とその勾配)」を担当し、Optimizer が
    「勾配を使ってパラメータをどう動かすか」を担当する。
    新しい最適化手法は、このクラスを継承して step を実装すれば MLP を変えずに差し替えられる。
    """

    @abstractmethod
    def step(self, mlp: MLP) -> None:
        """全ニューロンの weights / bias を 1 回更新する。

        使う勾配は、各ニューロンの grad_weights / grad_bias(MLP.compute_gradients で求めたもの)。
        """


class GradientDescent(Optimizer):
    """勾配降下法。パラメータを、勾配と逆の向きに「学習率 × 勾配」だけ動かす。"""

    def __init__(self, learning_rate: float):
        """
        Args:
            learning_rate: 学習率(1 回の更新でどれだけ動かすか)。正の値。

        Raises:
            ValueError: learning_rate が正でないとき。
        """
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")
        self.learning_rate = learning_rate

    def step(self, mlp: MLP) -> None:
        """全ニューロンの重み・バイアスを、勾配と逆の向きに「学習率 × 勾配」だけ動かす。"""
        # ヒント【段階 1】:
        # - mlp.neurons() で全ニューロンを順に取り出す
        # - 各ニューロンの重みとバイアスを、それぞれの勾配(grad_weights / grad_bias)と逆の向きに、
        #   学習率(self.learning_rate)× 勾配 だけ動かす
        # - 【段階 2】L2 正則化の分は compute_gradients で勾配に含めてあるので、ここで扱う必要はない
        raise NotImplementedError
