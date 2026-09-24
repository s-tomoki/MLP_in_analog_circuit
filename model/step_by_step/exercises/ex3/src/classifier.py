"""nn_core の MLP を使った、手書き数字の識別器。"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from nn_core import MLP, Optimizer

UNDECIDED = -1  # どのクラスとも決められない(全クラスのスコアが同じ)ときの予測値


class MnistClassifier:
    """画像ベクトルを受け取り、どの数字(クラス)かを答える識別器。

    出力層のニューロンをクラスに 1 つずつ対応させ、
    正解のクラスだけ 1、他は 0 を出力するように学習する。
    推論では、出力(スコア)が最も大きいクラスを答えにする。

    Attributes:
        num_classes: クラス数。
        mlp: 中身のネットワーク(nn_core.MLP)。
        rng: 学習時にサンプルの順番を入れ替えるための乱数生成器。
    """

    def __init__(
        self,
        num_inputs: int = 49,
        hidden: Sequence[int] = (2,),
        num_classes: int = 2,
        seed: int | None = None,
        activation: str = "relu",
        output_activation: str = "relu",
    ):
        """
        Args:
            num_inputs: 入力ベクトルの長さ(7x7 画像なら 49)。
            hidden: 隠れ層のニューロン数。例: (2,)、(16, 8)。
            num_classes: クラス数(= 出力層のニューロン数)。
            seed: 重みの初期化と、学習時の順番の入れ替えに使うシード。
            activation: 隠れ層の活性化関数の名前。
            output_activation: 出力層の活性化関数の名前。
        """
        self.num_classes = num_classes
        self.mlp = MLP(
            [num_inputs, *hidden, num_classes],
            seed=seed,
            activation=activation,
            output_activation=output_activation,
        )
        self.rng = np.random.default_rng(seed)

    def predict_scores(self, x) -> np.ndarray:
        """各クラスのスコア(出力層の出力)を計算する。

        Args:
            x: shape (N, num_inputs) の入力。

        Returns:
            shape (N, num_classes) の NumPy 配列。
        """
        return self.mlp.predict(x)

    def predict(self, x) -> np.ndarray:
        """各サンプルがどのクラスかを予測する。

        Args:
            x: shape (N, num_inputs) の入力。

        Returns:
            shape (N,) の整数の NumPy 配列。スコアが最大のクラスの番号。
            全クラスのスコアが同じサンプル(例: ReLU の出力がすべて 0)は UNDECIDED(-1)。
        """
        # ヒント(計算の順序):
        # 1. predict_scores でスコア(shape (N, num_classes))を求める
        # 2. 各行(= 各サンプル)で、スコアが最大の列の番号を求める
        #    (NumPy に「最大値の位置」を返す関数があり、axis で行ごと・列ごとを指定できる)
        # 3. 各行の全要素が同じ(先頭の要素と等しい)サンプルを見つけ、その予測を UNDECIDED にする
        #    (そうしないと、スコアが全部 0 のときに「クラス 0」と答えてしまう)
        raise NotImplementedError

    def accuracy(self, x, labels) -> float:
        """正解率(予測が正解と一致した割合)。UNDECIDED は不正解として数える。"""
        return float(np.mean(self.predict(x) == np.asarray(labels)))

    def confusion_matrix(self, x, labels) -> np.ndarray:
        """混同行列を作る。

        Returns:
            shape (num_classes, num_classes + 1) の整数の NumPy 配列。
            [i, j] は「正解が i で、j と予測した件数」。最後の列は UNDECIDED の件数。
        """
        matrix = np.zeros((self.num_classes, self.num_classes + 1), dtype=int)
        for true, pred in zip(np.asarray(labels), self.predict(x)):
            matrix[true, pred if pred != UNDECIDED else self.num_classes] += 1
        return matrix

    def fit(
        self,
        x,
        labels,
        epochs: int,
        optimizer: Optimizer,
        l2_lambda: float = 0.0,
        x_val=None,
        y_val=None,
        log: Callable[[int, dict], None] | None = None,
    ) -> dict:
        """学習データで識別器を学習する。

        各エポックでは、全サンプルを self.rng で決めたランダムな順番に 1 回ずつ使って更新する。

        Args:
            x: shape (N, num_inputs) の入力。
            labels: shape (N,) の整数ラベル。
            epochs: エポック数。
            optimizer: パラメータの更新に使う最適化手法。
            l2_lambda: L2 正則化の強さ λ。
            x_val, y_val: 検証データ(省略可)。与えると、各エポックの後に正解率を測る。
            log: 各エポックの後に log(エポック番号(1 始まり), 履歴) の形で呼ばれる関数(省略可)。

        Returns:
            学習の履歴 {"loss": [...], "val_accuracy": [...]}。
            loss はエポックごとの損失の平均、val_accuracy はエポックごとの検証データの正解率
            (検証データを与えなかった場合は空のリスト)。
        """
        # ヒント(計算の順序):
        # 1. x を浮動小数点の NumPy 配列にし、labels を one_hot で学習の目標(正解ベクトル)に変換する
        # 2. エポックごとに(エポック番号は 1 から数える):
        #    a. self.rng.permutation(サンプル数) で、サンプルを使う順番を決める
        #       (同じシードなら同じ順番になるので、結果が再現できる)
        #    b. その順番で 1 サンプルずつ
        #       self.mlp.train_step(入力, 目標, optimizer, l2_lambda) を呼び、戻り値の損失を合計する
        #    c. 合計をサンプル数で割った値を history["loss"] に追加する
        #    d. 検証データ(x_val)があれば、self.accuracy で正解率を求めて
        #       history["val_accuracy"] に追加する
        #    e. log が与えられていれば、log(エポック番号, history) を呼ぶ
        # 3. history を返す
        raise NotImplementedError
