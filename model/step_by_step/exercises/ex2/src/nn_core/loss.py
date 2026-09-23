"""損失関数(ネットワークの出力と正解のずれを測る関数)。"""

import numpy as np


def squared_error(pred, target) -> float:
    """1 サンプル分の二乗誤差 L = 1/2 Σ_k (pred_k - target_k)^2。

    Args:
        pred: ネットワークの出力。shape (n_out,)。
        target: 正解。shape (n_out,)。

    Returns:
        損失の値(実数)。
    """
    # ヒント【段階 1】:
    # - pred と target を浮動小数点の NumPy 配列に変換する
    # - 要素ごとの差を 2 乗して合計し、1/2 を掛ける
    # - 戻り値は Python の float に変換して返す
    raise NotImplementedError


def squared_error_grad(pred, target) -> np.ndarray:
    """squared_error を pred で偏微分した値 ∂L/∂pred。

    Args:
        pred: ネットワークの出力。shape (n_out,)。
        target: 正解。shape (n_out,)。

    Returns:
        shape (n_out,) の NumPy 配列。
    """
    # ヒント【段階 1】:
    # - squared_error を pred_k で偏微分するとどんな式になるか、まず紙の上で計算してみる
    #   (2 乗を微分して出てくる 2 と、前に付いている 1/2 が打ち消し合う)
    # - 要素ごとに計算した結果を、shape (n_out,) の NumPy 配列で返す
    raise NotImplementedError
