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
    diff = np.asarray(pred, dtype=float) - np.asarray(target, dtype=float)
    return float(0.5 * np.sum(diff**2))


def squared_error_grad(pred, target) -> np.ndarray:
    """squared_error を pred で偏微分した値 ∂L/∂pred。

    Args:
        pred: ネットワークの出力。shape (n_out,)。
        target: 正解。shape (n_out,)。

    Returns:
        shape (n_out,) の NumPy 配列。
    """
    return np.asarray(pred, dtype=float) - np.asarray(target, dtype=float)
