"""活性化関数を集めたモジュール。

ステージ1ではステップ関数だけを扱う。
"""

import numpy as np


def step(x):
    """ステップ関数。

    ニューロンが「発火するかどうか」を 0 / 1 で表す。

    Args:
        x: スカラー、または NumPy 配列。

    Returns:
        x と同じ形の値。各要素は 0 または 1。
        入力がちょうど 0 のときは 0 とする。
    """
    return np.where(np.asarray(x) > 0, 1, 0)
