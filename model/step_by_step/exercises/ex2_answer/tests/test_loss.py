"""損失関数のテスト。

テスト対象: nn_core.loss
"""

import numpy as np
import pytest
from nn_core import squared_error, squared_error_grad


@pytest.mark.stage1
def test_squared_error_value():
    """二乗誤差 L = 1/2 Σ (pred - target)^2 の値。

    条件: pred = [0.5, 1.0, 0.0]、target = [1.0, 1.0, 1.0] を与える。
    期待: 1/2 × (0.25 + 0 + 1) = 0.625 を実数(float)で返す(許容誤差 abs=1e-12)。
        pred と target が等しいときは 0 を返す。
    """
    loss = squared_error(np.array([0.5, 1.0, 0.0]), np.array([1.0, 1.0, 1.0]))
    assert isinstance(loss, float)
    assert loss == pytest.approx(0.625, abs=1e-12)
    assert squared_error(np.array([0.3]), np.array([0.3])) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.stage1
def test_squared_error_grad_matches_numerical_gradient():
    """squared_error_grad が、squared_error の数値微分と一致する。

    条件: pred = [0.2, 0.7, -0.4]、target = [1.0, 0.0, 0.5] で、pred の各要素を ±h(h = 1e-6)
        ずらしたときの squared_error の中心差分を計算する。
    期待: squared_error_grad(pred, target) の各要素と一致する(許容誤差 rel=1e-6, abs=1e-9)。
        戻り値の shape は pred と同じ (3,)。
    """
    # ヒント:
    # - pred をコピーして k 番目の要素だけ +h / -h した配列を作り、squared_error の差を 2h で割る
    # - これを k = 0, 1, 2 について繰り返し、数値微分の配列を作る
    # - 解析的な勾配(squared_error_grad)と pytest.approx で比べる
    pred = np.array([0.2, 0.7, -0.4])
    target = np.array([1.0, 0.0, 0.5])
    h = 1e-6
    numerical = np.zeros_like(pred)
    for k in range(len(pred)):
        plus = pred.copy()
        minus = pred.copy()
        plus[k] += h
        minus[k] -= h
        numerical[k] = (squared_error(plus, target) - squared_error(minus, target)) / (2 * h)

    grad = squared_error_grad(pred, target)
    assert np.shape(grad) == (3,)
    assert grad == pytest.approx(numerical, rel=1e-6, abs=1e-9)
