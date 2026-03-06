import numpy as np
import pytest

from . import weight_to_resister


class TestWeightToRegister:
    @pytest.fixture(autouse=True)
    def init(self):
        self.dut = weight_to_resister.WeightToRegister(cutoff=1e-4)

    def test_prune_single(self):
        """[1.0] を与えたときそのまま返す"""
        inp = np.array([1.0])
        out = self.dut.prune_params(inp)
        assert np.array_equal(out, np.array([1.0]))

    def test_prune_positive_negative(self):
        """[1.0, -1.1] は変化しない"""
        inp = np.array([1.0, -1.1])
        out = self.dut.prune_params(inp)
        assert np.array_equal(out, np.array([1.0, 1.1]))

    def test_prune_small_positive(self):
        """[-1.0, 1e-6] -> [-1.0, 0]"""
        inp = np.array([-1.0, 1e-6])
        out = self.dut.prune_params(inp)
        assert np.array_equal(out, np.array([1.0, 0.0]))

    def test_prune_mixed_small(self):
        """[-3e-8, 3.3] -> [0, 3.3]"""
        inp = np.array([-3e-8, 3.3])
        out = self.dut.prune_params(inp)
        assert np.array_equal(out, np.array([0.0, 3.3]))

    def test_prune_all_small(self):
        """[-2e-8, 5e-34] -> [0, 0]"""
        inp = np.array([-2e-8, 5e-34])
        out = self.dut.prune_params(inp)
        assert np.array_equal(out, np.array([0.0, 0.0]))

    def test_compute_one_neg_param(self):
        params_neg = np.array([-1.0])
        resisters, R = self.dut.compute_negative_series(params_neg)
        assert np.allclose(resisters, np.array([-1000.0]))
        assert np.isclose(R, 2.0)

        params_neg = np.array([-0.5])
        resisters, R = self.dut.compute_negative_series(params_neg)
        assert np.allclose(resisters, np.array([-2000.0]))
        assert np.isclose(R, 1.5)

    def test_compute_multiple_neg_params(self):
        params_neg = np.array([-1.0, -2.0])
        resisters, R = self.dut.compute_negative_series(params_neg)
        assert np.allclose(resisters, np.array([-1000.0, -500.0]))
        assert np.isclose(R, 4.0)

        params_neg = np.array([-0.5, -0.25, -0.125])
        resisters, R = self.dut.compute_negative_series(params_neg)
        assert np.allclose(resisters, np.array([-2000.0, -4000.0, -8000.0]))
        assert np.isclose(R, 1.875)
