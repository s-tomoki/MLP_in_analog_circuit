import argparse
import csv
import math
from typing import Tuple

import numpy as np


class WeightToRegister:
    """Helper class encapsulating weight/bias→resistor conversion.

    The various steps previously exposed as module-level functions are now
    methods on this class.  ``cutoff`` is stored on the instance for
    thresholding.
    """

    def __init__(self, cutoff: float = 1e-4) -> None:
        self.cutoff = cutoff

    # --- static helpers --------------------------------------------------
    @staticmethod
    def read_csv_values(file_path: str) -> np.ndarray:
        """Read all numerical values from a CSV file and flatten into 1‑D array.

        The file may contain multiple columns; they are concatenated in row
        order.
        """
        try:
            data = np.loadtxt(file_path, delimiter=",")
        except ValueError:
            # in case of a single value file
            data = np.array([float(open(file_path).read().strip())])

        if data.ndim == 0:
            return data.reshape(1)
        return data.flatten()

    @staticmethod
    def threshold_abs(values: np.ndarray, cutoff: float) -> np.ndarray:
        """Return absolute values with anything smaller than ``cutoff`` rounded
        to zero.
        """
        arr = np.abs(values).copy()
        arr[arr < cutoff] = 0.0
        return arr

    @staticmethod
    def compute_negative_series(neg_vals: np.ndarray) -> Tuple[np.ndarray, float]:
        """Compute negative branch series and derived R value."""
        with np.errstate(divide="ignore", invalid="ignore"):
            f = 1000.0 / neg_vals
        inv_sum = np.sum(1.0 / f[np.isfinite(f)]) + (1.0 / 1000.0)
        R = 1000.0 / inv_sum if inv_sum != 0 else math.inf
        return f, R

    @staticmethod
    def compute_positive_series(pos_vals: np.ndarray, R: float) -> np.ndarray:
        """Compute positive branch series given R from the negative side."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return R * 1000.0 / pos_vals

    @staticmethod
    def save_results(neg_series: np.ndarray, pos_series: np.ndarray, output: str) -> None:
        """Write the two series to a CSV with columns ``negative``/``positive``."""
        max_len = max(len(neg_series), len(pos_series))
        with open(output, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["negative", "positive"])
            for i in range(max_len):
                neg_val = neg_series[i] if i < len(neg_series) else ""
                pos_val = pos_series[i] if i < len(pos_series) else ""
                writer.writerow([neg_val, pos_val])

    # --- instance methods ------------------------------------------------
    def build_params(self, weights_path: str, bias_path: str) -> np.ndarray:
        """Read weight and bias CSVs and concatenate them (weights first)."""
        w = self.read_csv_values(weights_path)
        b = self.read_csv_values(bias_path)
        return np.concatenate([w, b])

    def prune_params(self, params: np.ndarray) -> np.ndarray:
        """Zero out small entries according to the instance cutoff."""
        return self.threshold_abs(params, cutoff=self.cutoff)

    def params_to_registers(self, params: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert flattened params into negative/positive resistor series."""
        negatives = params[params < 0]
        positives = params[params >= 0]

        neg_abs = self.threshold_abs(negatives, cutoff=self.cutoff)
        pos_abs = self.threshold_abs(positives, cutoff=self.cutoff)

        neg_series, R = self.compute_negative_series(neg_abs)
        pos_series = self.compute_positive_series(pos_abs, R)
        return neg_series, pos_series


def main():
    parser = argparse.ArgumentParser(
        description="Generate resistor values from weight and bias CSV files"
    )
    parser.add_argument("weights", help="Path to weights CSV")
    parser.add_argument("bias", help="Path to bias CSV")
    parser.add_argument(
        "-o",
        "--output",
        default="resistor_values.csv",
        help="Output CSV file for the results",
    )
    args = parser.parse_args()

    converter = WeightToRegister()
    params = converter.build_params(args.weights, args.bias)
    params_pruned = converter.prune_params(params)

    neg_series, pos_series = converter.params_to_registers(params_pruned)

    converter.save_results(neg_series, pos_series, args.output)
    print(f"results written to {args.output}")


if __name__ == "__main__":
    main()
