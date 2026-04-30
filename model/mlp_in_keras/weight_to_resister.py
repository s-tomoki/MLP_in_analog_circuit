import argparse
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
        # Check if the first line is a header
        with open(file_path, "r") as f:
            first_line = f.readline().strip()
            try:
                # Try to parse the first element as float
                float(first_line.split(",")[0])
                skiprows = 0
            except ValueError:
                # If not, assume it's a header
                skiprows = 1
        try:
            data = np.loadtxt(file_path, delimiter=",", skiprows=skiprows)
        except ValueError:
            # in case of a single value file
            data = np.array([float(open(file_path).read().strip())])

        if data.ndim == 0:
            return data.reshape(1)
        return data

    @staticmethod
    def threshold_round(values: np.ndarray, cutoff: float) -> np.ndarray:
        """Return values with anything smaller than ``cutoff`` rounded
        to zero.
        """
        arr = np.where(np.abs(values) >= cutoff, values, 0.0)
        return arr

    @staticmethod
    def compute_negative_series(
        params_neg: np.ndarray, res_feedback: float = 1000.0
    ) -> Tuple[np.ndarray, float]:
        """Compute negative branch series and derived R value."""
        with np.errstate(divide="ignore", invalid="ignore"):
            resistors = res_feedback / params_neg
        inv_sum = np.sum(1.0 / -resistors) + (1.0 / res_feedback)
        parallel_resistors = res_feedback * inv_sum
        return resistors, parallel_resistors

    @staticmethod
    def compute_positive_series(
        pos_vals: np.ndarray, R: float, res_k: float = 1000.0
    ) -> np.ndarray:
        """Compute positive branch series given R from the negative side."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return R * res_k / pos_vals

    @staticmethod
    def save_results(array_res: np.ndarray, output: str) -> None:
        """Write the two series to a CSV."""
        np.savetxt(output, array_res, delimiter=",", fmt="%g")

    # --- instance methods ------------------------------------------------
    def build_params(self, weights_path: str, bias_path: str) -> np.ndarray:
        """Read weight and bias CSVs and concatenate them (weights first)."""
        w = self.read_csv_values(weights_path)
        b = self.read_csv_values(bias_path)
        print(f"shape of w: {w.shape}, shape of b: {b.shape}")
        params = np.concatenate([w, b.reshape(-1, w.shape[1])])
        print(f"params: {params}")
        return params

    def prune_params(self, params: np.ndarray) -> np.ndarray:
        """Zero out small entries according to the instance cutoff."""
        return self.threshold_round(params, cutoff=self.cutoff)

    def params_to_resistors(self, params_pruned: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert flattened params into negative/positive resistor series."""

        array_round = params_pruned
        print(f"array_round: {array_round}")
        sorted_rank = array_round.argsort().argsort()
        print(f"sorted_rank: {sorted_rank}")
        sorted_array = np.sort(array_round)
        print(f"sorted_array: {sorted_array}")

        array_negatives = sorted_array[sorted_array < 0]
        array_positives = sorted_array[sorted_array > 0]
        num_zeros = np.size(array_round) - np.count_nonzero(array_round)
        print(f"array_negatives: {array_negatives}")
        print(f"array_positives: {array_positives}")

        neg_series, R = self.compute_negative_series(array_negatives)
        pos_series = self.compute_positive_series(array_positives, R)

        array_resistors_sorted = np.concatenate([neg_series, np.zeros(num_zeros), pos_series])
        print(f"array_resistors_sorted: {array_resistors_sorted}")
        array_resistors_original_order = array_resistors_sorted[sorted_rank]
        print(f"array_resistors_original_order: {array_resistors_original_order}")
        return array_resistors_original_order


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

    converter = WeightToRegister(cutoff=1e-3)
    params = converter.build_params(args.weights, args.bias)
    params_pruned = converter.prune_params(params)
    print(f"params_pruned: {params_pruned}")

    converted_resistors = np.apply_along_axis(converter.params_to_resistors, 0, params_pruned)
    converter.save_results(converted_resistors, args.output)
    print(f"results written to {args.output}")


if __name__ == "__main__":
    main()
