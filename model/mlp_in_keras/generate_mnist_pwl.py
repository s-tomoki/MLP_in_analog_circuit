import argparse
import os
from typing import List, Tuple

import converter
import numpy as np
from tensorflow.keras.datasets import mnist


def select_label_samples(X: np.ndarray, y: np.ndarray, label: int, count: int) -> np.ndarray:
    mask = y == label
    selected = X[mask]
    if len(selected) < count:
        raise ValueError(
            f"Label {label} has only {len(selected)} samples, " + f"but {count} were requested."
        )
    return selected[:count]


def voltage_for_pixel(value: int) -> float:
    return 3.3 if value == 255 else 0.0


def build_pwl_points(
    pixel_values: np.ndarray, hold: float = 0.9, transition: float = 0.1
) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []
    if pixel_values.size == 0:
        return points

    first_voltage = voltage_for_pixel(int(pixel_values[0]))
    points.append((0.0, first_voltage))
    points.append((hold, first_voltage))

    for idx in range(1, len(pixel_values)):
        value_voltage = voltage_for_pixel(int(pixel_values[idx]))
        start_time = idx * (hold + transition)
        points.append((start_time, value_voltage))
        points.append((start_time + hold, value_voltage))

    final_time = len(pixel_values) * (hold + transition)
    if points[-1][0] < final_time:
        points.append((final_time, points[-1][1]))

    return points


def write_pwl_files(data_3d: np.ndarray, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    num_patterns, height, width = data_3d.shape
    if height != 7 or width != 7:
        raise ValueError("Expected pooled images of shape (n, 7, 7)")

    for pixel_idx in range(height * width):
        row = pixel_idx // width
        col = pixel_idx % width
        pixel_values = data_3d[:, row, col]
        points = build_pwl_points(pixel_values)

        filename = os.path.join(output_dir, f"pwl_pixel_{pixel_idx + 1}.txt")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"# PWL source for pixel {pixel_idx + 1} (row={row}, col={col})\n")
            f.write("# time [s], voltage [V]\n")
            for time_value, voltage in points:
                f.write(f"{time_value:.6f} {voltage:.6f}\n")

    print(f"Wrote PWL files for {height * width} pixels to:" + f"{output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate LTSpice PWL files from MNIST 0/1 patterns."
    )
    parser.add_argument(
        "--samples-per-label",
        type=int,
        default=10,
        help="Number of patterns to extract for each label (default: 10).",
    )
    parser.add_argument(
        "--output-dir",
        default="pwl_outputs",
        help="Directory where pwl_pixel_X.txt files are written.",
    )
    args = parser.parse_args()

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    print(f"Loaded MNIST: X_test={X_test.shape}, y_test={y_test.shape}")

    labels = [0, 1]
    selected_X = np.concatenate(
        [select_label_samples(X_test, y_test, label, args.samples_per_label) for label in labels],
        axis=0,
    )

    cvt = converter.Converter()
    pooled, _ = cvt.pooling_4x4(selected_X, selected_X)
    binary, _ = cvt.binarize(pooled, pooled)
    data_3d = binary.reshape(binary.shape[0], 7, 7)

    print(f"Built 3D dataset with shape {data_3d.shape} (zeros first, then ones)")
    write_pwl_files(data_3d, args.output_dir)


if __name__ == "__main__":
    main()
