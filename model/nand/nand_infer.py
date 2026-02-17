#!/usr/bin/env python3
"""Simple inference script for a 2-input, 1-output perceptron.

Usage examples:
  python nand_infer.py --weights=1.0,-1.5 --bias=-2.5

The script builds a Keras Sequential model with a single Dense unit, sets
its weights and bias from the arguments, runs inference on the 4 NAND inputs
and prints per-sample results and overall accuracy.
"""
import argparse

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Lambda
from tensorflow.keras.models import Sequential


def parse_args():
    p = argparse.ArgumentParser(
        description="Infer NAND behavior from given perceptron weights and bias"
    )
    p.add_argument(
        "--weights", required=False, help="Comma-separated weights for the 2 inputs, e.g. 1.0,-1.5"
    )
    p.add_argument(
        "--bias", required=False, type=float, help="Bias value for the perceptron, e.g. -2.5"
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold applied to sigmoid output (default: 0.5)",
    )
    p.add_argument(
        "--activations",
        default="both",
        choices=["sigmoid", "step", "both"],
        help="Which activation(s) to evaluate (default: both)",
    )
    p.add_argument(
        "--weights_csv",
        help="Path to CSV file containing weights (two values)."
        + "If both --weights and --weights_csv are provided, --weights is used",
    )
    p.add_argument(
        "--bias_csv",
        help="Path to CSV file containing bias (single value)."
        + "If both --bias and --bias_csv are provided, --bias is used",
    )
    return p.parse_args()


def build_model(weights, bias, activation="sigmoid"):
    model = Sequential()
    if activation == "sigmoid":
        model.add(Dense(1, activation="sigmoid", input_shape=(2,)))
    else:
        # step activation: Dense (linear) followed by hard step via Lambda
        model.add(Dense(1, input_shape=(2,)))
        model.add(Lambda(lambda z: tf.where(z > 0, tf.ones_like(z), tf.zeros_like(z))))

    # set_weights expects [kernel, bias], kernel shape (in_dim, out_dim)
    kernel = np.array(weights, dtype=np.float32).reshape((2, 1))
    bias_arr = np.array([bias], dtype=np.float32)
    model.layers[0].set_weights([kernel, bias_arr])
    return model


def get_weights(args):
    # parse weights (prefer explicit --weights over --weights_csv)
    if getattr(args, "weights", None):
        w_strs = [s.strip() for s in args.weights.split(",") if s.strip()]
        if len(w_strs) != 2:
            raise SystemExit("--weights must contain exactly two comma-separated values")
        weights = [float(s) for s in w_strs]
    elif getattr(args, "weights_csv", None):
        try:
            arr = np.loadtxt(args.weights_csv, delimiter=",")
        except Exception:
            arr = np.loadtxt(args.weights_csv, delimiter=",", skiprows=1)
        arr = np.array(arr).flatten()
        if arr.size < 2:
            raise SystemExit("weights CSV must contain at least two numeric values")
        weights = [float(arr[0]), float(arr[1])]
    else:
        raise SystemExit("Either --weights or --weights_csv must be provided")
    return weights


def get_bias(args):
    # parse bias (prefer explicit --bias over --bias_csv)
    if getattr(args, "bias", None) is not None:
        bias = float(args.bias)
    elif getattr(args, "bias_csv", None):
        try:
            b = np.loadtxt(args.bias_csv, delimiter=",")
        except Exception:
            b = np.loadtxt(args.bias_csv, delimiter=",", skiprows=1)
        b_arr = np.array(b).flatten()
        if b_arr.size < 1:
            raise SystemExit("bias CSV must contain at least one numeric value")
        bias = float(b_arr[0])
    else:
        raise SystemExit("Either --bias or --bias_csv must be provided")
    return bias


def main():
    args = parse_args()

    weights = get_weights(args)
    bias = get_bias(args)
    activations = (
        [args.activations] if args.activations in ("sigmoid", "step") else ["sigmoid", "step"]
    )

    # NAND dataset
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    y = np.array([[1.0], [1.0], [1.0], [0.0]], dtype=np.float32)

    any_success = False
    for act in activations:
        model = build_model(weights, bias, activation=act)
        preds = model.predict(X)
        # for sigmoid use provided threshold; for step outputs are 0/1 but thresholding still works
        classes = (preds > args.threshold).astype(int)

        print(f"\n=== Activation: {act} ===")
        print("Input\tProb\tPred\tExpected")
        for xi, p, c, yi in zip(X, preds, classes, y):
            print(f"{list(map(int, xi))}\t{p[0]:.4f}\t{int(c[0])}\t{int(yi[0])}")

        accuracy = (classes.flatten() == y.flatten()).mean()
        print(f"Accuracy ({act}): {accuracy*100:.1f}%")

        if accuracy == 1.0:
            print(f"The perceptron with activation={act} reproduces the NAND truth table.")
            any_success = True
        else:
            print(f"The perceptron with activation={act} does NOT reproduce NAND exactly.")

    return 0 if any_success else 1


if __name__ == "__main__":
    main()
