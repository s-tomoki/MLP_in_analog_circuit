#!/usr/bin/env python3
"""Evaluate NAND perceptron trained with sigmoid vs step inference (Sigmoid STE check).

Steps performed:
1. Train a 2-input 1-output perceptron with `sigmoid` activation on NAND data.
2. Save trained weights and bias.
3. Repeat training `n_runs` times and save all parameters and sigmoid outputs.
4. Build inference perceptron with hard step activation and apply saved params.
5. Compare step outputs to sigmoid outputs for each run.
6. Build a confusion (fusion) matrix comparing sigmoid vs step outputs across all runs/samples.
7. Save per-run params, outputs and match flags to CSV, and save confusion matrix to CSV.

Usage:
  python eval_nand_by_sigmoid_ste.py --runs 100 --epochs 1000

"""
import argparse
import csv

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Lambda
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

NAND_X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
NAND_Y = np.array([[1.0], [1.0], [1.0], [0.0]], dtype=np.float32)
INPUT_LABELS = ["00", "01", "10", "11"]


def build_train_model():
    m = Sequential()
    m.add(Dense(1, activation="sigmoid", input_shape=(2,)))
    m.compile(
        optimizer=Adam(learning_rate=0.01), loss="binary_crossentropy", metrics=["accuracy"]
    )  # lr adjustable
    return m


def build_step_model():
    m = Sequential()
    m.add(Dense(1, input_shape=(2,)))
    m.add(Lambda(lambda z: tf.where(z > 0, tf.ones_like(z), tf.zeros_like(z))))
    return m


def set_model_params(model, weights, bias):
    # weights: iterable of two floats
    kernel = np.array(weights, dtype=np.float32).reshape((2, 1))
    bias_arr = np.array([bias], dtype=np.float32)
    model.layers[0].set_weights([kernel, bias_arr])


def evaluate_one_run(epochs):
    # Train model with sigmoid activation
    model = build_train_model()
    model.fit(NAND_X, NAND_Y, epochs=epochs, verbose=0)
    kernel, bias = model.layers[0].get_weights()
    weights = [float(kernel[0, 0]), float(kernel[1, 0])]
    bias_val = float(bias[0])

    # sigmoid predictions
    preds_sig = model.predict(NAND_X, verbose=0)  # probabilities
    sig_classes = (preds_sig > 0.5).astype(int).flatten()

    # build step model and apply params
    step_model = build_step_model()
    set_model_params(step_model, weights, bias_val)
    preds_step = step_model.predict(NAND_X, verbose=0)
    step_classes = (preds_step > 0.5).astype(int).flatten()

    # per-input match flags
    matches = (sig_classes == step_classes).astype(int)
    all_match = int(matches.all())

    return {
        "weights": weights,
        "bias": bias_val,
        "preds_sig": preds_sig.flatten().tolist(),
        "sig_classes": sig_classes.tolist(),
        "preds_step": preds_step.flatten().tolist(),
        "step_classes": step_classes.tolist(),
        "matches": matches.tolist(),
        "all_match": all_match,
    }


def aggregate_confusion(all_sig_classes, all_step_classes):
    # all_sig_classes and all_step_classes are flattened lists of 0/1
    cm = np.zeros((2, 2), dtype=int)
    for s, t in zip(all_sig_classes, all_step_classes):
        cm[int(s), int(t)] += 1
    return cm


def save_csv(rows, out_csv, cm, cm_csv=None):
    # write results CSV
    fieldnames = [
        "run",
        "w0",
        "w1",
        "bias",
        "sig_p_00",
        "sig_p_01",
        "sig_p_10",
        "sig_p_11",
        "sig_c_00",
        "sig_c_01",
        "sig_c_10",
        "sig_c_11",
        "step_p_00",
        "step_p_01",
        "step_p_10",
        "step_p_11",
        "step_c_00",
        "step_c_01",
        "step_c_10",
        "step_c_11",
        "m_00",
        "m_01",
        "m_10",
        "m_11",
        "all_match",
    ]

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # write confusion matrix CSV (rows: sigmoid 0/1, cols: step 0/1)
    with open(cm_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sig\\step", "0", "1"])  # header
        writer.writerow(["0", cm[0, 0], cm[0, 1]])
        writer.writerow(["1", cm[1, 0], cm[1, 1]])


def run_experiment(
    n_runs, epochs, out_csv="eval_nand_results.csv", cm_csv="eval_nand_confusion.csv"
):
    rows = []
    all_sig_flat = []
    all_step_flat = []

    for i in tqdm(range(n_runs), desc="runs", unit="run"):
        result = evaluate_one_run(epochs)
        w0, w1 = result["weights"]
        b = result["bias"]
        sig_probs = result["preds_sig"]
        sig_cls = result["sig_classes"]
        step_probs = result["preds_step"]
        step_cls = result["step_classes"]
        matches = result["matches"]
        all_match = result["all_match"]

        # flatten for confusion
        all_sig_flat.extend(sig_cls)
        all_step_flat.extend(step_cls)

        row = {
            "run": i + 1,
            "w0": w0,
            "w1": w1,
            "bias": b,
            # sigmoid probs and classes
            "sig_p_00": sig_probs[0],
            "sig_p_01": sig_probs[1],
            "sig_p_10": sig_probs[2],
            "sig_p_11": sig_probs[3],
            "sig_c_00": sig_cls[0],
            "sig_c_01": sig_cls[1],
            "sig_c_10": sig_cls[2],
            "sig_c_11": sig_cls[3],
            # step probs and classes
            "step_p_00": step_probs[0],
            "step_p_01": step_probs[1],
            "step_p_10": step_probs[2],
            "step_p_11": step_probs[3],
            "step_c_00": step_cls[0],
            "step_c_01": step_cls[1],
            "step_c_10": step_cls[2],
            "step_c_11": step_cls[3],
            # matches
            "m_00": matches[0],
            "m_01": matches[1],
            "m_10": matches[2],
            "m_11": matches[3],
            "all_match": all_match,
        }
        rows.append(row)

    cm = aggregate_confusion(all_sig_flat, all_step_flat)
    save_csv(rows, out_csv, cm, cm_csv)
    print(f"Saved per-run results to: {out_csv}")
    print(f"Saved confusion matrix to: {cm_csv}")
    print("Confusion matrix (rows: sigmoid=0/1, cols: step=0/1):")
    print(cm)


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate NAND perceptron trained with sigmoid vs step inference"
    )
    p.add_argument(
        "--runs",
        type=int,
        default=100,
        help="Number of independent trainings to run (default: 100)",
    )
    p.add_argument(
        "--epochs", type=int, default=1000, help="Epochs per training run (default: 1000)"
    )
    p.add_argument(
        "--out", default="eval_nand_results.csv", help="Output CSV path for per-run results"
    )
    p.add_argument(
        "--cm_out", default="eval_nand_confusion.csv", help="Output CSV path for confusion matrix"
    )
    return p.parse_args()


def main():
    args = parse_args()
    run_experiment(args.runs, args.epochs, out_csv=args.out, cm_csv=args.cm_out)


if __name__ == "__main__":
    main()
