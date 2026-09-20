import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "mlp_in_keras"))

from converter import Converter  # noqa: E402
from nn_core import NeuralNetwork  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a from-scratch MLP on 7x7 pooled + binarized MNIST."
    )
    parser.add_argument(
        "--classes", type=int, nargs="+", default=[0, 1], help="Digits to classify (default: 0 1)."
    )
    parser.add_argument(
        "--hidden",
        type=int,
        nargs="*",
        default=[2],
        help="Hidden layer sizes, e.g. --hidden 8 4 (default: 2). Input=49, output=#classes.",
    )
    parser.add_argument("--activation", choices=["relu", "sigmoid"], default="relu")
    parser.add_argument("--output-activation", choices=["relu", "sigmoid"], default="relu")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="1 = per-sample SGD (scalar path); >1 = vectorized mini-batch (default: 1).",
    )
    parser.add_argument("--l2", action="store_true", help="Enable L2 weight decay.")
    parser.add_argument("--l2-lambda", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if len(set(args.classes)) != len(args.classes):
        parser.error("--classes must not contain duplicates.")
    if len(args.classes) < 2:
        parser.error("--classes needs at least 2 classes.")
    if any(c < 0 or c > 9 for c in args.classes):
        parser.error("--classes must be digits 0-9.")
    if any(h < 1 for h in args.hidden) or args.batch_size < 1:
        parser.error("--hidden sizes and --batch-size must be >= 1.")
    return args


CACHE = Path(__file__).resolve().parent / "data" / "mnist.npz"
KERAS_CACHE = Path.home() / ".keras" / "datasets" / "mnist.npz"


def load_mnist():
    """Load raw MNIST, caching it under ./data so the slow tensorflow import is skipped."""
    if not CACHE.exists():
        CACHE.parent.mkdir(exist_ok=True)
        if KERAS_CACHE.exists():
            # Keras' own cache is a plain npz; copy it without importing tensorflow.
            CACHE.write_bytes(KERAS_CACHE.read_bytes())
        else:
            from tensorflow.keras.datasets import mnist

            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            np.savez_compressed(
                CACHE, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test
            )
    with np.load(CACHE) as d:
        return (d["x_train"], d["y_train"]), (d["x_test"], d["y_test"])


def load_data(classes):
    (x_train, y_train), (x_test, y_test) = load_mnist()
    cvt = Converter()
    x_train, x_test = cvt.pooling_4x4(x_train, x_test)
    x_train, x_test = cvt.binarize(x_train, x_test)
    (x_train, y_train), (x_test, y_test) = cvt.extract_labels(
        classes, x_train, y_train, x_test, y_test
    )
    # Binarized values are 0/255 -> 0/1. Labels are remapped to indices into `classes`.
    x_train = x_train.astype(float) / 255
    x_test = x_test.astype(float) / 255
    lut = {c: i for i, c in enumerate(classes)}
    y_train = np.array([lut[y] for y in y_train])
    y_test = np.array([lut[y] for y in y_test])
    return (x_train, y_train), (x_test, y_test)


def evaluate(nn, x, y, num_classes):
    """Return (accuracy, confusion matrix). All-zero outputs (ties) count as wrong."""
    out = nn.predict_batch(x)
    pred = np.argmax(out, axis=1)
    undecided = np.all(out == out[:, :1], axis=1)
    pred = np.where(undecided, -1, pred)
    cm = np.zeros((num_classes, num_classes + 1), dtype=int)  # last column = undecided
    for t, p in zip(y, pred):
        cm[t, p if p >= 0 else num_classes] += 1
    return float(np.mean(pred == y)), cm


def save_weights(nn, dirname, layers, classes):
    """Save weights in the same layout as mlp_in_keras/trainer.py (save_model_weights).

    layer_{i}_weights: (n_in, n_out), rows=inputs / cols=neurons; layer_{i}_bias: (n_out, 1).
    """
    dirname.mkdir(exist_ok=True)
    arrays = {}
    for i, layer_w in enumerate(nn.weights()):
        arrays[f"layer_{i}_weights"] = np.array([w for w, _ in layer_w]).T
        arrays[f"layer_{i}_bias"] = np.array([b for _, b in layer_w])
    np.savez(dirname / "model_weights.npz", layers=np.array(layers), classes=classes, **arrays)
    for key, value in arrays.items():
        np.savetxt(dirname / f"{key}.csv", value.reshape(value.shape[0], -1), delimiter=",")
    print(f"Weights saved to {dirname}/ (model_weights.npz, layer_*_weights.csv, layer_*_bias.csv)")


def main():
    args = parse_args()
    l2_lambda = args.l2_lambda if args.l2 else 0.0
    classes = args.classes
    layers = [49, *args.hidden, len(classes)]
    run_name = "".join(map(str, classes)) + "_" + "-".join(map(str, layers))

    (x_train, y_train), (x_test, y_test) = load_data(classes)
    t_train = np.eye(len(classes))[y_train]
    print(f"classes={classes} layers={layers} train={len(x_train)} test={len(x_test)}")

    nn = NeuralNetwork(
        layers,
        args.learning_rate,
        args.epochs,
        l2_lambda=l2_lambda,
        seed=args.seed,
        activation=args.activation,
        output_activation=args.output_activation,
        batch_size=args.batch_size,
        shuffle=True,
    )
    start = time.time()
    nn.train(x_train, t_train, log_every=1)
    elapsed = time.time() - start

    train_acc, _ = evaluate(nn, x_train, y_train, len(classes))
    test_acc, cm = evaluate(nn, x_test, y_test, len(classes))
    print(f"Training time: {elapsed:.1f}s (batch_size={args.batch_size})")
    print(f"Train accuracy: {train_acc:.4f}  Test accuracy: {test_acc:.4f}")
    print("Confusion matrix (rows=true, cols=pred; last col=undecided):\n", cm)

    out_dir = Path(__file__).resolve().parent
    save_weights(nn, out_dir / f"weights_{run_name}", layers, classes)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(nn.loss_history)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training MSE")
    fig.suptitle(
        f"layers={layers} act={args.activation}/{args.output_activation} lr={args.learning_rate} "
        f"bs={args.batch_size} test_acc={test_acc:.4f}",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(out_dir / f"loss_{run_name}.png")


if __name__ == "__main__":
    main()
