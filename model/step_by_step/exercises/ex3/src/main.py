"""MNIST の手書き数字を識別する MLP を学習・評価するユーティリティ。

既定では 0 と 1 の画像を 7x7 に縮小・二値化し、[49, 2, 2] の ReLU ネットワークを
L2 正則化つきで学習する。エポックごとの損失・テスト正解率を表示し、最後に混同行列を表示して、
学習曲線と予測例の図を PNG に保存する。
"""

import argparse
import time

import numpy as np
from classifier import UNDECIDED, MnistClassifier
from matplotlib.figure import Figure
from mnist_data import load_mnist, preprocess, select_classes
from nn_core import GradientDescent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a from-scratch MLP (nn_core) on MNIST handwritten digits."
    )
    parser.add_argument(
        "--classes", type=int, nargs="+", default=[0, 1], help="Digits to classify (default: 0 1)."
    )
    parser.add_argument(
        "--size",
        type=int,
        choices=[7, 28],
        default=7,
        help="Input image size: 7 = 4x4 average pooling, 28 = original (default: 7).",
    )
    parser.add_argument(
        "--binarize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Binarize pixels at 0.5 (default: on).",
    )
    parser.add_argument(
        "--hidden",
        type=int,
        nargs="*",
        default=[2],
        metavar="N",
        help="Hidden layer sizes, e.g. '--hidden 16' or '--hidden 16 8' (default: 2).",
    )
    parser.add_argument("--activation", choices=["relu", "sigmoid", "step"], default="relu")
    parser.add_argument("--output-activation", choices=["relu", "sigmoid", "step"], default="relu")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument(
        "--l2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="L2 weight regularization (default: on).",
    )
    parser.add_argument("--l2-lambda", type=float, default=0.01)
    parser.add_argument(
        "--train-size", type=int, default=None, help="Use only the first N training images."
    )
    parser.add_argument(
        "--test-size", type=int, default=None, help="Use only the first N test images."
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if len(set(args.classes)) != len(args.classes) or len(args.classes) < 2:
        parser.error("--classes needs at least 2 distinct digits.")
    if any(c < 0 or c > 9 for c in args.classes):
        parser.error("--classes must be digits 0-9.")
    if any(h < 1 for h in args.hidden):
        parser.error("--hidden sizes must be >= 1.")
    return args


def load_data(args):
    (x_train, y_train), (x_test, y_test) = load_mnist()
    x_train, y_train = select_classes(x_train, y_train, args.classes)
    x_test, y_test = select_classes(x_test, y_test, args.classes)
    x_train, y_train = x_train[: args.train_size], y_train[: args.train_size]
    x_test, y_test = x_test[: args.test_size], y_test[: args.test_size]
    return (
        (preprocess(x_train, args.size, args.binarize), y_train),
        (preprocess(x_test, args.size, args.binarize), y_test),
    )


def save_figure(path, history, clf, x_test, y_test, args, title):
    """学習曲線(損失・正解率)と、テスト画像 16 枚の予測例を 1 枚の図にまとめて保存する。"""
    fig = Figure(figsize=(10, 7.5))
    grid = fig.add_gridspec(3, 8, height_ratios=[2.2, 1, 1])

    epochs = np.arange(1, len(history["loss"]) + 1)
    ax_loss = fig.add_subplot(grid[0, :4])
    ax_loss.plot(epochs, history["loss"], marker="o")
    ax_loss.set_title("Training loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_yscale("log")
    ax_acc = fig.add_subplot(grid[0, 4:])
    ax_acc.plot(epochs, history["val_accuracy"], marker="o")
    ax_acc.set_title("Test accuracy")
    ax_acc.set_xlabel("Epoch")

    predicted = clf.predict(x_test[:16])
    for k, (image, true, pred) in enumerate(zip(x_test[:16], y_test[:16], predicted)):
        ax = fig.add_subplot(grid[1 + k // 8, k % 8])
        ax.imshow(image.reshape(args.size, args.size), cmap="gray_r", vmin=0, vmax=1)
        pred_text = "?" if pred == UNDECIDED else str(args.classes[pred])
        ax.set_title(f"{args.classes[true]}→{pred_text}", color="black" if pred == true else "red")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(title, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path)


def main():
    args = parse_args()
    l2_lambda = args.l2_lambda if args.l2 else 0.0
    (x_train, y_train), (x_test, y_test) = load_data(args)
    num_inputs = x_train.shape[1]
    layers = [num_inputs, *args.hidden, len(args.classes)]
    run_name = (
        "".join(map(str, args.classes))
        + "_"
        + "-".join(map(str, layers))
        + ("_bin" if args.binarize else "")
        + f"_seed{args.seed}"
    )
    print(
        f"classes={args.classes}  layers={layers}  size={args.size}x{args.size}  "
        f"binarize={args.binarize}  activation={args.activation}/{args.output_activation}  "
        f"lr={args.learning_rate}  l2_lambda={l2_lambda}  epochs={args.epochs}  seed={args.seed}"
    )
    print(f"train={len(x_train)}  test={len(x_test)}")

    clf = MnistClassifier(
        num_inputs,
        args.hidden,
        len(args.classes),
        seed=args.seed,
        activation=args.activation,
        output_activation=args.output_activation,
    )
    start = time.time()

    def log(epoch, history):
        print(
            f"Epoch {epoch:3d}  loss: {history['loss'][-1]:.5f}  "
            f"test accuracy: {history['val_accuracy'][-1]:.4f}  ({time.time() - start:.1f}s)"
        )

    history = clf.fit(
        x_train,
        y_train,
        args.epochs,
        GradientDescent(args.learning_rate),
        l2_lambda,
        x_val=x_test,
        y_val=y_test,
        log=log,
    )
    elapsed = time.time() - start

    test_accuracy = clf.accuracy(x_test, y_test)
    matrix = clf.confusion_matrix(x_test, y_test)
    print(f"\nTraining + evaluation time: {elapsed:.1f}s")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print("Confusion matrix (rows = true digit, columns = predicted digit, last = undecided):")
    header = "      " + "".join(f"{c:>6}" for c in args.classes) + "     ?"
    print(header)
    for c, row in zip(args.classes, matrix):
        print(f"{c:>6}" + "".join(f"{v:>6}" for v in row))

    out_path = f"result_{run_name}.png"
    title = (
        f"classes={args.classes}  layers={layers}  binarize={args.binarize}  "
        f"act={args.activation}/{args.output_activation}  lr={args.learning_rate}  "
        f"l2={l2_lambda}  epochs={args.epochs}  seed={args.seed}  test_acc={test_accuracy:.4f}"
    )
    save_figure(out_path, history, clf, x_test, y_test, args, title)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
