"""MLP に NAND / XOR を学習させ、結果を確かめるユーティリティ。

学習経過・予測・学習後の重みを標準出力に表示し、決定境界と学習曲線の図を PNG に保存する。
シードなどを省略すると、学習が成功することを確認済みの推奨設定(RECOMMENDED)を使う。
"""

import argparse

import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from nn_core import MLP, GradientDescent

INPUTS = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
TARGETS = {
    "nand": np.array([[1], [1], [1], [0]], dtype=float),
    "xor": np.array([[0], [1], [1], [0]], dtype=float),
}
DEFAULT_HIDDEN = {"nand": [], "xor": [2]}

# (gate, activation) ごとの推奨設定: (シード, 学習率, エポック数)
# 既定の隠れ層・L2 の有無(--l2-lambda 0.001)どちらでも正解率 100% になることを確認済み
RECOMMENDED = {
    ("nand", "sigmoid"): (0, 0.5, 500),
    ("nand", "relu"): (0, 0.05, 300),
    ("nand", "step"): (0, 0.5, 50),
    ("xor", "sigmoid"): (2, 0.5, 3000),
    ("xor", "relu"): (0, 0.05, 1000),
    ("xor", "step"): (4, 0.5, 300),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a from-scratch MLP on NAND / XOR "
        "with backpropagation and gradient descent."
    )
    parser.add_argument("--gate", choices=sorted(TARGETS), default="xor", help="Target gate.")
    parser.add_argument(
        "--activation",
        choices=["sigmoid", "relu", "step"],
        default="sigmoid",
        help="Activation of the hidden layers (default: sigmoid). "
        "'step' is trained with the straight-through estimator.",
    )
    parser.add_argument(
        "--output-activation",
        choices=["sigmoid", "relu", "step"],
        default=None,
        help="Activation of the output layer (default: same as --activation).",
    )
    parser.add_argument(
        "--hidden",
        type=int,
        nargs="*",
        default=None,
        metavar="N",
        help="Hidden layer sizes, e.g. '--hidden 3' or '--hidden 4 4'. "
        "'--hidden' alone means no hidden layer (default: nand=none, xor=2).",
    )
    parser.add_argument("--l2", action="store_true", help="Enable L2 weight regularization.")
    parser.add_argument(
        "--l2-lambda",
        type=float,
        default=0.001,
        help="L2 regularization strength, used only when --l2 is set (default: 0.001).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Default: recommended seed.")
    parser.add_argument(
        "--learning-rate", type=float, default=None, help="Default: recommended value."
    )
    parser.add_argument("--epochs", type=int, default=None, help="Default: recommended value.")
    args = parser.parse_args()

    seed, learning_rate, epochs = RECOMMENDED[(args.gate, args.activation)]
    args.seed = seed if args.seed is None else args.seed
    args.learning_rate = learning_rate if args.learning_rate is None else args.learning_rate
    args.epochs = epochs if args.epochs is None else args.epochs
    if args.hidden is None:
        args.hidden = DEFAULT_HIDDEN[args.gate]
    return args


def main():
    args = parse_args()
    l2_lambda = args.l2_lambda if args.l2 else 0.0
    l2_tag = f"l2_{args.l2_lambda}" if args.l2 else "no_l2"
    run_name = f"{args.gate}_{args.activation}_{l2_tag}_seed{args.seed}"
    targets = TARGETS[args.gate]

    layer_sizes = [2, *args.hidden, 1]
    mlp = MLP(
        layer_sizes,
        seed=args.seed,
        activation=args.activation,
        output_activation=args.output_activation,
    )
    optimizer = GradientDescent(args.learning_rate)
    print(
        f"gate={args.gate}  layers={layer_sizes}  activations={mlp.activations}  "
        f"l2_lambda={l2_lambda}  seed={args.seed}  lr={args.learning_rate}  epochs={args.epochs}"
    )

    # Train, printing the loss about 10 times
    log_every = max(1, args.epochs // 10)
    for start in range(0, args.epochs, log_every):
        n = min(log_every, args.epochs - start)
        history = mlp.train(INPUTS, targets, n, optimizer, l2_lambda)
        print(f"Epoch {start + n:6d}  loss: {history[-1]:.6f}")

    predicted = mlp.predict(INPUTS)[:, 0]
    predicted_binary = (predicted > 0.5).astype(int)
    accuracy = np.mean(predicted_binary == targets[:, 0])
    print("\n x1 x2 | output  (> 0.5) target")
    for x, y, yb, t in zip(INPUTS.astype(int), predicted, predicted_binary, targets[:, 0]):
        mark = "" if yb == t else "  <- mismatch"
        print(f"  {x[0]}  {x[1]} | {y:7.4f}    {yb}       {int(t)}{mark}")
    print("\nTrained weights:")
    for i, (W, b) in enumerate(mlp.get_weights(), start=1):
        print(f"Layer {i}: W={np.round(W, 4).tolist()}  b={np.round(b, 4).tolist()}")

    # Plot the decision boundary and the learning curve
    xs = np.linspace(-0.5, 1.5, 101)
    xx, yy = np.meshgrid(xs, xs)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    Z = (mlp.predict(grid)[:, 0] > 0.5).astype(float).reshape(xx.shape)

    cmap = ListedColormap(["#d7191c", "#2b83ba"])  # 0: red, 1: blue
    fig = Figure(figsize=(11, 4.8))
    ax_boundary, ax_loss = fig.subplots(1, 2)

    ax_boundary.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5], cmap=cmap, alpha=0.35)
    ax_boundary.scatter(
        INPUTS[:, 0],
        INPUTS[:, 1],
        c=targets[:, 0],
        cmap=cmap,
        vmin=0,
        vmax=1,
        s=120,
        edgecolors="k",
    )
    ax_boundary.set_title(f"{args.gate.upper()} Decision Boundary")
    ax_boundary.set_xlabel("Input 1")
    ax_boundary.set_ylabel("Input 2")
    ax_boundary.set_aspect("equal")

    ax_loss.plot(np.arange(1, len(mlp.loss_history) + 1), mlp.loss_history)
    ax_loss.set_title("Training loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss (mean of 1/2 squared error)")
    # A log axis cannot show a loss of exactly 0 (e.g. step activation after convergence)
    ax_loss.set_yscale("log" if min(mlp.loss_history) > 0 else "linear")

    param_text = (
        f"layers={layer_sizes}  activation={args.activation}"
        f"(out: {mlp.activations[-1]})  l2_lambda={l2_lambda}  seed={args.seed}  "
        f"lr={args.learning_rate}  epochs={args.epochs}  accuracy={accuracy:.2f}\n"
        "left: background = MLP output (> 0.5) / points = target (red=0, blue=1)"
    )
    fig.suptitle(param_text, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out_path = f"result_{run_name}.png"
    fig.savefig(out_path)

    print(f"\nAccuracy: {accuracy:.2f}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
