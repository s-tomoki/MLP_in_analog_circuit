"""手動で与えた重みで MLP を動かし、論理ゲートとして働くかを確かめるユーティリティ。

学習は行わない。重みは --gate のプリセットか、--layer で直接指定する。
真理値表を標準出力に表示し、決定境界の図を PNG に保存する。
"""

import argparse

import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from nn_core import MLP

NUM_INPUTS = 2
INPUTS = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# gate ごとの正解出力(INPUTS の順)と、そのゲートを実現するプリセットの重み [(W, b), ...]
GATES = {
    "nand": {
        "targets": np.array([1, 1, 1, 0]),
        "params": [
            (np.array([[-0.5, -0.5]]), np.array([0.8])),
        ],
    },
    "xor": {
        "targets": np.array([0, 1, 1, 0]),
        "params": [
            (np.array([[0.8, 0.8], [-0.8, -0.8]]), np.array([-0.5, 0.9])),
            (np.array([[0.8, 0.8]]), np.array([-1.2])),
        ],
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a hand-weighted MLP (step activation, no training) as a logic gate."
    )
    parser.add_argument(
        "--gate",
        choices=sorted(GATES),
        default="xor",
        help="Target gate: loads its preset weights and is used to compute accuracy "
        "(default: xor).",
    )
    parser.add_argument(
        "--layer",
        action="append",
        nargs="+",
        type=float,
        metavar="VALUE",
        help="Weights of one layer, overriding the preset. Repeat once per layer "
        "(the last one is the output layer and must have 1 neuron). "
        "List each neuron as its weights followed by its bias, e.g. for 2 inputs: "
        "--layer w1 w2 b [w1 w2 b ...]",
    )
    args = parser.parse_args()

    if args.layer is None:
        args.params = GATES[args.gate]["params"]
    else:
        args.params = layers_to_params(args.layer, parser)
    return args


def layers_to_params(layers, parser):
    """--layer の値の並びを [(W, b), ...] に変換する。入力数は前の層のニューロン数から決まる。"""
    params = []
    num_inputs = NUM_INPUTS
    for i, values in enumerate(layers, start=1):
        per_neuron = num_inputs + 1
        if len(values) % per_neuron != 0:
            parser.error(
                f"--layer #{i}: this layer has {num_inputs} inputs, so each neuron needs "
                f"{num_inputs} weights + 1 bias = {per_neuron} values and the count must be "
                f"a multiple of {per_neuron}, but got {len(values)} values"
            )
        rows = np.array(values).reshape(-1, per_neuron)
        params.append((rows[:, :-1], rows[:, -1]))
        num_inputs = rows.shape[0]
    if num_inputs != 1:
        parser.error(
            f"the last --layer is the output layer and must have exactly 1 neuron, "
            f"but got {num_inputs}"
        )
    return params


def main():
    args = parse_args()
    targets = GATES[args.gate]["targets"]
    weight_source = "custom" if args.layer else "preset"
    run_name = f"{args.gate}_{weight_source}"

    layer_sizes = [NUM_INPUTS] + [W.shape[0] for W, _ in args.params]
    mlp = MLP(layer_sizes)
    mlp.set_weights(args.params)

    print(f"gate={args.gate}  weights={weight_source}  layer_sizes={layer_sizes}")
    for i, (W, b) in enumerate(mlp.get_weights(), start=1):
        print(f"Layer {i}: W={W.tolist()}  b={b.tolist()}")

    predicted = np.array([mlp.forward(x)[0] for x in INPUTS])
    print("\n x1 x2 | output target")
    for x, y, t in zip(INPUTS, predicted, targets):
        mark = "" if y == t else "  <- mismatch"
        print(f"  {x[0]}  {x[1]} |   {y}      {t}{mark}")
    accuracy = np.mean(predicted == targets)

    # Plot the decision boundary
    xs = np.linspace(-0.5, 1.5, 201)
    xx, yy = np.meshgrid(xs, xs)
    Z = np.array([mlp.forward([x, y])[0] for x, y in zip(xx.ravel(), yy.ravel())])
    Z = Z.reshape(xx.shape)

    cmap = ListedColormap(["#d7191c", "#2b83ba"])  # 0: red, 1: blue
    fig = Figure(figsize=(6.0, 5.5))
    ax = fig.subplots()
    ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5], cmap=cmap, alpha=0.35)
    ax.scatter(
        INPUTS[:, 0], INPUTS[:, 1], c=targets, cmap=cmap, vmin=0, vmax=1, s=120, edgecolors="k"
    )
    ax.set_title(f"{args.gate.upper()} Decision Boundary ({run_name})")
    ax.set_xlabel("Input 1")
    ax.set_ylabel("Input 2")
    ax.set_aspect("equal")

    param_text = (
        f"layers={layer_sizes}  weights={weight_source}  accuracy={accuracy:.2f}\n"
        "background: MLP output / points: target (red=0, blue=1)"
    )
    fig.suptitle(param_text, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_path = f"result_{run_name}.png"
    fig.savefig(out_path)

    print(f"\nAccuracy: {accuracy:.2f}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
