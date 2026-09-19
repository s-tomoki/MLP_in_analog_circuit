import argparse

import matplotlib.pyplot as plt
import numpy as np
from neuralnetwork import NeuralNetwork


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a from-scratch MLP on XOR, with or without L2 weight regularization."
    )
    parser.add_argument(
        "--l2", action="store_true", help="Enable L2 weight regularization (weight decay)."
    )
    parser.add_argument(
        "--l2-lambda",
        type=float,
        default=0.001,
        help="L2 regularization strength, used only when --l2 is set (default: 0.001).",
    )
    parser.add_argument(
        "--activation",
        choices=["relu", "sigmoid"],
        default="relu",
        help="Activation function used by every neuron (default: relu).",
    )
    parser.add_argument("--epochs", type=int, default=50_000)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for weight initialization, so --l2 on/off runs are comparable.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    l2_lambda = args.l2_lambda if args.l2 else 0.0
    run_name = f"{args.activation}_" + (f"l2_{args.l2_lambda}" if args.l2 else "no_l2")

    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    outputs = np.array([[0], [1], [1], [0]])

    layers = [2, 2, 1]
    nn = NeuralNetwork(
        layers,
        args.learning_rate,
        args.epochs,
        l2_lambda=l2_lambda,
        seed=args.seed,
        activation=args.activation,
    )
    print(f"Training with activation={args.activation}, l2_lambda={l2_lambda} (run: {run_name})")
    nn.train(inputs, outputs)

    predicted_output = np.array([nn.predict(x) for x in inputs])
    print("Predicted Output:\n", predicted_output)

    weights = nn.weights()
    print("Trained weights:\n", weights)

    # Round the predicted output to get binary predictions
    predicted_output_binary = np.round(predicted_output)
    accuracy = np.mean(predicted_output_binary.ravel() == outputs.ravel())

    # Plot the decision boundary
    x_min, x_max = inputs[:, 0].min() - 0.5, inputs[:, 0].max() + 0.5
    y_min, y_max = inputs[:, 1].min() - 0.5, inputs[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    Z = np.array([nn.predict(np.array([x, y])) for x, y in zip(xx.ravel(), yy.ravel())])
    Z = np.round(Z.reshape(xx.shape))

    fig, (ax_boundary, ax_loss) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax_boundary.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    ax_boundary.scatter(inputs[:, 0], inputs[:, 1], c=outputs.ravel(), cmap=plt.cm.Spectral)
    ax_boundary.set_title(f"XOR Decision Boundary ({run_name})")
    ax_boundary.set_xlabel("Input 1")
    ax_boundary.set_ylabel("Input 2")

    ax_loss.plot(nn.loss_history)
    ax_loss.set_title("Training MSE")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("MSE")
    ax_loss.set_yscale("log")

    param_text = (
        f"activation={args.activation}  l2_lambda={l2_lambda}  seed={args.seed}  "
        f"lr={args.learning_rate}  epochs={args.epochs}  accuracy={accuracy:.2f}"
    )
    fig.suptitle(param_text, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(f"xor_result_{run_name}.png")
    plt.show()

    print("Accuracy:", accuracy)


if __name__ == "__main__":
    main()
