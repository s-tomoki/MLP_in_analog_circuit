import argparse
import csv
import json
import os
import random

import converter
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import trainer
from tensorflow.keras.datasets import mnist


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def run_experiment(
    min_classes=2,
    max_classes=5,
    repeats=3,
    layers=(2,),
    epochs=30,
    batch_size=250,
    out_dir="explore_n_classes",
):
    os.makedirs(out_dir, exist_ok=True)

    cvt = converter.Converter()
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    # Pool to 4x4 average pooling (7x7 -> 49 features)
    (X_train_pooled, X_test_pooled) = cvt.pooling_4x4(X_train, X_test)

    results = {}

    for n in range(min_classes, max_classes + 1):
        labels = list(range(n))
        (X_tr_sub, Y_tr_sub), (X_te_sub, Y_te_sub) = cvt.extract_labels(
            labels, X_train_pooled, Y_train, X_test_pooled, Y_test
        )

        # Skip if dataset is empty
        if len(Y_tr_sub) == 0 or len(Y_te_sub) == 0:
            print(f"No samples for first {n} labels, skipping")
            continue

        accs = []
        for run_idx in range(repeats):
            seed = run_idx + 1
            set_seed(seed)

            trainer_obj = trainer.Trainer(X_tr_sub, Y_tr_sub, X_te_sub, Y_te_sub, num_classes=n)
            # create a run-specific directory to save model artifacts
            run_dir = os.path.join(out_dir, f"n{n}", f"run{run_idx}")
            os.makedirs(run_dir, exist_ok=True)

            model, test_results, history = trainer_obj.compile_and_train(
                layers=layers, epochs=epochs, batch_size=batch_size, dirname=run_dir
            )
            trainer_obj.save_training_history(model, history, dirname=run_dir)
            # trainer_obj.save_model_weights(model, dirname=run_dir)

            # test_results[1] is accuracy (fraction)
            accs.append(float(test_results[1]))

        results[n] = accs

        # Save per-n results to CSV
        csv_path = os.path.join(out_dir, f"n{n}", f"accuracies_n_{n}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["run", "accuracy"])
            for i, a in enumerate(accs):
                writer.writerow([i, a])

    return results


def plot_training_histories(out_dir="explore_n_classes"):
    """Plot aggregated training histories with shaded regions for each class."""
    result_dir = os.path.join(out_dir, "result")
    os.makedirs(result_dir, exist_ok=True)

    # Find all n directories
    n_dirs = [
        d
        for d in os.listdir(out_dir)
        if d.startswith("n") and os.path.isdir(os.path.join(out_dir, d))
    ]
    n_values = sorted([int(d[1:]) for d in n_dirs])

    # Create subplots for loss and accuracy
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for n in n_values:
        n_dir = os.path.join(out_dir, f"n{n}")
        run_dirs = [d for d in os.listdir(n_dir) if d.startswith("run")]
        run_dirs = sorted(run_dirs, key=lambda x: int(x[3:]))

        # Load history from each run
        histories = []
        for run_dir in run_dirs:
            history_file = os.path.join(n_dir, run_dir, "training_history.json")
            if os.path.exists(history_file):
                with open(history_file, "r") as f:
                    histories.append(json.load(f))

        if not histories:
            print(f"No training histories found for n={n}")
            continue

        # Convert to numpy arrays
        loss = np.array([h["loss"] for h in histories])
        val_loss = np.array([h["val_loss"] for h in histories])
        accuracy = np.array([h["accuracy"] for h in histories])
        val_accuracy = np.array([h["val_accuracy"] for h in histories])

        epochs = np.arange(1, loss.shape[1] + 1)

        # Calculate mean and std
        loss_mean = np.mean(loss, axis=0)
        loss_std = np.std(loss, axis=0)
        val_loss_mean = np.mean(val_loss, axis=0)
        val_loss_std = np.std(val_loss, axis=0)
        accuracy_mean = np.mean(accuracy, axis=0)
        accuracy_std = np.std(accuracy, axis=0)
        val_accuracy_mean = np.mean(val_accuracy, axis=0)
        val_accuracy_std = np.std(val_accuracy, axis=0)

        # Plot training loss
        axes[0, 0].plot(epochs, loss_mean, label=f"n={n}", linewidth=2)
        axes[0, 0].fill_between(epochs, loss_mean - loss_std, loss_mean + loss_std, alpha=0.2)

        # Plot validation loss
        axes[0, 1].plot(epochs, val_loss_mean, label=f"n={n}", linewidth=2)
        axes[0, 1].fill_between(
            epochs, val_loss_mean - val_loss_std, val_loss_mean + val_loss_std, alpha=0.2
        )

        # Plot training accuracy
        axes[1, 0].plot(epochs, accuracy_mean * 100, label=f"n={n}", linewidth=2)
        axes[1, 0].fill_between(
            epochs,
            (accuracy_mean - accuracy_std) * 100,
            (accuracy_mean + accuracy_std) * 100,
            alpha=0.2,
        )

        # Plot validation accuracy
        axes[1, 1].plot(epochs, val_accuracy_mean * 100, label=f"n={n}", linewidth=2)
        axes[1, 1].fill_between(
            epochs,
            (val_accuracy_mean - val_accuracy_std) * 100,
            (val_accuracy_mean + val_accuracy_std) * 100,
            alpha=0.2,
        )

    # Configure axes
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Training Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].set_title("Validation Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy (%)")
    axes[1, 0].set_title("Training Accuracy")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Accuracy (%)")
    axes[1, 1].set_title("Validation Accuracy")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(result_dir, "training_histories.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"Training histories plot saved to {plot_path}")


def aggregate_and_plot(results, out_dir="explore_n_classes"):
    """Aggregate results and generate plots and summary statistics."""
    ns = sorted(results.keys())

    result_dir = os.path.join(out_dir, "result")
    os.makedirs(result_dir, exist_ok=True)

    plt.figure(figsize=(8, 5))
    box_data = [np.array(results[n]) * 100 for n in ns]
    plt.boxplot(box_data, tick_labels=ns)
    plt.xlabel("Number of classes")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs Number of Classes (boxplot over runs)")
    plt.grid(True, alpha=0.3, axis="y")
    plt.ylim(0, 100)
    plot_path = os.path.join(result_dir, "accuracy_vs_n_classes.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()

    # Calculate statistics for summary
    means = [np.mean(results[n]) * 100 for n in ns]
    stds = [np.std(results[n]) * 100 for n in ns]

    # Save summary CSV
    summary_csv = os.path.join(result_dir, "accuracy_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n_classes", "mean_accuracy", "std_accuracy"])
        for n, m, s in zip(ns, means, stds):
            writer.writerow([n, m, s])

    print(f"Experiment finished. Plots and CSVs saved to {out_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run neural network experiments with different number of classes"
    )
    parser.add_argument(
        "--epochs", type=int, default=30, help="Number of epochs for training (default: 30)"
    )
    parser.add_argument(
        "--n_classes", type=int, default=2, help="Maximum number of classes to use (default: 2)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=250, help="Batch size for training (default: 250)"
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of repeats for each configuration (default: 3)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results = run_experiment(
        min_classes=2,
        max_classes=args.n_classes,
        repeats=args.repeats,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    aggregate_and_plot(results)
    plot_training_histories()


if __name__ == "__main__":
    main()
