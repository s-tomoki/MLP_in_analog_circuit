import csv
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


def main():
    results = run_experiment()
    aggregate_and_plot(results)


if __name__ == "__main__":
    main()
