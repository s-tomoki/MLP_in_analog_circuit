"""
Compare MLP and CNN models on binary MNIST dataset.

Models:
- MLP: 1 hidden layer with 2 units, architecture: input(49) -> Dense(2) -> output(num_classes)
- CNN: Conv2D(1 filter) -> Flatten() -> Dense(16) -> output(num_classes), with input (7, 7, 1)

Dataset:
- MNIST preprocessed with pooling_4x4 and binarization
- Test class counts from 2 to 10

Output:
- Test accuracy comparison plots (3 plots: MLP, CNN, comparison)
- Training history plots for each class count (3 plots per count: MLP, CNN, comparison)
- Parameter count comparison plot
"""

import argparse
import json
import os

import converter
import matplotlib.pyplot as plt
from tensorflow.keras import Input, Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical


def build_mlp_model(input_shape, num_classes, hidden_layers=(2,)):
    """Build MLP model with specified hidden layers."""
    model = Sequential()
    model.add(Input(shape=input_shape))

    # Add hidden layers
    for units in hidden_layers:
        model.add(
            Dense(
                units,
                activation="relu",
                kernel_regularizer=l2(0.01),
                bias_regularizer=l2(0.01),
            )
        )

    # Output layer
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def build_cnn_model(input_shape, num_classes):
    """Build CNN model with Conv2D -> Flatten -> Dense(16) -> output."""
    model = Sequential()
    model.add(Input(shape=input_shape))

    # Conv2D layer
    model.add(
        Conv2D(
            filters=1,
            kernel_size=(3, 3),
            strides=2,
            padding="same",
            activation="relu",
        )
    )

    # Flatten
    model.add(Flatten())

    # Output layer
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def count_parameters(model):
    """Count trainable parameters in the model (excluding optimizer params)."""
    return model.count_params()


def save_model_summaries(
    mlp_model, cnn_model, num_classes, output_dir="compare_mlp_and_cnn_results"
):
    """Save model summaries for MLP and CNN."""
    class_dir = os.path.join(output_dir, f"history_{num_classes}classes")
    os.makedirs(class_dir, exist_ok=True)

    # Save MLP model summary
    with open(os.path.join(class_dir, "mlp_model_summary.txt"), "w") as f:
        mlp_model.summary(print_fn=lambda x: f.write(x + "\n"))

    # Save CNN model summary
    with open(os.path.join(class_dir, "cnn_model_summary.txt"), "w") as f:
        cnn_model.summary(print_fn=lambda x: f.write(x + "\n"))


def train_model(model, X_train, Y_train, X_test, Y_test, epochs=30, batch_size=250):
    """Train model and return history and test results."""
    history = model.fit(
        X_train,
        Y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=0,
    )
    test_results = model.evaluate(X_test, Y_test, verbose=0)
    return history, test_results


def save_training_history_plots(
    histories_dict, num_classes, output_dir="compare_mlp_and_cnn_results"
):
    """Save training history plots for a specific number of classes."""
    os.makedirs(output_dir, exist_ok=True)
    class_dir = os.path.join(output_dir, f"history_{num_classes}classes")
    os.makedirs(class_dir, exist_ok=True)

    # Plot 1: MLP only
    if "mlp" in histories_dict:
        history = histories_dict["mlp"]
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history.history["loss"], label="Training Loss")
        plt.plot(history.history["val_loss"], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title(f"MLP Model Loss ({num_classes} classes)")

        plt.subplot(1, 2, 2)
        plt.plot(history.history["accuracy"], label="Training Accuracy")
        plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title(f"MLP Model Accuracy ({num_classes} classes)")

        plt.tight_layout()
        plt.savefig(os.path.join(class_dir, "mlp_only.png"), dpi=150)
        plt.close()

    # Plot 2: CNN only
    if "cnn" in histories_dict:
        history = histories_dict["cnn"]
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history.history["loss"], label="Training Loss")
        plt.plot(history.history["val_loss"], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title(f"CNN Model Loss ({num_classes} classes)")

        plt.subplot(1, 2, 2)
        plt.plot(history.history["accuracy"], label="Training Accuracy")
        plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title(f"CNN Model Accuracy ({num_classes} classes)")

        plt.tight_layout()
        plt.savefig(os.path.join(class_dir, "cnn_only.png"), dpi=150)
        plt.close()

    # Plot 3: Both models comparison
    if "mlp" in histories_dict and "cnn" in histories_dict:
        plt.figure(figsize=(14, 5))

        mlp_history = histories_dict["mlp"]
        cnn_history = histories_dict["cnn"]

        plt.subplot(1, 3, 1)
        plt.plot(mlp_history.history["loss"], label="MLP Training Loss", marker="o", markersize=3)
        plt.plot(
            mlp_history.history["val_loss"], label="MLP Validation Loss", marker="s", markersize=3
        )
        plt.plot(cnn_history.history["loss"], label="CNN Training Loss", marker="^", markersize=3)
        plt.plot(
            cnn_history.history["val_loss"], label="CNN Validation Loss", marker="x", markersize=3
        )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title(f"Loss Comparison ({num_classes} classes)")

        plt.subplot(1, 3, 2)
        plt.plot(
            mlp_history.history["accuracy"],
            label="MLP Training Accuracy",
            marker="o",
            markersize=3,
        )
        plt.plot(
            mlp_history.history["val_accuracy"],
            label="MLP Validation Accuracy",
            marker="s",
            markersize=3,
        )
        plt.plot(
            cnn_history.history["accuracy"],
            label="CNN Training Accuracy",
            marker="^",
            markersize=3,
        )
        plt.plot(
            cnn_history.history["val_accuracy"],
            label="CNN Validation Accuracy",
            marker="x",
            markersize=3,
        )
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title(f"Accuracy Comparison ({num_classes} classes)")

        # Zoom in on validation metrics
        plt.subplot(1, 3, 3)
        plt.plot(
            mlp_history.history["val_accuracy"],
            label="MLP Val Accuracy",
            marker="s",
            markersize=3,
        )
        plt.plot(
            cnn_history.history["val_accuracy"],
            label="CNN Val Accuracy",
            marker="x",
            markersize=3,
        )
        plt.xlabel("Epoch")
        plt.ylabel("Validation Accuracy")
        plt.legend()
        plt.title(f"Validation Accuracy Zoom ({num_classes} classes)")

        plt.tight_layout()
        plt.savefig(os.path.join(class_dir, "comparison.png"), dpi=150)
        plt.close()


def run_comparison(epochs=30, batch_size=250, output_dir="compare_mlp_and_cnn_results"):
    """Run complete comparison between MLP and CNN models."""
    os.makedirs(output_dir, exist_ok=True)

    # Load and preprocess data
    cvt = converter.Converter()
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    # Apply pooling and binarization
    X_train_pooled, X_test_pooled = cvt.pooling_4x4(X_train, X_test)
    X_train_bin, X_test_bin = cvt.binarize(X_train_pooled, X_test_pooled)

    # Results storage
    results = {
        "mlp_test_accuracy": [],
        "cnn_test_accuracy": [],
        "mlp_parameters": [],
        "cnn_parameters": [],
        "class_counts": [],
    }

    # Test for each class count from 2 to 10
    for num_classes in range(2, 11):
        print(f"\n{'='*60}")
        print(f"Training models for {num_classes} classes")
        print(f"{'='*60}")

        # Extract labels for num_classes
        label_list = list(range(num_classes))
        (X_train_labels, Y_train_labels), (X_test_labels, Y_test_labels) = cvt.extract_labels(
            label_list, X_train_bin, Y_train, X_test_bin, Y_test
        )

        # Prepare for MLP
        X_train_mlp = X_train_labels.reshape(-1, 49).astype("float32") / 255.0
        X_test_mlp = X_test_labels.reshape(-1, 49).astype("float32") / 255.0

        # Prepare for CNN
        X_train_cnn = X_train_labels.reshape(-1, 7, 7, 1).astype("float32") / 255.0
        X_test_cnn = X_test_labels.reshape(-1, 7, 7, 1).astype("float32") / 255.0

        # Convert to categorical
        Y_train_cat = to_categorical(Y_train_labels, num_classes=num_classes)
        Y_test_cat = to_categorical(Y_test_labels, num_classes=num_classes)

        # Train MLP
        print(f"Training MLP for {num_classes} classes...")
        mlp_model = build_mlp_model(input_shape=(49,), num_classes=num_classes)
        mlp_history, mlp_test_results = train_model(
            mlp_model,
            X_train_mlp,
            Y_train_cat,
            X_test_mlp,
            Y_test_cat,
            epochs=epochs,
            batch_size=batch_size,
        )
        mlp_accuracy = mlp_test_results[1]
        mlp_params = count_parameters(mlp_model)
        print(f"MLP - Test Accuracy: {mlp_accuracy:.4f}, Parameters: {mlp_params}")

        # Train CNN
        print(f"Training CNN for {num_classes} classes...")
        cnn_model = build_cnn_model(input_shape=(7, 7, 1), num_classes=num_classes)
        cnn_history, cnn_test_results = train_model(
            cnn_model,
            X_train_cnn,
            Y_train_cat,
            X_test_cnn,
            Y_test_cat,
            epochs=epochs,
            batch_size=batch_size,
        )
        cnn_accuracy = cnn_test_results[1]
        cnn_params = count_parameters(cnn_model)
        print(f"CNN - Test Accuracy: {cnn_accuracy:.4f}, Parameters: {cnn_params}")

        # Store results
        results["mlp_test_accuracy"].append(mlp_accuracy)
        results["cnn_test_accuracy"].append(cnn_accuracy)
        results["mlp_parameters"].append(mlp_params)
        results["cnn_parameters"].append(cnn_params)
        results["class_counts"].append(num_classes)

        # Save training history plots for this class count
        save_training_history_plots(
            {"mlp": mlp_history, "cnn": cnn_history},
            num_classes,
            output_dir=output_dir,
        )

        # Save model summaries for this class count
        save_model_summaries(
            mlp_model,
            cnn_model,
            num_classes,
            output_dir=output_dir,
        )

    # Save results to JSON
    results_json = {
        "mlp_test_accuracy": results["mlp_test_accuracy"],
        "cnn_test_accuracy": results["cnn_test_accuracy"],
        "mlp_parameters": results["mlp_parameters"],
        "cnn_parameters": results["cnn_parameters"],
        "class_counts": results["class_counts"],
    }
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results_json, f, indent=2)

    # Plot 1: MLP test accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(
        results["class_counts"], results["mlp_test_accuracy"], marker="o", linewidth=2, markersize=8
    )
    plt.xlabel("Number of Classes", fontsize=12)
    plt.ylabel("Test Accuracy", fontsize=12)
    plt.title("MLP Test Accuracy vs Number of Classes", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(results["class_counts"])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "test_accuracy_mlp.png"), dpi=150)
    plt.close()

    # Plot 2: CNN test accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(
        results["class_counts"], results["cnn_test_accuracy"], marker="s", linewidth=2, markersize=8
    )
    plt.xlabel("Number of Classes", fontsize=12)
    plt.ylabel("Test Accuracy", fontsize=12)
    plt.title("CNN Test Accuracy vs Number of Classes", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(results["class_counts"])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "test_accuracy_cnn.png"), dpi=150)
    plt.close()

    # Plot 3: Comparison of test accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(
        results["class_counts"],
        results["mlp_test_accuracy"],
        marker="o",
        linewidth=2,
        markersize=8,
        label="MLP",
    )
    plt.plot(
        results["class_counts"],
        results["cnn_test_accuracy"],
        marker="s",
        linewidth=2,
        markersize=8,
        label="CNN",
    )
    plt.xlabel("Number of Classes", fontsize=12)
    plt.ylabel("Test Accuracy", fontsize=12)
    plt.title("Test Accuracy Comparison: MLP vs CNN", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(results["class_counts"])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "test_accuracy_comparison.png"), dpi=150)
    plt.close()

    # Plot 4: Parameter count comparison
    plt.figure(figsize=(10, 6))
    plt.plot(
        results["class_counts"],
        results["mlp_parameters"],
        marker="o",
        linewidth=2,
        markersize=8,
        label="MLP",
    )
    plt.plot(
        results["class_counts"],
        results["cnn_parameters"],
        marker="s",
        linewidth=2,
        markersize=8,
        label="CNN",
    )
    plt.xlabel("Number of Classes", fontsize=12)
    plt.ylabel("Number of Parameters", fontsize=12)
    plt.title("Model Parameter Count vs Number of Classes", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(results["class_counts"])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "parameter_count_comparison.png"), dpi=150)
    plt.close()

    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"Results saved to {output_dir}/")
    print(f"{'='*60}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Compare MLP and CNN models on binary MNIST.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=250, help="Training batch size")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="compare_mlp_and_cnn_results",
        help="Directory to save results",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_comparison(
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
