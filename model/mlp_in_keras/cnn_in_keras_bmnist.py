import argparse
import json
import os

import converter
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical


def build_cnn_model(input_shape, num_classes):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(
        Conv2D(
            filters=1,
            kernel_size=(3, 3),
            strides=2,
            padding="same",
            activation="relu",
        )
    )
    model.add(Flatten())
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )
    return model


def save_training_history(history, dirname):
    os.makedirs(dirname, exist_ok=True)

    # Plot training curves
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Model Loss")

    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Model Accuracy")

    plt.tight_layout()
    plt.savefig(os.path.join(dirname, "training_history.png"), dpi=150)
    plt.close()

    history_dict = {
        "loss": [float(x) for x in history.history["loss"]],
        "val_loss": [float(x) for x in history.history["val_loss"]],
        "accuracy": [float(x) for x in history.history["accuracy"]],
        "val_accuracy": [float(x) for x in history.history["val_accuracy"]],
    }
    with open(os.path.join(dirname, "training_history.json"), "w") as f:
        json.dump(history_dict, f)


def save_model_weights(model, dirname):
    os.makedirs(dirname, exist_ok=True)
    weights_dict = {}
    for i, layer in enumerate(model.layers):
        layer_weights = layer.get_weights()
        if not layer_weights:
            continue
        if len(layer_weights) == 2:
            w, b = layer_weights
            weights_dict[f"layer_{i}_weights"] = w
            weights_dict[f"layer_{i}_bias"] = b
        elif len(layer_weights) == 1:
            weights_dict[f"layer_{i}_weights"] = layer_weights[0]

    np.savez(os.path.join(dirname, "model_weights.npz"), **weights_dict)
    for key, value in weights_dict.items():
        np.savetxt(
            os.path.join(dirname, f"{key}.csv"), value.reshape(value.shape[0], -1), delimiter=","
        )


def save_model_summary(model, dirname):
    os.makedirs(dirname, exist_ok=True)
    with open(os.path.join(dirname, "model_summary.txt"), "w") as f:
        model.summary(print_fn=lambda line: f.write(line + "\n"))


def save_confusion_matrix(confusion, dirname):
    os.makedirs(dirname, exist_ok=True)
    np.savetxt(
        os.path.join(dirname, "confusion_matrix.csv"),
        confusion,
        delimiter=",",
        fmt="%d",
    )


def prepare_dataset(X_train, X_test, Y_train, Y_test, labels, cvt):
    X_train_pooled, X_test_pooled = cvt.pooling_4x4(X_train, X_test)
    X_train_bin, X_test_bin = cvt.binarize(X_train_pooled, X_test_pooled)

    if labels is not None:
        (X_train_bin, Y_train), (X_test_bin, Y_test) = cvt.extract_labels(
            labels, X_train_bin, Y_train, X_test_bin, Y_test
        )

    X_train_bin = X_train_bin.reshape(-1, 7, 7, 1).astype("float32") / 255.0
    X_test_bin = X_test_bin.reshape(-1, 7, 7, 1).astype("float32") / 255.0

    num_target_classes = len(labels) if labels is not None else 10
    Y_train_cat = to_categorical(Y_train, num_classes=num_target_classes)
    Y_test_cat = to_categorical(Y_test, num_classes=num_target_classes)

    return X_train_bin, X_test_bin, Y_train_cat, Y_test_cat, Y_test


def run_experiment(num_classes, labels, epochs, batch_size, dirname):
    os.makedirs(dirname, exist_ok=True)
    cvt = converter.Converter()

    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    X_train, X_test, Y_train_cat, Y_test_cat, Y_test_raw = prepare_dataset(
        X_train, X_test, Y_train, Y_test, labels, cvt
    )

    input_shape = (7, 7, 1)
    model = build_cnn_model(input_shape, num_classes)
    history = model.fit(
        X_train,
        Y_train_cat,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1,
    )

    test_results = model.evaluate(X_test, Y_test_cat, verbose=0)
    print(f"[{dirname}] Test loss: {test_results[0]:.4f}, Test accuracy: {test_results[1]:.4f}")

    save_training_history(history, dirname)
    save_model_weights(model, dirname)
    save_model_summary(model, dirname)

    predictions = model.predict(X_test, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)
    confusion = tf.math.confusion_matrix(
        Y_test_raw, predicted_labels, num_classes=num_classes
    ).numpy()
    save_confusion_matrix(confusion, dirname)

    return test_results


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate CNN on compressed MNIST.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=250, help="Training batch size")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="cnn_model_analysis",
        help="Base directory for saving outputs",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    settings = [
        {
            "num_classes": 2,
            "labels": list(range(2)),
            "dirname": os.path.join(args.output_dir, "bmnist_2_classes"),
        },
        {
            "num_classes": 5,
            "labels": list(range(5)),
            "dirname": os.path.join(args.output_dir, "bmnist_5_classes"),
        },
        {
            "num_classes": 10,
            "labels": None,
            "dirname": os.path.join(args.output_dir, "bmnist_10_classes"),
        },
    ]

    for config in settings:
        print(f"Running experiment for {config['num_classes']} classes -> {config['dirname']}")
        run_experiment(
            num_classes=config["num_classes"],
            labels=config["labels"],
            epochs=args.epochs,
            batch_size=args.batch_size,
            dirname=config["dirname"],
        )


if __name__ == "__main__":
    main()
