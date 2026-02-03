# Ref:
# https://machinecurve.com/index.php/2019/07/27/how-to-create-a-basic-mlp-classifier-with-the-keras-sequential-api

import os

import converter
import numpy as np
import trainer
from PIL import Image
from tensorflow.keras.datasets import mnist


def visualize_dataset(X, Y, dirname="visualizations"):
    # Create output directory for visualizations
    os.makedirs(dirname, exist_ok=True)

    # Visualize and save training samples
    x_width = int(np.sqrt(X.shape[1]))
    for i in range(min(10, X.shape[0])):
        label_index = np.argmax(Y == i)
        print(f"Saving visualization for sample {label_index} with label {i}")
        # Reshape back to len(x_width) x len(x_width)
        img_array = X[label_index].reshape(x_width, x_width).astype(np.uint8)
        img = Image.fromarray(img_array, mode="L")
        img.save(f"{dirname}/train_sample_label={i}_index={label_index}.png")

    print(f"Visualizations saved to {dirname}/ directory")
    return None


def training(
    X_train,
    X_test,
    Y_train,
    Y_test,
    num_classes=10,
    layers=(
        8,
        2,
    ),
    epochs=100,
    batch_size=250,
    dirname="model_analysis",
):
    train = trainer.Trainer(X_train, Y_train, X_test, Y_test, num_classes=num_classes)
    model, test_results, history = train.compile_and_train(
        layers=layers, epochs=epochs, batch_size=batch_size, dirname=dirname
    )
    train.save_training_history(model, history, dirname=dirname)
    train.save_model_weights(model, dirname=dirname)
    return None


def main():
    # Load the data
    cvt = converter.Converter()
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    print(f"Original X_train shape: {X_train.shape}")
    print(f"Original Y_train shape: {Y_train.shape}")

    # Case 1:
    #   Input: Pooling 4x4 (7x7 images, grayscale)
    #   Category: 10 classes (0-9)
    #   Layers: (10, 8, 2)
    (X_train_pooled, X_test_pooled) = cvt.pooling_4x4(X_train, X_test)
    visualize_dataset(X_train_pooled, Y_train, dirname="visualizations_pooled")
    training(
        X_train_pooled, X_test_pooled, Y_train, Y_test, dirname="model_analysis_pooled"
    )

    # Case 2:
    #   Input: Pooling 4x4 and binarization (7x7 images, binary)
    #   Category: 10 classes (0-9)
    #   Layers: (10, 8, 2)
    (X_train_pooled_bin, X_test_pooled_bin) = cvt.binarize(
        X_train_pooled, X_test_pooled
    )
    visualize_dataset(X_train_pooled_bin, Y_train, dirname="visualizations_pooled_bin")
    training(
        X_train_pooled_bin,
        X_test_pooled_bin,
        Y_train,
        Y_test,
        dirname="model_analysis_pooled_bin",
    )

    # Case 3:
    #   Input: Pooling 4x4 (7x7 images, grayscale)
    #   Category: 2 classes (0 and 1)
    #   Layers: (2, 2)
    (
        (X_train_pooled_01, Y_train_pooled_01),
        (X_test_pooled_01, Y_test_01),
    ) = cvt.extract_labels([0, 1], X_train_pooled, Y_train, X_test_pooled, Y_test)
    print(f"Number of training samples for digits 0 and 1: {len(Y_train_pooled_01)}")
    print(f"Number of test samples for digits 0 and 1: {len(Y_test_01)}")
    training(
        X_train_pooled_01,
        X_test_pooled_01,
        Y_train_pooled_01,
        Y_test_01,
        num_classes=2,
        layers=(2,),
        dirname="model_analysis_pooled_01",
    )

    # Case 4:
    #   Input: Pooling 4x4 and binarization (7x7 images, binary)
    #   Category: 2 classes (0 and 1)
    #   Layers: (2, 2)
    (X_train_pooled_bin_01, X_test_pooled_bin_01) = cvt.binarize(
        X_train_pooled_01, X_test_pooled_01
    )
    training(
        X_train_pooled_bin_01,
        X_test_pooled_bin_01,
        Y_train_pooled_01,
        Y_test_01,
        num_classes=2,
        layers=(2,),
        dirname="model_analysis_pooled_bin_01",
    )


if __name__ == "__main__":
    main()
