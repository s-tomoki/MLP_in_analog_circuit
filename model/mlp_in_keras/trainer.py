import os

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tqdm.keras import TqdmCallback


class Trainer:
    def __init__(self, X_train, Y_train, X_test, Y_test, num_classes=10):
        self.X_train = X_train.astype("float32") / 255
        self.Y_train = to_categorical(Y_train, num_classes)
        self.X_test = X_test.astype("float32") / 255
        self.Y_test = to_categorical(Y_test, num_classes)
        # Convert target classes to categorical ones
        self.num_classes = num_classes

    def compile_and_train(
        self,
        layers=(
            8,
            2,
        ),
        epochs=100,
        batch_size=250,
        dirname="model_analysis",
    ):
        input_shape = (49,)

        # Create the model
        model = Sequential()
        model.add(Input(shape=input_shape))
        model.add(
            Dense(
                layers[0],
                activation="relu",
                kernel_regularizer=l2(0.01),
                bias_regularizer=l2(0.01),
            )
        )
        for i in range(1, len(layers)):
            model.add(
                Dense(
                    layers[i],
                    activation="relu",
                    kernel_regularizer=l2(0.01),
                    bias_regularizer=l2(0.01),
                )
            )
        model.add(Dense(self.num_classes, activation="softmax"))

        # Configure the model and start training
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        history = model.fit(
            self.X_train,
            self.Y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            validation_split=0.2,
            callbacks=[TqdmCallback(verbose=0)],
        )
        test_results = model.evaluate(self.X_test, self.Y_test, verbose=0)
        print(f"Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]}%")

        return (model, test_results, history)

    def save_training_history(self, model, history, dirname="model_analysis"):
        # Visualize model properties
        os.makedirs(dirname, exist_ok=True)

        # Plot training history
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
        plt.savefig(f"{dirname}/training_history.png", dpi=150)
        plt.close()

        # Print model summary
        #         model.summary()
        with open(f"{dirname}/model_summary.txt", "w") as f:
            model.summary(print_fn=lambda x: f.write(x + "\n"))
        print(f"Model analysis saved to {dirname}/ directory")
        return None

    def save_model_weights(self, model, dirname="model_analysis"):
        # Create output directory
        os.makedirs(dirname, exist_ok=True)

        # Save all weights and biases
        weights_dict = {}
        for i, layer in enumerate(model.layers):
            if hasattr(layer, "get_weights") and len(layer.get_weights()) > 0:
                w, b = layer.get_weights()
                weights_dict[f"layer_{i}_weights"] = w
                weights_dict[f"layer_{i}_bias"] = b

        # Save to NPZ
        np.savez(f"{dirname}/model_weights.npz", **weights_dict)
        print(f"Weights saved to {dirname}/model_weights.npz")

        # Save to CSV
        for key, value in weights_dict.items():
            np.savetxt(f"{dirname}/{key}.csv", value.reshape(value.shape[0], -1), delimiter=",")
        print(f"Weights saved to {dirname}/*.csv files")

        return None
