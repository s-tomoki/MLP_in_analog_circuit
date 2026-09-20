import numpy as np

from .layer import Layer
from .vlayer import VLayer


class NeuralNetwork:
    def __init__(
        self,
        layers,
        learning_rate=0.1,
        epochs=10_000,
        l2_lambda=0.0,
        seed=None,
        activation="relu",
        output_activation=None,
        batch_size=1,
        shuffle=False,
    ):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2_lambda = l2_lambda
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.loss_history = []

        # Hidden layers use `activation`; the last layer uses `output_activation` (default: same).
        output_activation = output_activation or activation
        self.activations = [activation] * (len(layers) - 2) + [output_activation]

        self.rng = np.random.default_rng(seed)
        # batch_size == 1 keeps the original per-neuron scalar path; larger batches are vectorized.
        layer_cls = Layer if batch_size == 1 else VLayer
        self.layers = [
            layer_cls(layers[i + 1], layers[i], rng=self.rng, activation=self.activations[i])
            for i in range(len(layers) - 1)
        ]

    def train(self, inputs, outputs, verbose=True, log_every=1000):
        inputs = np.asarray(inputs)
        outputs = np.asarray(outputs)
        n = len(inputs)
        for epoch in range(self.epochs):
            order = self.rng.permutation(n) if self.shuffle else np.arange(n)
            if self.batch_size == 1:
                total_error = self._train_epoch_scalar(inputs, outputs, order)
            else:
                total_error = self._train_epoch_batch(inputs, outputs, order)

            mse = total_error / n
            self.loss_history.append(mse)

            if verbose and epoch % log_every == 0:
                print(f"Epoch {epoch}, MSE: {mse}")

    def _train_epoch_scalar(self, inputs, outputs, order):
        total_error = 0
        for idx in order:
            x, y = inputs[idx], outputs[idx]
            # Forward pass
            activations = [x]
            for layer in self.layers:
                activations.append(layer.forward(activations[-1]))

            # Calculate error
            output_errors = y - activations[-1]
            total_error += np.sum(output_errors**2)

            # Backward pass
            errors = output_errors
            for i in reversed(range(len(self.layers))):
                errors = self.layers[i].backward(errors, self.learning_rate, self.l2_lambda)
        return total_error

    def _train_epoch_batch(self, inputs, outputs, order):
        total_error = 0
        for start in range(0, len(order), self.batch_size):
            idx = order[start : start + self.batch_size]
            a = inputs[idx]
            for layer in self.layers:
                a = layer.forward(a)

            errors = outputs[idx] - a
            total_error += np.sum(errors**2)

            for i in reversed(range(len(self.layers))):
                errors = self.layers[i].backward(errors, self.learning_rate, self.l2_lambda)
        return total_error

    def predict(self, inputs):
        activations = inputs
        for layer in self.layers:
            activations = layer.forward(activations)
        return activations

    def predict_batch(self, inputs):
        """Vectorized forward pass for (N, n_in) inputs, usable with either layer type."""
        a = np.asarray(inputs, dtype=float)
        for layer_weights, act in zip(self.weights(), self.activations):
            w = np.array([w for w, _ in layer_weights])
            b = np.array([b for _, b in layer_weights])
            z = a @ w.T + b
            a = np.maximum(z, 0.0) if act == "relu" else 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        return a

    def weights(self):
        return [layer.weights() for layer in self.layers]
