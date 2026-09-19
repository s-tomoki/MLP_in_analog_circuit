import numpy as np
from layer import Layer


class NeuralNetwork:
    def __init__(self, layers, learning_rate=0.1, epochs=10_000, l2_lambda=0.0, seed=None):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2_lambda = l2_lambda
        self.loss_history = []

        rng = np.random.default_rng(seed)
        self.layers = [Layer(layers[i + 1], layers[i], rng=rng) for i in range(len(layers) - 1)]

    def train(self, inputs, outputs, verbose=True):
        for epoch in range(self.epochs):
            total_error = 0
            for x, y in zip(inputs, outputs):
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

            mse = total_error / len(inputs)
            self.loss_history.append(mse)

            if verbose and epoch % 1000 == 0:
                print(f"Epoch {epoch}, MSE: {mse}")

    def predict(self, inputs):
        activations = inputs
        for layer in self.layers:
            activations = layer.forward(activations)
        return activations

    def weights(self):
        return [layer.weights() for layer in self.layers]
