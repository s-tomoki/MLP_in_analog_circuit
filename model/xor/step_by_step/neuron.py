import numpy as np


class Neuron:
    def __init__(self, num_inputs, rng=None, activation="relu"):
        rng = rng if rng is not None else np.random.default_rng()
        self.activation = activation
        if activation == "relu":
            # He initialization (suited for ReLU activations)
            self.weights = rng.normal(scale=(2 / num_inputs) ** 0.5, size=num_inputs)
        elif activation == "sigmoid":
            # Xavier/Glorot initialization (suited for sigmoid activations)
            self.weights = rng.normal(scale=(1 / num_inputs) ** 0.5, size=num_inputs)
        else:
            raise ValueError(f"Unknown activation: {activation}")
        self.bias = rng.uniform()

    def activate(self, inputs):
        self.inputs = inputs
        self.sum = np.dot(inputs, self.weights) + self.bias
        if self.activation == "relu":
            self.output = self.relu(self.sum)
        else:
            self.output = self.sigmoid(self.sum)
        return self.output

    def activation_derivative(self):
        if self.activation == "relu":
            return self.relu_derivative()
        return self.sigmoid_derivative()

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self):
        return self.output * (1 - self.output)

    def relu(self, x):
        return x if x > 0 else 0

    def relu_derivative(self):
        return 1 if self.sum > 0 else 0

    def update_weights(self, delta, learning_rate, l2_lambda=0.0):
        # Gradient step on the error term, plus L2 weight decay (bias is not regularized).
        self.weights += learning_rate * (delta * self.inputs - l2_lambda * self.weights)
        self.bias += learning_rate * delta
