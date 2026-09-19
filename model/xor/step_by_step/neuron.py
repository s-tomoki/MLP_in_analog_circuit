import numpy as np


class Neuron:
    def __init__(self, num_inputs, rng=None):
        rng = rng if rng is not None else np.random.default_rng()
        # He initialization (suited for ReLU activations)
        self.weights = rng.normal(scale=(2 / num_inputs) ** 0.5, size=num_inputs)
        self.bias = rng.uniform()

    def activate(self, inputs):
        self.inputs = inputs
        self.sum = np.dot(inputs, self.weights) + self.bias
        self.output = self.relu(self.sum)
        return self.output

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
