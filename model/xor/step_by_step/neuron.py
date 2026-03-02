import numpy as np


class Neuron:
    def __init__(self, num_inputs):
        # self.weights = np.random.uniform(size=num_inputs)
        self.weights = np.random.normal(size=num_inputs, scale=(2 / num_inputs) ** 0.5)
        self.weights = np.random.normal(size=num_inputs)
        self.bias = np.random.uniform()

    def activate(self, inputs):
        self.inputs = inputs
        # self.output = self.sigmoid(np.dot(inputs, self.weights) + self.bias)
        self.sum = np.dot(inputs, self.weights) + self.bias
        self.output = self.relu(self.sum)
        return self.output

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self):
        return self.output * (1 - self.output)

    def relu(self, x):
        if x > 0:
            return x
        else:
            return 0

    def relu_derivative(self):
        if self.sum > 0:
            return 1
        else:
            return 0

    def update_weights(self, delta, learning_rate):
        self.weights += learning_rate * delta * self.inputs
        self.bias += learning_rate * delta
