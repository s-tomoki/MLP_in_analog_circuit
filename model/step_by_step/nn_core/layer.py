import numpy as np

from .neuron import Neuron


class Layer:
    def __init__(self, num_neurons, num_inputs_per_neuron, rng=None, activation="relu"):
        self.neurons = [
            Neuron(num_inputs_per_neuron, rng=rng, activation=activation)
            for _ in range(num_neurons)
        ]

    def forward(self, inputs):
        return np.array([neuron.activate(inputs) for neuron in self.neurons])

    def backward(self, errors, learning_rate, l2_lambda=0.0):
        deltas = []
        for i, neuron in enumerate(self.neurons):
            delta = errors[i] * neuron.activation_derivative()
            neuron.update_weights(delta, learning_rate, l2_lambda)
            deltas.append(delta)
        return np.dot(np.array([neuron.weights for neuron in self.neurons]).T, deltas)

    def weights(self):
        return [(neuron.weights, neuron.bias) for neuron in self.neurons]
