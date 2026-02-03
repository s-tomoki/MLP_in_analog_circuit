import numpy as np
from neuron import Neuron


class Layer:
    def __init__(self, num_neurons, num_inputs_per_neuron):
        self.neurons = [Neuron(num_inputs_per_neuron) for _ in range(num_neurons)]

    def forward(self, inputs):
        return np.array([neuron.activate(inputs) for neuron in self.neurons])

    def backward(self, errors, learning_rate):
        deltas = []
        for i, neuron in enumerate(self.neurons):
            # delta = errors[i] * neuron.sigmoid_derivative()
            delta = errors[i] * neuron.relu_derivative()
            neuron.update_weights(delta, learning_rate)
            deltas.append(delta)
        return np.dot(np.array([neuron.weights for neuron in self.neurons]).T, deltas)

    def weights(self):
        weights = []
        for neuron in self.neurons:
            weights.append((neuron.weights, neuron.bias))
        return weights
