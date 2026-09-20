import numpy as np


class VLayer:
    """Vectorized layer: holds W (n_out, n_in) and b (n_out,) and handles a whole batch at once.

    Same activations and update rule as Layer/Neuron, except that the gradient is averaged
    over the batch and errors are propagated back with the weights *before* the update.
    """

    def __init__(self, num_neurons, num_inputs_per_neuron, rng=None, activation="relu"):
        rng = rng if rng is not None else np.random.default_rng()
        if activation not in ("relu", "sigmoid"):
            raise ValueError(f"Unknown activation: {activation}")
        self.activation = activation
        self.W = rng.normal(size=(num_neurons, num_inputs_per_neuron))
        self.b = rng.uniform(size=num_neurons)

    def forward(self, inputs):
        # inputs: (n_in,) or (B, n_in)
        self.inputs = np.asarray(inputs, dtype=float)
        self.sum = self.inputs @ self.W.T + self.b
        if self.activation == "relu":
            self.output = np.maximum(self.sum, 0.0)
        else:
            self.output = 1 / (1 + np.exp(-np.clip(self.sum, -500, 500)))
        return self.output

    def activation_derivative(self):
        if self.activation == "relu":
            return (self.sum > 0).astype(float)
        return self.output * (1 - self.output)

    def backward(self, errors, learning_rate, l2_lambda=0.0):
        # errors: (B, n_out) -> returns (B, n_in)
        delta = errors * self.activation_derivative()
        batch = delta.shape[0]
        prev_errors = delta @ self.W
        grad_w = delta.T @ self.inputs / batch
        self.W += learning_rate * (grad_w - l2_lambda * self.W)
        self.b += learning_rate * delta.mean(axis=0)
        return prev_errors

    def weights(self):
        return [(w, b) for w, b in zip(self.W, self.b)]
