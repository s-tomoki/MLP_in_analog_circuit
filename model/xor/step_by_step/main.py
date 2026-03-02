import matplotlib.pyplot as plt
import numpy as np
from neuralnetwork import NeuralNetwork

inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = np.array([[0], [1], [1], [0]])

layers = [2, 2, 1]
nn = NeuralNetwork(layers, 0.01, 50_000)
nn.train(inputs, outputs)

predicted_output = np.array([nn.predict(x) for x in inputs])
print("Predicted Output:\n", predicted_output)

weights = nn.weights()
print("Trained weights:\n", weights)


# Round the predicted output to get binary predictions
predicted_output_binary = np.round(predicted_output)

# Plot the decision boundary
x_min, x_max = inputs[:, 0].min() - 0.5, inputs[:, 0].max() + 0.5
y_min, y_max = inputs[:, 1].min() - 0.5, inputs[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = np.array([nn.predict(np.array([x, y])) for x, y in zip(xx.ravel(), yy.ravel())])
Z = np.round(Z.reshape(xx.shape))

plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
plt.scatter(inputs[:, 0], inputs[:, 1], c=outputs.ravel(), cmap=plt.cm.Spectral)
plt.title("XOR Neural Network Decision Boundary")
plt.xlabel("Input 1")
plt.ylabel("Input 2")
plt.show()

# Print the accuracy
accuracy = np.mean(predicted_output_binary.ravel() == outputs.ravel())
print("Accuracy:", accuracy)
