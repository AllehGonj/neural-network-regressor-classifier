import datasets
import numpy as np

ir = 0.1
lr = 0.02
weights = np.random.uniform(-ir, ir, (3, 1))
bias = np.random.uniform(-ir, ir, 1)

for i in range (60):
    outputs = np.dot(datasets.X_TRAIN, weights) + bias

    sigmoid = 1/(1 + np.exp(-outputs))
    sigmoidDerivate = sigmoid * (1-sigmoid)

    deltas = sigmoidDerivate - datasets.Y_TRAIN_DOT
    loss = (np.sum(deltas ** 2)/2)
    print("Loss:", loss)

    weights = weights - lr * np.dot(datasets.X_TRAIN.T, deltas)
    bias = bias - lr * np.sum(deltas)

print("\nWeights:\n", weights, end="\n\n")
print("Biases:\n", bias, end="\n\n")

pred = np.dot(datasets.X_TEST, weights)

sigmoid = 1/(1 + np.exp(-pred))
print("Prediction:\n", sigmoid, end="\n\n")
print("Test:\n", datasets.Y_TEST, end="\n\n")