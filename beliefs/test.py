
import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def softmax_gradient(x, loss):
    s = softmax(x)
    jacob = s * (np.eye(len(x)) - s[:, np.newaxis])
    return jacob * loss[:, np.newaxis]

# Example usage:
np.random.seed(42)

# Initialize parameters
logits = np.random.randn(3)
learning_rate = 0.01
num_iterations = 100
print("Initial Logits:", logits)
loss = np.array([1,2,3], np.float32)

# Optimization loop
for i in range(num_iterations):
    grad = softmax_gradient(logits, loss)
    logits = logits + learning_rate * grad.sum(axis=0)  # Update parameters using the sum of gradients

# Display final parameters
print("Final Logits:", logits)
print("Final Softmax:", softmax(logits))