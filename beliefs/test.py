import numpy as np

def linear_function(inputs, weights):
    return np.dot(inputs, weights)

def gradient(inputs, weights):
    return inputs

def project_onto_simplex(v, z=1):
    n_features = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w

def minimize_linear_function_with_constraint(initial_weights, inputs, learning_rate=0.01, num_iterations=1000):
    weights = initial_weights
    for _ in range(num_iterations):
        grad = gradient(inputs, weights)
        weights -= learning_rate * grad
        weights = project_onto_simplex(weights)
        print(weights)
    return weights

# Example usage:
k = 3  # Number of inputs
inputs = np.random.rand(k)  # Random input values
initial_weights = np.random.rand(k)  # Initial guess for weights

optimized_weights = minimize_linear_function_with_constraint(initial_weights, inputs)
print("Optimized weights:", optimized_weights)