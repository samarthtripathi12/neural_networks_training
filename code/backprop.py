# backprop.py
import numpy as np
from forward import forward_pass, relu_derivative, sigmoid_derivative

# --------------------------
# LOSS FUNCTION
# --------------------------
def compute_loss(A2, Y):
    """
    Binary cross-entropy loss
    A2: predictions (num_samples x 1)
    Y: true labels (num_samples x 1)
    """
    m = Y.shape[0]
    # Clip predictions to avoid log(0)
    A2 = np.clip(A2, 1e-8, 1 - 1e-8)
    loss = - (1/m) * np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2))
    return loss

# --------------------------
# BACKPROP FUNCTION
# --------------------------
def backpropagation(parameters, cache, Y, learning_rate=0.01):
    """
    Performs backpropagation and updates parameters
    parameters: dictionary with W1, b1, W2, b2
    cache: dictionary from forward_pass
    Y: true labels
    learning_rate: gradient descent step size
    """
    m = Y.shape[0]

    W1, b1 = parameters['W1'], parameters['b1']
    W2, b2 = parameters['W2'], parameters['b2']
    
    A1, A2, X, Z1, Z2 = cache['A1'], cache['A2'], cache['X'], cache['Z1'], cache['Z2']

    # Output layer gradients
    dZ2 = A2 - Y  # Binary cross-entropy derivative for sigmoid output
    dW2 = (A1.T @ dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    # Hidden layer gradients
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = (X.T @ dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    # Update weights and biases
    parameters['W1'] -= learning_rate * dW1
    parameters['b1'] -= learning_rate * db1
    parameters['W2'] -= learning_rate * dW2
    parameters['b2'] -= learning_rate * db2

    return parameters

# --------------------------
# TEST BACKPROP
# --------------------------
if __name__ == "__main__":
    np.random.seed(42)

    # Sample input (4 samples, 2 features)
    X = np.random.randn(4, 2)
    Y = np.array([[1], [0], [1], [0]])  # Sample labels

    # Initialize parameters
    parameters = {
        "W1": np.random.randn(2, 4) * 0.01,
        "b1": np.zeros((1, 4)),
        "W2": np.random.randn(4, 1) * 0.01,
        "b2": np.zeros((1, 1))
    }

    # Forward pass
    A2, cache = forward_pass(X, parameters)
    print("Loss before backprop:", compute_loss(A2, Y))

    # Backprop
    parameters = backpropagation(parameters, cache, Y, learning_rate=0.1)

    # Forward pass again
    A2_new, _ = forward_pass(X, parameters)
    print("Loss after one backprop step:", compute_loss(A2_new, Y))
