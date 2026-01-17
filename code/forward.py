# forward.py
import numpy as np

# --------------------------
# ACTIVATION FUNCTIONS
# --------------------------
def sigmoid(z):
    """Sigmoid activation"""
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    """Derivative of sigmoid"""
    return sigmoid(z) * (1 - sigmoid(z))

def relu(z):
    """ReLU activation"""
    return np.maximum(0, z)

def relu_derivative(z):
    """Derivative of ReLU"""
    return (z > 0).astype(float)

# --------------------------
# FORWARD PASS FUNCTIONS
# --------------------------
def forward_pass(X, parameters):
    """
    Performs forward propagation through a 2-layer network
    X: input data (num_samples x input_size)
    parameters: dictionary with weights and biases
    Returns:
        cache: intermediate values needed for backprop
        A2: output of the network
    """
    W1, b1 = parameters['W1'], parameters['b1']
    W2, b2 = parameters['W2'], parameters['b2']

    # Layer 1
    Z1 = X.dot(W1) + b1
    A1 = relu(Z1)

    # Layer 2 (Output)
    Z2 = A1.dot(W2) + b2
    A2 = sigmoid(Z2)  # For classification (binary)

    # Store values for backpropagation
    cache = {
        "X": X,
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2
    }

    return A2, cache

# --------------------------
# TEST FORWARD PASS
# --------------------------
if __name__ == "__main__":
    np.random.seed(42)
    
    # Example input (4 samples, 2 features)
    X = np.random.randn(4, 2)
    
    # Example network parameters
    parameters = {
        "W1": np.random.randn(2, 4) * 0.01,  # 2 inputs -> 4 neurons
        "b1": np.zeros((1, 4)),
        "W2": np.random.randn(4, 1) * 0.01,  # 4 neurons -> 1 output
        "b2": np.zeros((1, 1))
    }

    # Forward pass
    output, _ = forward_pass(X, parameters)
    print("Forward pass output:\n", output)
