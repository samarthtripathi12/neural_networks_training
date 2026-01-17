import numpy as np
import matplotlib.pyplot as plt
from forward import forward_pass
from backprop import backpropagation, compute_loss
import os

# --------------------------
# CREATE PLOTS FOLDER
# --------------------------
os.makedirs("../plots", exist_ok=True)

# --------------------------
# Dataset
# --------------------------
np.random.seed(42)
num_samples = 200
X = np.random.randn(num_samples, 2)
Y = np.array([[0] if x[0]*x[1] < 0 else [1] for x in X])

# --------------------------
# Hyperparameters to test
# --------------------------
learning_rates = [0.01, 0.05, 0.1]
epochs = 300
hidden_size = 4
input_size = 2
output_size = 1

# Store loss history for each LR
loss_histories = {}

for lr in learning_rates:
    # Initialize parameters
    parameters = {
        "W1": np.random.randn(input_size, hidden_size) * 0.01,
        "b1": np.zeros((1, hidden_size)),
        "W2": np.random.randn(hidden_size, output_size) * 0.01,
        "b2": np.zeros((1, output_size))
    }
    
    loss_history = []
    
    for epoch in range(epochs):
        # Forward pass
        A2, cache = forward_pass(X, parameters)
        # Loss
        loss = compute_loss(A2, Y)
        loss_history.append(loss)
        # Backprop
        parameters = backpropagation(parameters, cache, Y, lr)
    
    loss_histories[lr] = loss_history

# --------------------------
# Plot loss curves for different LRs
# --------------------------
plt.figure(figsize=(6,4))
for lr, losses in loss_histories.items():
    plt.plot(losses, label=f'LR={lr}')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss vs Epochs for different Learning Rates')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../plots/loss_comparison_lr.png")
plt.show()
