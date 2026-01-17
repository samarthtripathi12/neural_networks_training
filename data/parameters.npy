import numpy as np

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def forward_pass(X, parameters):
    W1, b1 = parameters['W1'], parameters['b1']
    W2, b2 = parameters['W2'], parameters['b2']
    
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    
    cache = {'X': X, 'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2}
    return A2, cache


import numpy as np

def compute_loss(A2, Y):
    m = Y.shape[0]
    # binary cross-entropy
    return -np.mean(Y*np.log(A2 + 1e-8) + (1-Y)*np.log(1-A2 + 1e-8))

def backpropagation(parameters, cache, Y, learning_rate):
    X, A1, A2 = cache['X'], cache['A1'], cache['A2']
    m = X.shape[0]

    dZ2 = A2 - Y
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    dA1 = np.dot(dZ2, parameters['W2'].T)
    dZ1 = dA1 * A1 * (1-A1)  # sigmoid derivative
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    parameters['W1'] -= learning_rate * dW1
    parameters['b1'] -= learning_rate * db1
    parameters['W2'] -= learning_rate * dW2
    parameters['b2'] -= learning_rate * db2

    return parameters

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from forward import forward_pass
from backprop import backpropagation, compute_loss

# --------------------------
# Create folders
# --------------------------
os.makedirs("../plots", exist_ok=True)
os.makedirs("../gifs", exist_ok=True)

# --------------------------
# Dataset (XOR)
# --------------------------
np.random.seed(42)
num_samples = 200
X = np.random.randn(num_samples, 2)
Y = np.array([[0] if x[0]*x[1] < 0 else [1] for x in X])

# --------------------------
# Initialize parameters
# --------------------------
input_size = 2
hidden_size = 4
output_size = 1

parameters = {
    "W1": np.random.randn(input_size, hidden_size) * 0.01,
    "b1": np.zeros((1, hidden_size)),
    "W2": np.random.randn(hidden_size, output_size) * 0.01,
    "b2": np.zeros((1, output_size))
}

# --------------------------
# Hyperparameters
# --------------------------
epochs = 500
learning_rate = 0.1

loss_history = []
training_frames = []
decision_boundary_frames = []

# --------------------------
# Training loop
# --------------------------
for epoch in range(epochs):
    # Forward pass
    A2, cache = forward_pass(X, parameters)
    # Compute loss
    loss = compute_loss(A2, Y)
    loss_history.append(loss)
    # Backprop
    parameters = backpropagation(parameters, cache, Y, learning_rate)

    # Save plots for GIF every 20 epochs
    if epoch % 20 == 0 or epoch == epochs-1:
        # ---------------- Training Progress Frame ----------------
        plt.figure(figsize=(5,5))
        plt.scatter(X[Y[:,0]==0,0], X[Y[:,0]==0,1], color='red', label='Class 0')
        plt.scatter(X[Y[:,0]==1,0], X[Y[:,0]==1,1], color='blue', label='Class 1')
        pred_labels = (A2 > 0.5).astype(int)
        plt.scatter(X[pred_labels[:,0]==1,0], X[pred_labels[:,0]==1,1],
                    facecolors='none', edgecolors='yellow', s=80, label='Pred 1')
        plt.title(f'Epoch {epoch}, Loss={loss:.4f}')
        plt.legend()
        plt.xlim(-4,4)
        plt.ylim(-4,4)
        plt.tight_layout()
        path = f"../plots/temp_epoch_{epoch}.png"
        plt.savefig(path)
        plt.close()
        training_frames.append(path)

        # ---------------- Decision Boundary Frame ----------------
        h = 0.01
        x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
        y_min, y_max = X[:,1].min()-1, X[:,1].max()+1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        grid = np.c_[xx.ravel(), yy.ravel()]
        Z, _ = forward_pass(grid, parameters)
        Z = (Z > 0.5).astype(int).reshape(xx.shape)

        plt.figure(figsize=(5,5))
        plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
        plt.scatter(X[Y[:,0]==0,0], X[Y[:,0]==0,1], color='red', label='Class 0')
        plt.scatter(X[Y[:,0]==1,0], X[Y[:,0]==1,1], color='blue', label='Class 1')
        plt.title(f'Decision Boundary Epoch {epoch}')
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.tight_layout()
        path_db = f"../plots/db_epoch_{epoch}.png"
        plt.savefig(path_db)
        plt.close()
        decision_boundary_frames.append(path_db)

# --------------------------
# Save final loss curve
# --------------------------
plt.figure()
plt.plot(loss_history, label='Loss')
plt.xlabel("Epochs")
plt.ylabel("Binary Cross-Entropy Loss")
plt.title("Loss Convergence")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("../plots/loss_curve.png")
plt.close()

# --------------------------
# Save final parameters
# --------------------------
np.save("../parameters.npy", parameters)

# --------------------------
# Create GIFs
# --------------------------
# Training Progress GIF
images = [Image.open(f) for f in training_frames]
images[0].save('../gifs/training_progress.gif',
               save_all=True, append_images=images[1:], duration=200, loop=0)

# Decision Boundary GIF
images_db = [Image.open(f) for f in decision_boundary_frames]
images_db[0].save('../gifs/decision_boundary_evolution.gif',
                  save_all=True, append_images=images_db[1:], duration=200, loop=0)

# --------------------------
# Print final accuracy
# --------------------------
A2, _ = forward_pass(X, parameters)
accuracy = np.mean((A2 > 0.5) == Y)
print(f"Training complete! Final loss: {loss_history[-1]:.4f}")
print(f"Final accuracy: {accuracy*100:.2f}%")
print("Plots saved in 'plots/' folder, GIFs saved in 'gifs/' folder, parameters saved in 'parameters.npy'")


