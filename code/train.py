# train.py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from forward import forward_pass
from backprop import backpropagation, compute_loss
import os

# --------------------------
# CREATE FOLDERS IF NOT EXIST
# --------------------------
os.makedirs("../plots", exist_ok=True)
os.makedirs("../gifs", exist_ok=True)

# --------------------------
# GENERATE SAMPLE DATA (2D CLASSIFICATION)
# --------------------------
np.random.seed(42)
num_samples = 200
X = np.random.randn(num_samples, 2)
# Labels: XOR problem
Y = np.array([[0] if x[0]*x[1] < 0 else [1] for x in X])

# --------------------------
# INITIALIZE PARAMETERS
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
# HYPERPARAMETERS
# --------------------------
epochs = 500
learning_rate = 0.1
loss_history = []
frames = []  # For GIF frames

# --------------------------
# TRAINING LOOP
# --------------------------
for epoch in range(epochs):
    # Forward pass
    A2, cache = forward_pass(X, parameters)
    
    # Compute loss
    loss = compute_loss(A2, Y)
    loss_history.append(loss)
    
    # Backpropagation
    parameters = backpropagation(parameters, cache, Y, learning_rate)
    
    # Save prediction plot every 20 epochs for GIF
    if epoch % 20 == 0 or epoch == epochs - 1:
        plt.figure(figsize=(5,5))
        # Class 0
        plt.scatter(X[Y[:,0]==0,0], X[Y[:,0]==0,1], color='red', label='Class 0')
        # Class 1
        plt.scatter(X[Y[:,0]==1,0], X[Y[:,0]==1,1], color='blue', label='Class 1')
        # Overlay predictions
        pred_labels = (A2 > 0.5).astype(int)
        plt.scatter(X[pred_labels[:,0]==1,0], X[pred_labels[:,0]==1,1],
                    facecolors='none', edgecolors='yellow', s=80, label='Pred 1')
        plt.title(f'Epoch {epoch}, Loss={loss:.4f}')
        plt.legend()
        plt.xlim(-4,4)
        plt.ylim(-4,4)
        plt.tight_layout()
        
        # Save temporary frame
        frame_path = f"../plots/temp_epoch_{epoch}.png"
        plt.savefig(frame_path)
        plt.close()
        frames.append(frame_path)

# --------------------------
# PLOT LOSS CONVERGENCE
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
# SAVE FINAL PREDICTION PLOT
# --------------------------
plt.figure(figsize=(5,5))
plt.scatter(X[Y[:,0]==0,0], X[Y[:,0]==0,1], color='red', label='Class 0')
plt.scatter(X[Y[:,0]==1,0], X[Y[:,0]==1,1], color='blue', label='Class 1')
pred_labels = (A2 > 0.5).astype(int)
plt.scatter(X[pred_labels[:,0]==1,0], X[pred_labels[:,0]==1,1],
            facecolors='none', edgecolors='yellow', s=80, label='Pred 1')
plt.title(f'Final Predictions, Loss={loss:.4f}')
plt.legend()
plt.xlim(-4,4)
plt.ylim(-4,4)
plt.tight_layout()
plt.savefig("../plots/final_predictions.png")
plt.close()

# --------------------------
# CREATE GIF
# --------------------------
images = [Image.open(frame) for frame in frames]
images[0].save('../gifs/training_progress.gif',
               save_all=True, append_images=images[1:], duration=200, loop=0)

# --------------------------
# SAVE FINAL PARAMETERS
# --------------------------
np.save("../parameters.npy", parameters)

print("Training complete!")
print(f"Final loss: {loss:.4f}")
print("Plots saved in 'plots/' folder, GIF saved in 'gifs/' folder, parameters saved in 'parameters.npy'")
