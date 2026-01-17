import numpy as np
import matplotlib.pyplot as plt
from forward import forward_pass  # your Phase 1 file
import os

# --------------------------
# CREATE PLOTS FOLDER IF NOT EXIST
# --------------------------
os.makedirs("../plots", exist_ok=True)

# --------------------------
# LOAD YOUR DATASET AND PARAMETERS
# --------------------------
# Example XOR dataset (replace with your dataset if different)
np.random.seed(42)
num_samples = 200
X = np.random.randn(num_samples, 2)
Y = np.array([[0] if x[0]*x[1] < 0 else [1] for x in X])

# Load trained parameters from Phase 3
parameters = np.load("../parameters.npy", allow_pickle=True).item()

# --------------------------
# FORWARD PASS
# --------------------------
A2, _ = forward_pass(X, parameters)

# --------------------------
# FINAL PREDICTIONS PLOT
# --------------------------
plt.figure(figsize=(5,5))
plt.scatter(X[Y[:,0]==0,0], X[Y[:,0]==0,1], color='red', label='Class 0')
plt.scatter(X[Y[:,0]==1,0], X[Y[:,0]==1,1], color='blue', label='Class 1')
pred_labels = (A2 > 0.5).astype(int)
plt.scatter(X[pred_labels[:,0]==1,0], X[pred_labels[:,0]==1,1],
            facecolors='none', edgecolors='yellow', s=80, label='Pred 1')
plt.title('Final Predictions')
plt.legend()
plt.xlim(-4,4)
plt.ylim(-4,4)
plt.tight_layout()
plt.savefig("../plots/final_predictions.png")
plt.show()

# --------------------------
# DECISION BOUNDARY PLOT
# --------------------------
h = 0.01
x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
grid = np.c_[xx.ravel(), yy.ravel()]

Z, _ = forward_pass(grid, parameters)
Z = (Z > 0.5).astype(int).reshape(xx.shape)

plt.figure(figsize=(5,5))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
plt.scatter(X[Y[:,0]==0,0], X[Y[:,0]==0,1], color='red', label='Class 0')
plt.scatter(X[Y[:,0]==1,0], X[Y[:,0]==1,1], color='blue', label='Class 1')
plt.title('Decision Boundary')
plt.legend()
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.tight_layout()
plt.savefig("../plots/decision_boundary.png")
plt.show()

# --------------------------
# ACCURACY
# --------------------------
accuracy = np.mean((A2 > 0.5) == Y)
print(f"Final accuracy: {accuracy*100:.2f}%")
