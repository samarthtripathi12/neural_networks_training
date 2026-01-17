# Neural Network from Scratch

Build a fully functional neural network **from scratch** using only NumPy. This project includes forward pass, backpropagation, training, and visualization of loss convergence and decision boundaries.

---

## Abstract

This project demonstrates the fundamentals of neural networks:

- Implements **forward pass** and **backpropagation** from scratch.  
- Trains on a simple classification dataset.  
- Visualizes **loss convergence** and **decision boundary evolution**.  
- Produces GIFs and plots to illustrate training dynamics.

Emphasizes understanding **core mechanics of neural networks** without high-level libraries.

---

## Requirements

- Python 3.11+  
- NumPy  
- Matplotlib  

---

## Phase 0: Setup & Dataset Preparation

**Input:** Feature matrix `X`, target vector `y`  
**Output:** Initialized weights and biases saved in `data/parameters.npy`  

**Code:** `code/setup_dataset.py`  

**Purpose:** Prepares dataset and initializes network parameters.

---

## Phase 1: Forward Pass

**Input:** Dataset + initial weights  
**Output:** Network predictions  

**Code:** `code/forward_pass.py`  

**Example Output:**
Forward pass output:
[[0.50000401]
[0.49999837]
[0.49998104]
[0.5 ]]

**Purpose:** Demonstrates how inputs propagate through the network.

---

## Phase 2: Backpropagation

**Input:** Forward pass outputs + targets  
**Output:** Updated weights saved to `data/parameters.npy`  

**Code:** `code/backprop.py`  

**Example Output:**
Loss before backprop: 0.6931538403054052
Loss after one backprop step: 0.6931522900226582

**Purpose:** Shows gradient computation and weight updates.

---

## Phase 3: Training Loop

**Input:** Dataset + initial weights  
**Output:** Trained weights, loss per epoch  

**Code:** `code/training_loop.py`  
**Weights:** `data/parameters.npy`  

**Purpose:** Trains network over multiple epochs, records loss for visualization.

---

## Phase 4: Loss Convergence & Training Dynamics

**Scientific Question:**  
“How does the network learn over time, and how do individual predictions evolve during training?”

**Implementation:**  
- Track loss at every epoch  
- Plot **overall loss curve** to show convergence  
- Include **snapshots of network predictions / graphs at key epochs** (0, 240, 499)  
- Optional GIF showing **loss evolution over all epochs**  

---

### **Loss Curve (Overall Training)**

This graph shows how the loss decreases over the entire training period:

![Loss Curve](plots/phase_4_training_dynamics/main/loss_curve.png)  

**Purpose:**  
- Demonstrates **convergence trend** of the network  
- Highlights effects of learning rate and optimization  

---

### **Snapshot Graphs at Key Epochs**

These show the **actual network behavior at specific points during training**:  

| Epoch 0 (Initial) | Epoch 240 (Mid Training) | Epoch 499 (Final) |
|------------------|-------------------------|-----------------|
| ![Epoch 0](plots/phase_4_training_dynamics/all_epochs/epoch_000_loss.png) | ![Epoch 240](plots/phase_4_training_dynamics/all_epochs/epoch_240_loss.png) | ![Epoch 499](plots/phase_4_training_dynamics/all_epochs/epoch_499_loss.png) |

**Explanation:**  
- **Epoch 0:** Network starts untrained → high loss (e.g., 0.6932 for binary classification).  
- **Epoch 240:** Network starts learning → loss decreases, predictions begin improving.  
- **Epoch 499:** Network converges → low loss, predictions stabilize.  

---

### **Loss Evolution GIF (Optional)**

A GIF showing **how the loss changes at every epoch** during training:

![Loss Evolution](plots/phase_4_training_dynamics/main/loss_evolution.gif)  

**Purpose:**  
- Makes training dynamics **visually clear**  
- Shows learning progression continuously rather than just snapshots  

**What This Phase Proves:**  
1. The network is **learning and converging** correctly.  
2. You can **see concrete improvements** in predictions over epochs.  
3. Combining **curve + snapshot graphs + GIF** makes your README **professional and competitive**.

---

## Phase 5: Decision Boundary Visualization

**Scientific Question:**  
“How does the network classify points in feature space?”  

**Implementation:**  
- Compute predictions for a grid of points  
- Plot decision regions and overlay true data points  
- Snapshots at key epochs: 0, 340, 499  
- Optional GIF showing evolution  

**Outputs:**  

**Key Epoch Snapshots:**  

| Epoch 0 | Epoch 340 | Epoch 499 | Final Predictions |
|---------|-----------|-----------|-----------------|
| ![Epoch 0](plots/phase_5_decision_boundary/main/decision_boundary_epoch_0.png) | ![Epoch 340](plots/phase_5_decision_boundary/main/decision_boundary_epoch_340.png) | ![Epoch 499](plots/phase_5_decision_boundary/main/decision_boundary_epoch_499.png) | ![Final Predictions](plots/phase_5_decision_boundary/main/final_predictions.png) |

**Decision Boundary Evolution GIF:**  
![Decision Boundary GIF](plots/phase_5_decision_boundary/main/decision_boundary.gif)  

**Purpose:**  
- Shows network learning to separate classes correctly  
- Demonstrates improvement over epochs visually  

---

## Phase 6: Optional Enhancements

**Ideas:**  
- Animate combined **loss + decision boundary**  
- Compare different activation functions (ReLU, Sigmoid, Tanh)  
- Experiment with learning rates and network depth  

**Purpose:**  
- Extends the project creatively  
- Showcases deeper understanding and experimentation  

---

## Conclusion

This project demonstrates **full end-to-end neural network implementation from scratch**:

1. Dataset preparation and weight initialization  
2. Forward pass computation  
3. Backpropagation and weight updates  
4. Full training loop with loss tracking  
5. Visualization of **loss convergence** and **decision boundaries**  

- Emphasizes **fundamentals of neural networks** without libraries.  
- Combines **computation, visualization, and experimentation**.  
- Optional enhancements demonstrate creativity and advanced understanding.
