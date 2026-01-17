# Neural Network from Scratch

Build a fully functional neural network **from scratch** using only NumPy. Includes forward pass, backpropagation, training, and visualization of loss convergence and decision boundaries.

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
**Output:** Updated weights saved to `parameters.npy`  

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

## Phase 4: Loss Convergence Visualization

**Scientific Question:**  
“How does the loss evolve during training?”

**Implementation:**  
- Plot **loss vs epochs** for different learning rates  
- Snapshots at key epochs: 0, 240, 499  
- Optional GIF showing full evolution  

**Outputs:**  

**Loss Curve (Main Plot):**  
![Loss Curve](plots/phase_4_training_dynamics/main/loss_curve.png)  

**Loss Evolution GIF:**  
![Loss Evolution](plots/phase_4_training_dynamics/main/loss_evolution.gif)  

**Snapshots at Key Epochs:**  

| Epoch 0 | Epoch 240 | Epoch 499 |
|---------|-----------|-----------|
| ![Epoch 0](plots/phase_4_training_dynamics/all_epochs/epoch_000_loss.png) | ![Epoch 240](plots/phase_4_training_dynamics/all_epochs/epoch_240_loss.png) | ![Epoch 499](plots/phase_4_training_dynamics/all_epochs/epoch_499_loss.png) |

**Purpose:**  
- Visual demonstration of learning dynamics  
- Shows how loss decreases over time  
- Highlights effects of learning rates on convergence  

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
