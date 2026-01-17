# Neural Network from Scratch

Builds a fully functional neural network **from scratch** using only NumPy (no TensorFlow / PyTorch), with forward pass, backpropagation, training on a dataset, and visualization of loss convergence and decision boundaries.

---

## Abstract

This project demonstrates how a neural network works at a fundamental level:

- Implements **forward pass** and **backpropagation** from scratch.  
- Trains on a simple classification or regression dataset.  
- Visualizes **loss convergence** and **decision boundary evolution**.  
- Produces GIFs and plots to illustrate the training dynamics.  

The project emphasizes understanding the **core mechanics of neural networks**, rather than using high-level libraries.

---

## Why This Project

- Shows you **don’t just “use AI”**, you understand it.  
- Highlights fundamentals: matrix operations, activation functions, and gradient computation.  
- Demonstrates training dynamics visually, showing convergence and learning rates.  
- Admissions signal: "Builder, not consumer."  

---

## Development Iterations

- **v1.0:** Forward pass implementation and initial loss computation  
- **v2.0:** Backpropagation and weight updates  
- **v3.0:** Training loop with multiple epochs and loss convergence plots  
- **v4.0:** Decision boundary visualization  
- **v5.0:** Optional GIFs showing evolution over epochs  

---

## Requirements

- Python 3.11+  
- NumPy  
- Matplotlib  

---

## Phase 0: Setup & Dataset Preparation

**Scientific Question:**  
“How do we structure inputs and outputs for training a neural network?”  

**Description:**  
- Define dataset (classification or regression).  
- Initialize network architecture and weights randomly.  

**Implementation:**  
- Input: Feature matrix X, target vector y  
- Output: Initialized weights and biases saved as `parameters.npy`  

**End-state / Outputs:**  
- Code: `code/setup_dataset.py`  
- Saved weights: `data/parameters.npy`  

**What This Proves:**  
- Understanding of input/output preparation and network initialization.  

---

## Phase 1: Forward Pass

**Scientific Question:**  
“How does input propagate through the network?”  

**Description:**  
- Compute activations layer by layer.  
- Calculate initial loss (binary cross-entropy or MSE).  

**Implementation:**  
- Forward pass through all layers  
- Compute network output  

**Example Output:**  
Forward pass output:
[[0.50000401]
[0.49999837]
[0.49998104]
[0.5 ]]

**End-state / Outputs:**  
- Code: `code/forward_pass.py`  
- Output: Forward activations printed or logged  

**What This Proves:**  
- Network computes predictions correctly using initialized weights.  

---

## Phase 2: Backpropagation

**Scientific Question:**  
“How do we compute gradients to update weights?”  

**Description:**  
- Compute derivatives of loss w.r.t weights and biases.  
- Update weights using gradient descent.  

**Implementation:**  
- Backpropagation through all layers  
- Single training step demonstrated  

**Example Output:**  
Loss before backprop: 0.6931538403054052
Loss after one backprop step: 0.6931522900226582

markdown
Copy code

**End-state / Outputs:**  
- Code: `code/backprop.py`  
- Updated weights saved in `parameters.npy`  

**What This Proves:**  
- Gradients are correctly computed and weights updated.  

---

## Phase 3: Training Loop

**Scientific Question:**  
“How does the network learn over multiple epochs?”  

**Description:**  
- Train network for multiple epochs  
- Track loss at each epoch  
- Optionally save weights periodically  

**Implementation:**  
- Loop over epochs  
- Forward pass → Loss → Backpropagation → Weight update  
- Record loss and optionally intermediate outputs  

**End-state / Outputs:**  
- Code: `code/training_loop.py`  
- Saved weights: `data/parameters.npy`  
- Loss values per epoch (for plotting)  

**What This Proves:**  
- Network can learn patterns in data over time.  
- Training dynamics can be visualized.  

---

## Phase 4: Loss Convergence Visualization

**Scientific Question:**  
“How does the loss evolve during training?”  

**Description:**  
- Plot loss vs epochs for different learning rates.  
- Include snapshots of loss at key epochs (e.g., 0, 240, 499).  
- Optional: GIF showing evolution of loss curve.  

**Implementation:**  
- Use `matplotlib` to plot loss curves  
- Annotate loss at specific epochs  

**Example Outputs:**  
- `plots/phase_4_training_dynamics/main/loss_curve.png`  
- `plots/phase_4_training_dynamics/all_epochs/epoch_000_loss.png`  
- `plots/phase_4_training_dynamics/all_epochs/epoch_240_loss.png`  
- `plots/phase_4_training_dynamics/all_epochs/epoch_499_loss.png`  
- Optional GIF: `loss_evolution.gif`  

**What This Proves:**  
- Visual demonstration of learning progress  
- Effect of learning rate and optimization  

---

## Phase 5: Decision Boundary Visualization

**Scientific Question:**  
“How does the network classify data points in feature space?”  

**Description:**  
- Plot decision boundaries as training progresses  
- Include snapshots at key epochs (0, 240, 499)  
- Optional: GIF showing evolution of decision boundary over all epochs  

**Implementation:**  
- Compute network predictions for a meshgrid  
- Plot regions colored by predicted class  
- Overlay true data points  

**Example Outputs:**  
- `plots/phase_5_decision_boundary/main/decision_boundary_epoch_0.png`  
- `plots/phase_5_decision_boundary/main/decision_boundary_epoch_340.png`  
- `plots/phase_5_decision_boundary/main/decision_boundary_epoch_499.png`  
- `plots/phase_5_decision_boundary/main/final_predictions.png`  
- Optional GIF: `decision_boundary.gif`  

**What This Proves:**  
- Network learns to separate classes correctly  
- Visualizes improvement in predictions over time  

---

## Phase 6: Optional Enhancements

**Ideas:**  
- Animate combined loss + decision boundary  
- Test different activation functions (ReLU, Sigmoid, Tanh)  
- Compare learning rates  
- Evaluate on a slightly more complex dataset  

**Outputs:**  
- GIF combining training dynamics  
- Plots comparing activations, losses, or decision boundaries  

**What This Proves:**  
- Ability to extend the project creatively  
- Shows deeper understanding and experimentation  

---

## Conclusion

This project demonstrates **full end-to-end neural network implementation from scratch**:

1. Dataset preparation and weight initialization  
2. Forward pass computation  
3. Backpropagation and weight updates  
4. Full training loop with loss tracking  
5. Visualization of loss convergence and decision boundaries  

- Emphasizes **fundamentals of neural networks** without using high-level libraries.  
- Combines **numerical computation, visualization, and experimentation**.  
- Optional extensions allow showcasing creativity and understanding of hyperparameters and architectures.  

---

