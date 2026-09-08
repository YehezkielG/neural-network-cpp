# Neural Network from Scratch in C++

A lightweight neural network and automatic differentiation framework built from scratch in C++ for learning and experimentation.

## 🎯 Project Overview

The primary goal of this project is to build a deep learning framework entirely from scratch, without relying on existing libraries like PyTorch or TensorFlow. This serves as an educational exercise to deeply understand the inner workings of neural networks.

Key objectives:
- Understanding forward propagation and backpropagation first-hand.
- Building a custom dynamic computational graph.
- Implementing an automatic differentiation (Autograd) engine.
- Creating custom optimizers (like SGD and Adam) to update weights.
- Validating the framework by training a model on a real-world dataset (MNIST).

*Note: This project is heavily oriented towards learning and understanding the internal mechanisms of neural networks, rather than achieving production-level performance or matching the feature set of established frameworks.*

## 🧠 What I Built

At the core of the framework is the custom `Tensor` abstraction, utilizing the Handle-Body idiom for safe memory management.

```text
Tensor (Handle)
  ↓
std::shared_ptr<TensorImpl>
  ↓
TensorImpl (Body)
  ├── vector<double> data                // Flattened 1D array for matrix values
  ├── shared_ptr<TensorImpl> grads       // Gradients for backpropagation
  ├── set<shared_ptr<TensorImpl>> _prev  // Links to previous nodes (Computational Graph)
  ├── function _backward                 // Lambda for calculating local gradients
  ├── string name                        // Operation or variable name
  ├── int rows                           // Matrix row dimension
  ├── int cols                           // Matrix column dimension
  └── bool isOperation                   // Flag for topological sort
```

The `Tensor` class serves as a lightweight wrapper that users interact with. The actual tensor data, gradients, links to the computational graph, and the specific backward derivative functions are stored inside `TensorImpl` on the heap.

**Memory Representation:** Even though the framework conceptually handles 2D matrices, internally, all matrix data and gradients are flattened and stored as 1-dimensional arrays (`std::vector<double>`). This continuous memory allocation (mapping 2D indices `[row][col]` to a 1D index `[row * cols + col]`) ensures better CPU cache locality and computational efficiency compared to nested vectors.

## 🔗 Automatic Differentiation

Every mathematical operation performed on a `Tensor` dynamically builds a computational graph.

For example, a simple feed-forward network creates a graph like this:

```text
X ──→ Linear ──→ ReLU ──→ Linear ──→ Loss
       │           │          │
       └──────── computational graph ────────┘
```

When you call:

```cpp
loss.backward();
```

The Autograd engine builds a topological ordering of the graph starting from the `loss` node, and traverses it backwards, calling the `_backward()` lambda associated with each node to compute and accumulate gradients using the chain rule.

## ➕ Operations

The framework supports essential tensor operations, each with its own manually derived backward pass.

- Addition (`+`)
- Subtraction (`-`)
- Element-wise multiplication (`*`)
- Element-wise division (`/`)
- Matrix multiplication (`.matmul()` or `.mn()`)
- Transpose (`.t()` or `.transpose()`)
- Sigmoid (`.sigmoid()`)
- ReLU (`.ReLU()`)
- Softmax (`.softmax()`)

**Example (Matrix Multiplication $C = AB$):**

Given $C = AB$, the gradients during the backward pass are computed as:

$$ 
\begin{aligned}
\frac{\partial L}{\partial A} &= \frac{\partial L}{\partial C}B^T \\
\frac{\partial L}{\partial B} &= A^T\frac{\partial L}{\partial C}
\end{aligned} 
$$

## 📉 Loss Functions

Implemented loss functions for calculating model error:

- **Binary Cross Entropy** (`BinaryCrossEntropy`)
- **Mean Squared Error** (`MSE`)
- **Categorical Cross Entropy** (`CategoricalCrossEntropy`)

**Categorical Cross Entropy Formula:**

$$ L = -\sum_i y_i \log(\hat{y}_i) $$

## ⚡ Optimizers

Optimizers handle updating the model parameters using the calculated gradients.

- **SGD (Stochastic Gradient Descent)**
  $$ \theta_t = \theta_{t-1} - \alpha g_t $$

- **Adam (Adaptive Moment Estimation)**
  Maintains moving averages of the gradients and squared gradients.
  $$
  \begin{aligned}
  m_t &= \beta_1 m_{t-1} + (1-\beta_1)g_t \\
  v_t &= \beta_2 v_{t-1} + (1-\beta_2)g_t^2 \\
  \hat{m}_t &= \frac{m_t}{1-\beta_1^t} \\
  \hat{v}_t &= \frac{v_t}{1-\beta_2^t} \\
  \theta_t &= \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}
  \end{aligned}
  $$

## 🏗️ Neural Network Layers

The framework provides building blocks for neural networks, such as fully connected layers.

- `nn::linear`

A linear layer takes an input and applies a linear transformation using weights and biases:

$$ Y = XW + b $$

## ✍️ Example Usage

Here is a short example of defining a 2-layer network, running a forward pass, and updating parameters:

```cpp
#include "neuron.h"

using namespace nn;

// Define layers
linear layer1(784, 128);
linear layer2(128, 10);
adam optimizer; // Assuming optimizer setup

// Forward pass
Tensor hidden = layer1(X);
hidden = hidden.ReLU();

Tensor logits = layer2(hidden);
Tensor probabilities = logits.softmax();

// Compute loss
Tensor loss = CategoricalCrossEntropy(Y, probabilities);

// Backward pass (Compute gradients)
loss.backward();

// Update weights
optimizer.step(loss, 0.001);
loss.zero_grad();
```

## 🧪 MNIST Experiment

To prove that this isn't just an implementation of formulas and that the framework *actually learns*, it was tested on the classic MNIST dataset.

- **Dataset**: MNIST (Handwritten digits)
- **Input**: 28 × 28 pixels
- **Flattened Input**: 784 features
- **Architecture**: 784 → 128 → 10
- **Optimizer**: Adam

**Training Log Snippet:**

```text
Epoch: 1 Step: 2200 Loss: 24.0493
...
Epoch: 1 Step: 10000 Loss: 9.33218
Epoch 1 Loss: 9.33218

Epoch: 2 Step: 100 Loss: 3.20316
Epoch: 2 Step: 200 Loss: 2.70731
...
Epoch: 2 Step: 10000 Loss: 2.25807
Epoch 2 Loss: 2.25807

Epoch: 3 Step: 10000 Loss: 1.18073
Epoch 3 Loss: 1.18073

Accuracy: 87.7%
```

## 📊 Results

After 3 epochs of training, the custom model successfully learned to classify handwritten digits.

| Metric | Result |
| :--- | :--- |
| Training samples | 60,000 |
| Epochs | 3 |
| Architecture | 784 → 128 → 10 |
| Optimizer | Adam |
| Test Accuracy | 87.7% |


## 🔬 Implementation Details

### Computational Graph Memory Model

Managing the lifecycle of nodes in a computational graph is tricky in C++. If nodes are destroyed too early, `loss.backward()` will encounter dangling pointers. If they are never destroyed, a memory leak occurs.

This framework solves this using `std::shared_ptr<TensorImpl>`.

- Variables and intermediate results are held as `shared_ptr`.
- The `_prev` set in `TensorImpl` stores `shared_ptr`s to the parent nodes that created it.
- This creates a directed acyclic graph (DAG) where nodes are kept alive exactly as long as they are needed for the backward pass, and automatically destroyed when the graph is no longer referenced.

## 🧪 Limitations

As an educational project, this framework has several practical limitations:

- **2D Tensors Only**: Currently focuses on 2D matrices; no robust support for 3D/4D tensors.
- **CPU Bound**: No GPU acceleration (CUDA/OpenCL) implemented.
- **No Convolutional Layers**: Does not yet support CNNs.
- **Simple Broadcasting**: Automatic broadcasting is not fully comprehensive across all operations.
- **Performance**: Not optimized for large-scale training or extreme computational efficiency.
- **Data Types**: Strictly utilizes `double` for precision, lacking `float16`/`float32` optimizations.
- **API**: The API is experimental and subject to change.

## 📚 Why I Built This

This project was built as a learning exercise to understand neural networks beyond high-level frameworks. Instead of relying on existing automatic differentiation and optimization libraries, the core mathematical operations, computational graph, backward propagation, and optimizers were implemented manually in C++.

It demonstrates the journey from pure mathematics, to tensor abstractions, to computational graphs, to autograd, and finally to a functioning neural network that can actually learn from real data.
