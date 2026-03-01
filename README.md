# stepik: https://stepik.org/a/270048

# Stochastic Average Gradient (SAG) & SAGA Solver Course

> 🚀 Professional implementation and mathematical explanation of **Stochastic Average Gradient (SAG)** and **Stochastic Average Gradient Accelerated (SAGA)** optimization algorithms for large-scale machine learning.

---

## 🔥 Project Overview

This repository provides a complete course-style treatment of:

- Stochastic Average Gradient (SAG)
- SAGA algorithm
- Variance reduction methods
- Convergence analysis
- Sparse optimization support
- Python implementation from scratch

---

## Keywords

```

stochastic average gradient
sag algorithm
saga optimization
variance reduction methods
large scale machine learning
sparse optimization solver
sag python implementation
saga solver from scratch
convex optimization
regularized regression optimization

```

---

## 📚 Optimization Problem

We solve empirical risk minimization:

$$
\min_{\theta} \frac{1}{n} \sum_{i=1}^{n} f_i(\theta)
$$

Where:

- $$f_i(\theta)$$ — loss for sample $$i$$
- $$n$$ — number of training samples
- $$\theta$$ — model parameters

---

# 🔵 Stochastic Average Gradient (SAG)

Standard SGD update:

$$
\theta_{k+1} = \theta_k - \eta \nabla f_{i_k}(\theta_k)
$$

SAG improves this by maintaining a memory of past gradients:

$$
\theta_{k+1} =
\theta_k -
\eta \frac{1}{n} \sum_{i=1}^{n} g_i
$$

Where:

- $$g_i$$ stores last gradient for sample $$i$$
- Variance is reduced compared to SGD

---

# 🔵 SAGA Algorithm

SAGA corrects SAG bias and supports composite objectives:

$$
\theta_{k+1} =
\theta_k -
\eta \left(
\nabla f_{i_k}(\theta_k)
- g_{i_k}
+ \frac{1}{n} \sum_{i=1}^{n} g_i
\right)
$$

Advantages:

✅ Unbiased gradient estimator  
✅ Supports L1 regularization  
✅ Better theoretical guarantees  

---

## ⚡ Why SAG and SAGA Matter

Used in:

- Logistic regression
- Ridge regression
- Lasso regression
- Large-scale convex optimization
- Sparse high-dimensional models
- Industrial machine learning pipelines

They provide:

- Faster convergence than SGD
- Lower variance
- Better scalability

---

## 🧠 Convergence Properties

For strongly convex functions:

$$
J(\theta) \text{ strongly convex}
$$

SAG and SAGA achieve:

$$
\mathcal{O}((n + \kappa)\log(1/\epsilon))
$$

Where:

- $$\kappa$$ — condition number
- $$\epsilon$$ — target accuracy

This is significantly faster than standard SGD.

---

## 🏗 Project Structure

```

stochastic-average-gradient-sag-solver-course/
│
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
│
├── src/
│   ├── sag.py
│   ├── saga.py
│   ├── loss_functions.py
│   ├── optimizer.py
│
├── examples/
│   └── demo.py
│
├── docs/
│   ├── theory.md
│   ├── convergence.md
│
├── images/
│   └── convergence_plot.png
│
└── index.html

````

Clean structure improves:

✔ Discoverability  
✔ Academic credibility  
✔ Portfolio strength  

---

## 🐍 Example — Simplified SAGA Implementation

```python
import numpy as np

class SAGA:
    def __init__(self, X, y, lr=0.01):
        self.X = X
        self.y = y
        self.lr = lr
        self.n, self.d = X.shape
        self.theta = np.zeros(self.d)
        self.grad_memory = np.zeros((self.n, self.d))
        self.grad_avg = np.zeros(self.d)

    def step(self):
        i = np.random.randint(0, self.n)

        grad = self.compute_gradient(i)

        self.theta -= self.lr * (
            grad
            - self.grad_memory[i]
            + self.grad_avg
        )

        self.grad_avg += (grad - self.grad_memory[i]) / self.n
        self.grad_memory[i] = grad

    def compute_gradient(self, i):
        xi = self.X[i]
        yi = self.y[i]
        return xi * (xi @ self.theta - yi)
````

---

## 🚀 Installation

```bash
pip install -r requirements.txt
```

Run example:

```bash
python examples/demo.py
```

---

## 📊 Visualization (Recommended)

Add:

* Loss curve vs iterations
* Variance comparison (SGD vs SAG vs SAGA)
* Convergence speed comparison

Example:

```python
import matplotlib.pyplot as plt

plt.plot(loss_history)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("SAGA Convergence")
plt.show()
```

