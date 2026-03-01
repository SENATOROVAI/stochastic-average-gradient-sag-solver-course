# stepik: [https://stepik.org/a/270048](https://stepik.org/a/270048)

# Stochastic Average Gradient (SAG) & SAGA Solver Course

### Variance Reduction Optimization for Large-Scale Machine Learning

[![License](https://img.shields.io/badge/license-MIT-brightgreen.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Website](https://img.shields.io/badge/website-live-blue.svg)](https://senatorovai.github.io/stochastic-average-gradient-sag-solver-course/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18818738.svg)](https://doi.org/10.5281/zenodo.18821191)

---

## 🚀 About This Repository

This project provides a professional and mathematically rigorous explanation of:

* **Stochastic Average Gradient (SAG) algorithm**
* **SAGA optimization algorithm**
* Variance reduction methods
* Large-scale convex optimization
* Sparse machine learning solvers
* Regularized regression optimization

This repository is designed for:

* Data Scientists
* Machine Learning Engineers
* Optimization Researchers
* Students studying empirical risk minimization

---

# 🔎 Keywords (for GitHub search & Google indexing)

stochastic average gradient, sag algorithm, saga optimization, variance reduction methods, convex optimization solver, sparse machine learning optimization, large scale optimization, ridge regression solver, lasso regression solver, empirical risk minimization algorithm

---

# 📚 Empirical Risk Minimization (ERM)

We solve the optimization problem:

$$
\min_{\theta \in \mathbb{R}^d}
F(\theta)
\frac{1}{n}
\sum_{i=1}^{n}
f_i(\theta)
$$

Where:

* $$\theta$$ — model parameters
* $$n$$ — number of samples
* $$f_i(\theta)$$ — loss of sample $$i$$

Example (Ridge regression):

$$
f_i(\theta)
\frac{1}{2}(x_i^T\theta - y_i)^2
+
\frac{\lambda}{2}|\theta|^2
$$

---

# 🔵 Stochastic Gradient Descent (Baseline)

Standard SGD update:

$$
\theta_{k+1}


\theta_k

\eta \nabla f_{i_k}(\theta_k)
$$

Issues:

* High gradient variance
* Slow convergence
* Sublinear rate

---

# 🔵 Stochastic Average Gradient (SAG)

SAG stores gradients for all samples.

Let:

$$
g_i^k = \text{stored gradient for sample } i
$$

Update rule:

$$
\theta_{k+1}
\theta_k
\eta
\frac{1}{n}
\sum_{i=1}^{n}
g_i^k
$$

At each iteration:

1. Sample index $$i_k$$
2. Compute new gradient
3. Replace stored gradient

Key property:

$$
\frac{1}{n}\sum g_i^k
\approx
\nabla F(\theta_k)
$$

Variance decreases over time.

---

# 🔵 SAGA Algorithm (Unbiased Variant)

SAGA improves SAG by removing bias:

$$
\theta_{k+1}
\theta_k
\eta
\left(
\nabla f_{i_k}(\theta_k)
$$

$$
g_{i_k}^k
+
\frac{1}{n}
\sum_{i=1}^{n}
g_i^k
\right)
$$

Properties:

* Unbiased gradient estimator
* Supports composite objectives
* Works with L1 regularization
* Linear convergence for strongly convex problems

---

# 🧠 Convergence Theory

Assume:

* $$F(\theta)$$ is $$\mu$$-strongly convex
* Gradient is $$L$$-Lipschitz continuous

Condition number:

$$
\kappa = \frac{L}{\mu}
$$

Then SAG / SAGA achieve:

$$
\mathcal{O}
\left(
(n + \kappa)
\log\frac{1}{\epsilon}
\right)
$$

Compared to SGD:

$$
\mathcal{O}
\left(
\frac{1}{\epsilon}
\right)
$$

This explains why variance reduction methods dominate in large-scale convex optimization.

---

# 📉 Why Variance Reduction Works

SGD gradient variance:

$$
\mathrm{Var}(\nabla f_{i_k})
$$

SAG/SAGA reduce variance because:

$$
\lim_{k \to \infty}
\mathrm{Var}
\left(
\frac{1}{n}\sum g_i^k
\right)
0
$$

This yields **linear convergence rate**.

---

# 🏗 Project Structure

```
stochastic-average-gradient-sag-solver-course/
│
├── README.md
├── LICENSE
├── requirements.txt
│
├── src/
│   ├── sag.py
│   ├── saga.py
│   ├── loss_functions.py
│
├── examples/
│   └── demo.py
│
├── docs/
│   ├── theory.md
│   ├── convergence.md
│
└── index.html
```

Clean structure improves:

* Academic credibility
* Search visibility
* Portfolio quality

---

# 🐍 Minimal SAGA Implementation (Educational)

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
```

---

# 🚀 Installation

```bash
pip install -r requirements.txt
```

Run demo:

```bash
python examples/demo.py
```

---

# 📌 Applications

* Logistic regression optimization
* Ridge regression solver
* Lasso regression solver
* Sparse machine learning models
* Large-scale convex optimization
* Industrial ML systems

---

# 📖 Related Topics

* Stochastic Gradient Descent (SGD)
* SVRG
* L-BFGS
* Conjugate Gradient
* Variance Reduction Methods
* Convex Optimization

---

If you are studying **stochastic optimization algorithms for machine learning**, this repository provides both theoretical foundations and practical implementation of SAG and SAGA solvers.

⭐ Star the repository if it helps your learning or research.
