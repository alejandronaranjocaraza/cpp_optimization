# Basic Numerical Optimization in C++

*Developed as part of the Numerical Optimization course. Revisited and refactored for clarity.*

Implementation of gradient-based optimization algorithms using finite difference derivatives, built with [Eigen](https://libeigen.gitlab.io/eigen/docs-5.0/GettingStarted.html).

## Algorithms

- **Gradient Descent** — with Wolfe condition line search (sufficient decrease + curvature)
- **Newton's Method** — using LDLT Cholesky factorization to solve the Newton system

## Benchmark Functions

| Function | Properties |
|---|---|
| Quadratic | Convex, smooth — used for basic convergence testing |
| Rosenbrock (banana) | Non-convex, narrow curved valley, single minimum at (1,1,...,1) |
| Rastrigin | Highly multimodal — challenging for gradient methods |
| Ackley | Many local minima with large flat plateau |

## Structure
```
.
├── main.cpp                    # Entry point — runs optimizers on benchmark functions
├── optimization_functions.cpp  # Gradient descent and Newton's method
├── grads.cpp                   # Finite difference gradient and Hessian
├── funcs.cpp                   # Benchmark objective functions
├── optimization_dependencies.h # Shared declarations
```

## Build

Requires [Eigen 3.x](https://eigen.tuxfamily.org/). On Debian based systems:
```bash
sudo apt install libeigen3-dev
mkdir build && cd build
cmake ..
make
./optimize
```

## References

- Nocedal, J. & Wright, S. J. (2006). *Numerical Optimization* (2nd ed.). Springer.
