#pragma once

#include <Eigen/Dense>

// Sum of squares: f(x) = sum(x_i^2), minimum at origin
double quadratic(const Eigen::VectorXd &x);

// f(x) = sum(3*x_i^3 + 5*x_i^2 + 2)
double cubic(const Eigen::VectorXd &x);

// Rosenbrock: f(x) = sum 100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2
// Non-convex, banana-shaped valley, global minimum at (1,1,...,1)
double banana(const Eigen::VectorXd &x);

// Rastrigin: f(x) = 10n + sum(x_i^2 - 10*cos(2*pi*x_i))
// Highly multimodal — gradient methods typically get trapped in local minima
// Global minimum at x = (0,...,0)
double rastrigin(const Eigen::VectorXd &x);

// Ackley: large flat plateau with many local minima
// Global minimum at x = (0,...,0)
double ackley(const Eigen::VectorXd &x);
