#include "optimization_dependencies.h"
#include <Eigen/Dense>
#include <cmath>

// Sum of squares: f(x) = sum(x_i^2), minimum at origin
double quadratic(const Eigen::VectorXd &x) {
  return x.squaredNorm();
}

// f(x) = sum(3*x_i^3 + 5*x_i^2 + 2)
double cubic(const Eigen::VectorXd &x) {
  double res = 0.0;
  for (double xi : x) {
    res += 3.0 * xi * xi * xi + 5 * xi * xi + 2;
  }
  return res;
}

// Rosenbrock: f(x) = sum 100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2
// Non-convex, banana-shaped valley, global minimum at (1,1,...,1)
double banana(const Eigen::VectorXd &x) {
  double res = 0;
  int n = x.size();
  for (int i = 0; i < n - 1; ++i) {
    const double dif1 = x(i + 1) - x(i) * x(i);
    const double dif2 = 1 - x(i);
    res += 100.0 * dif1 * dif1 + dif2 * dif2;
  }
  return res;
}

// Rastrigin: f(x) = 10n + sum(x_i^2 - 10*cos(2*pi*x_i))
// Highly multimodal — gradient methods typically get trapped in local minima
// Global minimum at x = (0,...,0)
double rastrigin(const Eigen::VectorXd &x) {
  double res = 0;
  const int n = x.size();
  res += 10 * n;
  for (double xi : x) {
    res += xi * xi - 10.0 * std::cos(2 * M_PI * xi);
  }
  return res;
}

// Ackley: large flat plateau with many local minima
// Global minimum at x = (0,...,0)
double ackley(const Eigen::VectorXd &x) {
  double res = 0;
  double sum1 = 0;
  double sum2 = 0;
  double e = std::exp(1);
  const int n = x.size();
  for (double xi : x) {
    sum1 += xi * xi;
    sum2 += std::cos(2 * M_PI * xi);
  }
  res =
      -20 * std::exp(-0.2 * std::sqrt(sum1 / n)) - std::exp(sum2 / n) + 20 + e;
  return res;
}

