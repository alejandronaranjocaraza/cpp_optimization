#include "optimization_dependencies.h"
#include <Eigen/Dense>
#include <cmath>

// Gradient descent with Wolfe condition line search
// At each step, searches along -grad(f). Step size alpha satisfies:
//   (1) Armijo sufficient decrease: f(x + a*p) <= f(x) + c1*a*grad^T*p
//   (2) Curvature condition:        grad(x + a*p)^T*p >= c2*grad^T*p
// See Nocedal & Wright (2006), Ch. 3
Eigen::VectorXd gradient_descent(Eigen::VectorXd x0,
                                 double (*fun)(const Eigen::VectorXd &)) {
  const double c1 = 1e-4;
  const double c2 = 0.9;
  const double tol = 1e-5;
  const int max_iter = 10000;
 
  Eigen::VectorXd xk = x0;
  Eigen::VectorXd grad = gradient(x0, fun);

  for (int iter = 0; iter < max_iter && grad.norm() > tol; ++iter) {
    
    // Calculate alpha
    const Eigen::VectorXd p = -grad;
    const double fx = fun(xk);
    const double slope = grad.dot(p);
    double alpha = 1.0;

    for (int ls = 0; ls < 10000; ++ls) {
      const Eigen::VectorXd xk_trial = xk + alpha * p;
      const bool armijo   = fun(xk_trial) <= fx + c1 * alpha * slope;
      const bool curvature = gradient(xk_trial, fun).dot(p) >= c2 * slope;
      if (armijo && curvature) break;
      alpha *= 0.5;
    }
    xk = xk + alpha * p;
    grad = gradient(xk, fun);
  }
  return xk;
}

// Newton's method
// Computes Newton step p = -H(x)^{-1} * grad(x) at each iteration
// Solved via LDLT factorization — numerically stable for symmetric matrices
// Converges quadratically near minimum, but no line search so may diverge far from it
// See Nocedal & Wright (2006), Ch. 3
Eigen::VectorXd newton_descent(Eigen::VectorXd x0,
                               double (*fun)(const Eigen::VectorXd &)) {
  const int max_iter = 10000;
  const double tol = 1e-5;
  Eigen::VectorXd xk = x0;
  for (int iter = 0; iter < max_iter; ++iter) {
    const Eigen::VectorXd grad = gradient(xk, fun);
    if (grad.norm() < tol) break;
    const Eigen::MatrixXd H = hessian(xk, fun);
    xk += H.ldlt().solve(-grad);  // Newton step
  }
  return xk;
}
