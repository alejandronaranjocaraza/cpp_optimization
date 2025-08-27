#include "optimization_dependencies.h"
#include <Eigen/Dense>
#include <cmath>

Eigen::VectorXd gradient_descent(Eigen::VectorXd x0,
                                 double (*fun)(const Eigen::VectorXd &)) {
  double c2 = 0.9;
  double c1 = 0.1;
  double tol = std::pow(10, -5);
  int kiter = 0;
  int max_kiter = 10000;
  Eigen::VectorXd grad = gradient(x0, fun);
  double fx = fun(x0);
  double norm_grad = grad.norm();
  Eigen::VectorXd xk = x0;
  while (norm_grad > tol && kiter < max_kiter) {
    Eigen::VectorXd p = -grad;
    // Calculate alpha
    double alpha = 1;
    int alpha_max_iter = 10000;
    int alpha_iter = 0;
    double fx_trial = fun(xk + alpha * p);
    Eigen::VectorXd grad_trial = gradient(xk + alpha * p, fun);
    while ((alpha_iter < alpha_max_iter) &&
           (fx_trial > fx + alpha * c1 * grad.dot(p) ||
            grad_trial.dot(p) < c2 * grad.dot(p))) {
      alpha = alpha / 2.0;
      alpha_iter += 1;
      fx_trial = fun(xk + alpha * p);
    }

    xk = xk + alpha * p;
    grad = gradient(xk, fun);
    norm_grad = grad.norm();
    fx = fun(xk);
    kiter += 1;
  }
  return xk;
}

Eigen::VectorXd newton_descent(Eigen::VectorXd x0,
                               double (*fun)(const Eigen::VectorXd &)) {
  int max_kiter = 10000;
  int kiter = 0;
  double tol = std::pow(10, -5);
  Eigen::VectorXd grad = gradient(x0, fun);
  double fx = fun(x0);
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> hes = hessian(x0, fun);
  Eigen::VectorXd pn = hes.ldlt().solve(-grad);
  Eigen::VectorXd xk = x0 + pn;
  double norm_grad = grad.norm();
  while (kiter < max_kiter && norm_grad > tol) {
    grad = gradient(xk, fun);
    hes = hessian(xk, fun);
    pn = hes.ldlt().solve(-grad);
    xk = xk + pn;
    kiter += 1;
    norm_grad = grad.norm();
  }
  return xk;
}
