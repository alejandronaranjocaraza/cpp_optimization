#include "../include/grads.h"
#include <Eigen/Dense>
#include <cmath>

// Forward difference: grad_i ≈ (f(x + h*e_i) - f(x)) / h
// O(h) accuracy, requires n+1 function evaluations
Eigen::VectorXd gradient(
    const Eigen::VectorXd &x,
    double (*fun)(const Eigen::VectorXd &)
    ) {
  const double h = 1e-5;
  int n = x.size();
  Eigen::VectorXd grad_x = Eigen::VectorXd::Zero(n);
  double f_x = fun(x);
  for (int i = 0; i < n; ++i) {
    Eigen::VectorXd xh = x;
    xh(i) += h;
    double f_xh = fun(xh);
    grad_x(i) = (f_xh - f_x) / h;
  }
  return grad_x;
}

// Central difference: grad_i ≈ (f(x + h*e_i) - f(x - h*e_i)) / 2h
// O(h^2) accuracy, requires 2n evaluations — more precise but costlier
Eigen::VectorXd gradient_cen(
    const Eigen::VectorXd &x,
    double (*fun)(const Eigen::VectorXd &)
    ) {
  const double h = 1e-5;
  int n = x.size();
  Eigen::VectorXd grad_x = Eigen::VectorXd::Zero(n);
  double f_x = fun(x);
  for (int i = 0; i < n; ++i) {
    Eigen::VectorXd xp = x;
    Eigen::VectorXd xm = x;
    xp(i) += h;
    xm(i) -= h;
    double f_xp = fun(xp);
    double f_xm = fun(xm);
    grad_x(i) = (f_xp - f_xm) / (2.0 * h);
  }
  return grad_x;
}

// Finite difference Hessian: H_{ij} ≈ (f(x+h*ei+h*ej) - f(x+h*ei) - f(x+h*ej) + f(x)) / h^2
// symmetric=true exploits H_{ij} = H_{ji}, halves off-diagonal evaluations
Eigen::MatrixXd hessian(
    const Eigen::VectorXd &x,
    double (*fun)(const Eigen::VectorXd &),
    bool symmetric
    ) {
  const double epsilon = 1e-5;
  int n = x.size();
  double f_x = fun(x);
  Eigen::MatrixXd hes(n, n);
  Eigen::VectorXd x_shift = x;

  if (symmetric) {
    for (int i = 0; i < n; ++i) {
      x_shift(i) += epsilon;
      double f_x_shift_i = fun(x_shift);
      x_shift(i) -= epsilon;
      for (int j = 0; j <= i; ++j) {

        x_shift(i) += epsilon;
        x_shift(j) += epsilon;
        double f_x_shift_ij = fun(x_shift);
        x_shift(i) -= epsilon;

        double f_x_shift_j = fun(x_shift);
        x_shift(j) -= epsilon;

        hes(i, j) = (f_x_shift_ij - f_x_shift_i - f_x_shift_j + f_x) /
                    (epsilon * epsilon);
        if (i != j) {
          hes(j, i) = hes(i, j);
        }
      }
    }
  } else {
    for (int i = 0; i < n; ++i) {
      x_shift(i) += epsilon;
      double f_x_shift_i = fun(x_shift);
      x_shift(i) -= epsilon;
      for (int j = 0; j < n; ++j) {

        x_shift(i) += epsilon;
        x_shift(j) += epsilon;
        double f_x_shift_ij = fun(x_shift);
        x_shift(i) -= epsilon;

        double f_x_shift_j = fun(x_shift);
        x_shift(j) -= epsilon;

        hes(i, j) = (f_x_shift_ij - f_x_shift_i - f_x_shift_j + f_x) /
                    (epsilon * epsilon);
      }
    }
  }
  return hes;
}
