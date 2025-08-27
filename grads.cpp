#include "optimization_dependencies.h"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <set>
#include <vector>

Eigen::VectorXd gradient(const Eigen::VectorXd &x,
                         double (*fun)(const Eigen::VectorXd &)) {
  double h = std::pow(10, -5);
  int n = x.size();
  Eigen::VectorXd grad_x = Eigen::VectorXd::Zero(n);
  double f_x = fun(x);
  for (int i = 0; i < n; ++i) {
    Eigen::VectorXd xh = x;
    xh(i) += h;
    double f_xh = fun(xh);
    double temp = (f_xh - f_x) / h;
    grad_x(i) = temp;
  }
  return grad_x;
}

Eigen::VectorXd gradient_cen(const Eigen::VectorXd &x,
                             double (*fun)(const Eigen::VectorXd &)) {
  double h = std::pow(10, -5);
  int n = x.size();
  Eigen::VectorXd grad_x = Eigen::VectorXd::Zero(n);
  double f_x = fun(x);
  for (int i = 0; i < n; ++i) {
    Eigen::VectorXd xh1 = x;
    Eigen::VectorXd xh2 = x;
    xh1(i) += h;
    xh2(i) -= h;
    double f_xh1 = fun(xh1);
    double f_xh2 = fun(xh2);
    grad_x(i) = (f_xh1 - f_xh2) / (2.0 * h);
  }
  return grad_x;
}

Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
hessian(const Eigen::VectorXd &x, double (*fun)(const Eigen::VectorXd &),
        bool simetric) {
  double epsilon = std::pow(10, -5);
  int n = x.size();
  double f_x = fun(x);
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> hes(n, n);
  // std::vector<std::vector<double>> hes(n, std::vector<double>(n, 0.0));
  Eigen::VectorXd x_shift = x;

  if (simetric) {
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
