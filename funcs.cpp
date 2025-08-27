#include "optimization_dependencies.h"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <set>

/*
class BasicFunctions {
private:
  std::string type;
  Eigen::VectorXd params;
  std::set<std::string> valid_types{"polynomial"};

public:
  BasicFunctions(const std::string &type, const Eigen::VectorXd &params)
      : type(type), params(params) {
    if (valid_types.find(type) == valid_types.end()) {
      std::cerr << "Invalid Function" << std::endl;
      exit(1);
    }
  }
  void params() {
    std::cout << "Tipo: " << type << std::endl;
    std::cout << "Parámetros: ";
    for (const double &param : params) {
      std::cout << param << " ";
    }
    std::cout << std::endl;
  }
  double eval(double x) {
    double res{0};
    if (type == "polynomial") {
      // Horner's method to evaluate the polynomial
      for (int i = params.size() - 1; i >= 0; i--) {
        res = x * res + params[i];
      }
    }
    return res;
  }
};
*/

// QUADRATIC
double quadratic(const Eigen::VectorXd &x) {
  double res = 0.0;
  for (double xi : x) {
    res += xi * xi;
  }
  return res;
}

// CUBIC
double cubic(const Eigen::VectorXd &x) {
  double res = 0.0;
  for (double xi : x) {
    res += 3 * xi * xi * xi + 5 * xi * xi + 2;
  }
  return res;
}

// ROSENBROCK
// *not convex
// *banana shape
// *single local minimum
double banana(const Eigen::VectorXd &x) {
  double res = 0;
  int n = x.size();
  for (int i = 0; i < n - 1; ++i) {
    double dif1 = x(i + 1) - x(i) * x(i);
    double dif2 = 1 - x(i);
    res += 100 * dif1 * dif1 + dif2 * dif2;
  }
  return res;
}

// RASTRIGIN
// *many local minima
// *difficult for gradient based methods due to oscillation
// *global minimum x=(0,0,...,0)
double rastrigin(const Eigen::VectorXd &x) {
  double res = 0;
  int n = x.size();
  res += 10 * n;
  for (double xi : x) {
    res += xi * xi - 10 * std::cos(2 * M_PI * xi);
  }
  return res;
}

// ACKLEY
// *many local minima
// *large plateau
// *single global minimum x=(0,0,...,0)
double ackley(const Eigen::VectorXd &x) {
  double res = 0;
  double sum1 = 0;
  double sum2 = 0;
  double e = std::exp(1);
  int n = x.size();
  for (double xi : x) {
    sum1 += xi * xi;
    sum2 += std::cos(2 * M_PI * xi);
  }
  res =
      -20 * std::exp(-0.2 * std::sqrt(sum1 / n)) - std::exp(sum2 / n) + 20 + e;
  return res;
}

/*
int main() {
  std::vector<double> x{2, 2};
  double res1 = cuadratic(x);
  double res2 = cubic(x);
  double res3 = banana(x);
  double res4 = rastrigin(x);
  double res5 = ackley(x);
  std::cout << res1 << std::endl;
  std::cout << res2 << std::endl;
  std::cout << res3 << std::endl;
  std::cout << res4 << std::endl;
  std::cout << res5 << std::endl;
  return 0;
}
*/
