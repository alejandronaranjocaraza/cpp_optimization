// g++ -I ~/eigen-3.4.0 main.cpp
#include "optimization_dependencies.h"
#include <Eigen/Dense>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

int main() {
  // Ensure x is an Eigen::VectorXd, not a std::vector<double>
  Eigen::VectorXd x(2);
  x << 6.0, 4.0;
  Eigen::VectorXd xf_gd = gradient_descent(x, banana);
  Eigen::VectorXd xf_nd = newton_descent(x, banana);

  std::cout << "Optimized x (gradient descent): ";
  std::cout << std::setprecision(2) << std::fixed;
  for (int i = 0; i < xf_gd.size(); ++i) {
    if (xf_gd[i] > -0.05 && xf_gd[i] < 0.05) {
      std::cout << 0.00 << " ";
    } else {
      std::cout << xf_gd[i] << " ";
    }
  }
  std::cout << std::endl;
  std::cout << "Optimized x (newton descent): ";
  std::cout << std::setprecision(2) << std::fixed;
  for (int i = 0; i < xf_nd.size(); ++i) {
    if (xf_nd[i] > -0.05 && xf_nd[i] < 0.05) {
      std::cout << 0.00 << " ";
    } else {
      std::cout << xf_nd[i] << " ";
    }
  }
  std::cout << std::endl;
  return 0;
}
