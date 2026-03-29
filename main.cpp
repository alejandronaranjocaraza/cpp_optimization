// Build: g++ -I ~/eigen-3.4.0 -O2 main.cpp optimization_functions.cpp grads.cpp funcs.cpp -o optimizer

#include "optimization_dependencies.h"
#include <Eigen/Dense>
#include <iomanip>
#include <iostream>

static void print_result(const std::string &label, const Eigen::VectorXd &x) {
  std::cout << label << ": ";
  std::cout << std::fixed << std::setprecision(4);
  for (int i = 0; i < x.size(); ++i) {
    std::cout << (std::abs(x(i)) < 5e-3 ? 0.0 : x(i));
    if (i < x.size() - 1) std::cout << ", ";
  }
  std::cout << "\n";
}

int main() {
  Eigen::VectorXd x0(2);
  x0 << 6.0, 4.0;

  const Eigen::VectorXd xf_gd = gradient_descent(x0, banana);
  const Eigen::VectorXd xf_nd = newton_descent(x0, banana);

  std::cout << "Rosenbrock minimization from x0 = (6, 4)\n";
  print_result("Gradient descent", xf_gd);
  print_result("Newton's method ", xf_nd);

  return 0;
}
