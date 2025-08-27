#include "optimization_dependencies.h"
#include <Eigen/Dense>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
LU_fact(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>) {
  return 0;
}

int main() {
  Eigen::Matrix<double, 3, 3> A;
  A << 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9;
  return 0;
}
