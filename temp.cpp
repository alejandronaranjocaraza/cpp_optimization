#include "optimization_dependencies.h"
#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
  Eigen::VectorXd x(4);
  x << 1.0, 2.0, 3.0, 4.0;
  for(double xi:x){
    std::cout<<xi<<std::endl;
  }
}
