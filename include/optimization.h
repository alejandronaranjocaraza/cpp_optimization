#pragma once

#include "funcs.h"
#include "grads.h"

#include <Eigen/Dense>

// Gradient descent with Wolfe condition line search
// See Nocedal & Wright (2006), Ch. 3
Eigen::VectorXd gradient_descent(
    Eigen::VectorXd x0,
    double (*fun)(const Eigen::VectorXd &),
    const double tol = 1e-5,
    const int max_iter = 10000
    );

// Newton's method
// See Nocedal & Wright (2006), Ch. 3
Eigen::VectorXd newton_descent(
    Eigen::VectorXd x0,
    double (*fun)(const Eigen::VectorXd &),
    const double tol = 1e-5,
    const int max_iter = 10000
    );
