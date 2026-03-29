#pragma once

#include <Eigen/Dense>

// --- Objective functions (funcs.cpp) ---
double cubic(const Eigen::VectorXd &x);
double quadratic(const Eigen::VectorXd &x);
double banana(const Eigen::VectorXd &x);
double rastrigin(const Eigen::VectorXd &x);
double ackley(const Eigen::VectorXd &x);

// --- Numerical differentiation (grads.cpp) ---

// Forward difference gradient: O(h) accuracy
Eigen::VectorXd gradient(const Eigen::VectorXd &x,
                         double (*fun)(const Eigen::VectorXd &));

// Central difference gradient: O(h^2) accuracy
Eigen::VectorXd gradient_cen(const Eigen::VectorXd &x,
                             double (*fun)(const Eigen::VectorXd &));

// Finite difference Hessian
// symmetric=true exploits H_{ij} = H_{ji} to halve function evaluations
Eigen::MatrixXd hessian(const Eigen::VectorXd &x, double (*fun)(const Eigen::VectorXd &),
        bool symmetric = false);

// --- Optimizers (optimization_functions.cpp) ---

Eigen::VectorXd gradient_descent(Eigen::VectorXd x0,
                                 double (*fun)(const Eigen::VectorXd &));
Eigen::VectorXd newton_descent(Eigen::VectorXd x0,
                               double (*fun)(const Eigen::VectorXd &));

