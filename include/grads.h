#pragma once

#include <Eigen/Dense>

// Forward difference
// O(h) accuracy, requires n+1 function evaluations
Eigen::VectorXd gradient(
    const Eigen::VectorXd &x,
    double (*fun)(const Eigen::VectorXd &)
    );

// Central difference
// O(h^2) accuracy, requires 2n evaluations — more precise but costlier
Eigen::VectorXd gradient_cen(
    const Eigen::VectorXd &x,
    double (*fun)(const Eigen::VectorXd &)
    );

// Finite difference Hessian
// symmetric=true exploits H_{ij} = H_{ji}, halves off-diagonal evaluations
Eigen::MatrixXd hessian(
    const Eigen::VectorXd &x,
    double (*fun)(const Eigen::VectorXd &),
    bool symmetric = false
    );
