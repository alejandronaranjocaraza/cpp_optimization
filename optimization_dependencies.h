#ifndef OPTIMIZATION_DEPENDENCIES_H
#define OPTIMIZATION_DEPENDENCIES_H
#include <Eigen/Dense>
#include <vector>

// FUNCTIONS
double cubic(const Eigen::VectorXd &x);

double quadratic(const Eigen::VectorXd &x);

double banana(const Eigen::VectorXd &x);

double rastrigin(const Eigen::VectorXd &x);

double ackley(const Eigen::VectorXd &x);

// GRADS
Eigen::VectorXd gradient(const Eigen::VectorXd &x,
                         double (*fun)(const Eigen::VectorXd &));

Eigen::VectorXd gradient_cen(const Eigen::VectorXd &x,
                             double (*fun)(const Eigen::VectorXd &));

Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
hessian(const Eigen::VectorXd &x, double (*fun)(const Eigen::VectorXd &),
        bool simetric = false);
/*
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
hessian(const Eigen::VectorXd &x, double (*fun)(const Eigen::VectorXd &),
        bool simetric = false);
*/
// OPTIMIZATION
Eigen::VectorXd gradient_descent(Eigen::VectorXd x0,
                                 double (*fun)(const Eigen::VectorXd &));
Eigen::VectorXd newton_descent(Eigen::VectorXd x0,
                               double (*fun)(const Eigen::VectorXd &));

#endif
