#pragma once

#include "Eigen/Dense"

#include "dynamics/dynamics_with_jacobian.hpp"

/**
 * Constant velocity model.
 */
template <typename T = double>
class CV : public DynamicsWithJacobian<T, 4, 2> {
 private:
  /** a matrix that we use when calculating the covariance. */
  Eigen::Matrix<T, 4, 2> g;

  /** a matrix used for computing the dynamics */
  Eigen::Matrix<T, 4, 4> A;

 public:
  CV(const Eigen::Matrix<T, 2, 2> &Pv) : DynamicsWithJacobian<T, 4, 2>(Pv) {
    this->g = Eigen::Matrix<T, 4, 2>::Zero();
    this->A = Eigen::Matrix<T, 4, 4>::Identity();
  }

  /** predict the state x(t+dt) given the previous state x(t) and zero noise */
  Eigen::Matrix<T, 4, 1> predictState(const T &t, const T &dt,
                                      const Eigen::Matrix<T, 4, 1> &x) {
    A(0, 2) = dt;
    A(1, 3) = dt;
    return A * x;
  }

  /** process noise multiplier G */
  Eigen::Matrix<T, 4, 2> G(const T &t, const T &dt,
                           const Eigen::Matrix<T, 4, 1> &x) {
    T dt2 = dt * dt / 2;
    g(0, 0) = dt2;
    g(1, 1) = dt2;
    g(2, 0) = dt;
    g(3, 1) = dt;
    return g;
  }

  /** jacobian of the state transition function with respect to x */
  Eigen::Matrix<T, 4, 4> dF(const T &t, const T &dt,
                            const Eigen::Matrix<T, 4, 1> &x) {
    A(0, 2) = dt;
    A(1, 3) = dt;
    return A;
  }
};