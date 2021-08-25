#pragma once

#include "Eigen/Dense"

#include "dynamics/dynamics.hpp"

/**
 * This class represents a set of autonomous dynamic equations of the form
 *
 * x(t+dt) = F(t,dt,x(t)) + G(t,dt,x(t)) * v(t)
 *
 * with a known jacobian df/dx, and covariance Pv of v.
 */
template <typename T = double, int x_size = Eigen::Dynamic,
          int v_size = Eigen::Dynamic>
class DynamicsWithJacobian : public Dynamics<T, x_size, v_size> {
 public:
  DynamicsWithJacobian(const Eigen::Matrix<T, v_size, v_size> &Pv)
      : Dynamics<T, x_size, v_size>(Pv) {}

  virtual ~DynamicsWithJacobian(){};

  /** jacobian of the state transition function with respect to x */
  virtual Eigen::Matrix<T, x_size, x_size> dF(
      const T &t, const T &dt, const Eigen::Matrix<T, x_size, 1> &x) = 0;
};
