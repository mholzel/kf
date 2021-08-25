#pragma once

#include "Eigen/Dense"

/**
 * This class represents a set of autonomous dynamic equations of the form
 *
 * x(t+dt) = F(t,dt,x(t)) + G(t,dt,x(t)) * v(t)
 *
 * where v(t) has some covariance Pv.
 * Hence the signal G(t,dt,x(t)) * v(t) has a covariance Q of
 *
 * Q = G * Pv * G^T
 */
template <typename T = double, int x_size = Eigen::Dynamic,
          int v_size = Eigen::Dynamic>
class Dynamics {
 public:
  const Eigen::Matrix<T, v_size, v_size> Pv;

  Dynamics(const Eigen::Matrix<T, v_size, v_size> &Pv) : Pv(Pv) {}

  virtual ~Dynamics(){};

  /** predict the state x(t+dt) given the previous state x(t) and zero noise */
  virtual Eigen::Matrix<T, x_size, 1> predictState(
      const T &t, const T &dt, const Eigen::Matrix<T, x_size, 1> &x) = 0;

  /** predict the state x(t+dt) given the previous state x(t) and nonzero
   * process noise */
  Eigen::Matrix<T, x_size, 1> predictState(
      const T &t, const T &dt, const Eigen::Matrix<T, x_size, 1> &x,
      const Eigen::Matrix<T, v_size, 1> &v) {
    return predictState(t, dt, x) + G(t, dt, x) * v;
  };

  /** process noise multiplier G */
  virtual Eigen::Matrix<T, x_size, v_size> G(
      const T &t, const T &dt, const Eigen::Matrix<T, x_size, 1> &x) = 0;

  /** process noise covariance G * Pv * G^T */
  Eigen::Matrix<T, x_size, x_size> Q(const T &t, const T &dt,
                                     const Eigen::Matrix<T, x_size, 1> &x) {
    Eigen::Matrix<T, x_size, v_size> g = G(t, dt, x);
    return g * Pv * g.transpose();
  }
};