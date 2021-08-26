#pragma once

#include "Eigen/Dense"

/**
 * This class represents a measurement function of the form
 *
 * y(t) = H(t,x(t)) + w(t)
 *
 * To use this function, you must override:
 *
 * - H(t,x(t)): The measurement function
 * - Pw(t) : The covariance of w (often this is constant)
 * - inv(t,y(t)): The approximate inverse mapping such that
 *
 * x(t) approx inv(t,y(t))
 *
 * which may be necessary to initialize the state given a measurement.
 *
 * In addition, you can optionally override the prediction error method.
 * This function typically should just be of the form
 *
 * error(t,x,y) = y(t) - H(t,x(t)).
 *
 * However, when a component of the state represents a quantity like an angle,
 * you may find it beneficial to override this error method.
 */
template <typename T = double, int x_size = Eigen::Dynamic,
          int y_size = Eigen::Dynamic>
class Measurement {
 public:
  virtual ~Measurement() = default;

  /** measurement function */
  virtual Eigen::Matrix<T, y_size, 1> H(
      const T &t, const Eigen::Matrix<T, x_size, 1> &x) = 0;

  /** prediction error function */
  virtual Eigen::Matrix<T, y_size, 1> error(
      const T &t, const Eigen::Matrix<T, x_size, 1> &x,
      const Eigen::Matrix<T, y_size, 1> &y) {
    return y - H(t, x);
  };

  /** an approximate inverse measurement function that may be used
   * to initialize the state given a measurement. */
  virtual Eigen::Matrix<T, x_size, 1> inv(
      const T &t, const Eigen::Matrix<T, y_size, 1> &y) = 0;

  /** measurement noise covariance */
  virtual Eigen::Matrix<T, y_size, y_size> Pw(const T &t) = 0;
};