#pragma once

#include "Eigen/Dense"

#include "measurements/measurement.hpp"

/**
 * This class represents a measurement function of the form
 *
 * y(t) = H(t,x(t)) + w(t)
 *
 * This class should provide also provide an approximate inverse mapping Hinv
 * such that
 *
 * x(t) approx Hinv(t,y(t))
 *
 * and it should provide a method for computing the prediction error.
 * By default, this is simply y(t) - H(t,x(t)).
 *
 * Finally, this class also provides the jacobian of the measurement function H.
 */
template <typename T = double, int x_size = Eigen::Dynamic,
          int y_size = Eigen::Dynamic>
class MeasurementWithJacobian : public Measurement<T, x_size, y_size> {
 public:
  virtual ~MeasurementWithJacobian() = default;

  /** jacobian of the measurement function with respect to x */
  virtual Eigen::Matrix<T, y_size, x_size> dH(
      const T &t, const Eigen::Matrix<T, x_size, 1> &x) = 0;
};