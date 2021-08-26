#pragma once

#include "Eigen/Dense"

#include "measurements/measurement.hpp"

/**
 * This class simply extends the Measurement class, and requires that you
 * provide a jacobian of the measurement function H with respect to x.
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