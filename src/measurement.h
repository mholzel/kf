#pragma once

#include "Eigen/Dense"

/**
 * This class represents a measurement function of the form
 *
 * y(t) = H(t,x(t)) + w(t)
 *
 * This class should provide also provide an approximate inverse mapping such that
 *
 * x(t) approx inv(t,y(t))
 *
 * and it should provide a method for computing the prediction error.
 * By default, this is simply y(t) - H(t,x(t)).
 *
 * Finally, this function should also provide a function which computes the
 * variance of w at time t (although this will probably just return a constant matrix).
 */
template<typename T = double, int x_size = Eigen::Dynamic, int y_size = Eigen::Dynamic>
class Measurement {

public:

    virtual ~Measurement() {};

    /** measurement function */
    virtual Eigen::Matrix<T, y_size, 1>
    H(const T &t,
      const Eigen::Matrix<T, x_size, 1> &x) = 0;

    /** the prediction error function */
    virtual Eigen::Matrix<T, y_size, 1>
    error(const T &t,
          const Eigen::Matrix<T, x_size, 1> &x,
          const Eigen::Matrix<T, y_size, 1> &y) {
        return y - H(t, x);
    };

    /** an approximate inverse measurement function that may be used
     * to initialize the state given a measurement. */
    virtual Eigen::Matrix<T, x_size, 1>
    inv(const T &t,
        const Eigen::Matrix<T, y_size, 1> &y) = 0;

    /** measurement noise covariance */
    virtual Eigen::Matrix<T, y_size, y_size>
    Pw(const T &t) = 0;

};