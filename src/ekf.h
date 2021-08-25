#pragma once

#include "dynamics_with_jacobian.h"
#include "measurement_with_jacobian.h"
#include "Eigen/Dense"

template <typename T = double, int x_size = Eigen::Dynamic,
          int v_size = Eigen::Dynamic, int y_size = Eigen::Dynamic>
class EKF {
 private:
  DynamicsWithJacobian<T, x_size, v_size> &dynamics;
  MeasurementWithJacobian<T, x_size, y_size> &measurement_model;

 public:
  /** the time corresponding to the state vector */
  T &t;

  /** state vector */
  Eigen::Matrix<T, x_size, 1> &x;

  /** state covariance matrix */
  Eigen::Matrix<T, x_size, x_size> &P;

  /** A flag indicating whether we have received a measurement */
  bool initialized = false;

  EKF(T &t, Eigen::Matrix<T, x_size, 1> &x, Eigen::Matrix<T, x_size, x_size> &P,
      DynamicsWithJacobian<T, x_size, v_size> &dynamics,
      MeasurementWithJacobian<T, x_size, y_size> &measurement_model)
      : t(t),
        x(x),
        P(P),
        dynamics(dynamics),
        measurement_model(measurement_model) {}

  ~EKF(){};

  /** predict the state at the specified time */
  Eigen::Matrix<T, x_size, 1> predictState(const T &t) {
    T dt = t - this->t;
    return dynamics.predictState(t, dt, x);
  }

  /** predict the state covariance at the specified time */
  Eigen::Matrix<T, x_size, x_size> predictStateCovariance(const T &t) {
    T dt = t - this->t;
    Eigen::Matrix<T, x_size, x_size> dFdx = dynamics.dF(t, dt, x);
    return dFdx * P * dFdx.transpose() + dynamics.Q(t, dt, x);
  }

  /** update the internal state and state covariance predictions to the
   * specified time */
  void predict(const T &t) {
    this->x = predictState(t);
    this->P = predictStateCovariance(t);
    this->t = t;
  }

  /** update the internal state and state covariance predictions given the
   * measurement y at the specified time. Note that you will almost always want
   * to call `predict` before this function so that you can be sure that the
   * internal state representation corresponds to the specified time. */
  void update(const T &t, const Eigen::Matrix<T, y_size, 1> &y) {
    Eigen::Matrix<T, y_size, 1> e = measurement_model.error(t, x, y);
    Eigen::Matrix<T, y_size, x_size> C = measurement_model.dH(t, x);
    Eigen::Matrix<T, y_size, x_size> C_P = C * P;
    Eigen::Matrix<T, x_size, y_size> Ct = C.transpose();
    Eigen::Matrix<T, y_size, y_size> S = C_P * Ct + measurement_model.Pw(t);
    Eigen::Matrix<T, x_size, y_size> K = P * Ct * S.inverse();
    this->x += K * e;
    this->P -= K * C_P;
    this->t = t;
  }

  /** update the internal state and state covariance predictions given the
   * measurement y at the specified time */
  void predictAndUpdate(const T &t, const Eigen::Matrix<T, y_size, 1> &y) {
    if (!initialized) {
      initialized = true;
      this->t = t;
      this->x = measurement_model.inv(t, y);
      /* Note that we are assuming that you initialized P externally. */
    } else {
      predict(t);
      update(t, y);
    }
  }
};