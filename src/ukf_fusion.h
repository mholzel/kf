#pragma once

#include "utils.h"
#include "dynamics.h"
#include "measurement.h"
#include "Eigen/Dense"

#include <iostream>

using namespace std;

template <typename T = double, int x_size = Eigen::Dynamic,
          int v_size = Eigen::Dynamic, int y_size = Eigen::Dynamic>
class UKFFusion {
 private:
  const bool verbose = false;
  Dynamics<T, x_size, v_size> &dynamics;
  std::unordered_map<int, Measurement<T, x_size, y_size> *> &measurement_models;

  /** A handy function for retrieving the filter of the specified type and
   * throwing an error if it doesn't exist. */
  Measurement<T, x_size, y_size> *getMeasurementModel(const EventType &type) {
    auto measurement_model = measurement_models.find(type);
    // TODO This should be an assert, but to satisfy the project goals we
    // comment it out
    //        assert(measurement_model != measurement_models.end());
    if (measurement_model == measurement_models.end()) {
      return NULL;
    }
    return measurement_model->second;
  }

  static T defaultLambda(int size) { return 3 - size; }

 public:
  /** the time corresponding to the state vector */
  T &t;

  /** state vector */
  Eigen::Matrix<T, x_size, 1> &x;

  /** state covariance matrix */
  Eigen::Matrix<T, x_size, x_size> &P;

  /** A flag indicating whether we have received a measurement */
  bool initialized = false;

  UKFFusion(T &t, Eigen::Matrix<T, x_size, 1> &x,
            Eigen::Matrix<T, x_size, x_size> &P,
            Dynamics<T, x_size, v_size> &dynamics,
            std::unordered_map<int, Measurement<T, x_size, y_size> *>
                &measurement_models)
      : t(t),
        x(x),
        P(P),
        dynamics(dynamics),
        measurement_models(measurement_models) {}

  ~UKFFusion(){};

  template <int size>
  static Eigen::Matrix<T, size, 2 * size + 1> generateSigmaPoints(
      Eigen::Matrix<T, size, 1> &x, Eigen::Matrix<T, size, size> &P) {
    const T lambda = defaultLambda(size);
    Eigen::Matrix<T, size, size> sqrtP = P.llt().matrixL();
    sqrtP *= sqrt(lambda + size);
    Eigen::Matrix<T, size, 2 *size + 1> sigma_points =
        Eigen::Matrix<T, size, 2 * size + 1>::Zero();
    sigma_points.block(0, 0, size, 1) = x;
    sigma_points.block(0, 1, size, size) = sqrtP.colwise() + x;
    sigma_points.block(0, size + 1, size, size) = (-sqrtP).colwise() + x;
    return sigma_points;
  }

  static Eigen::Matrix<T, x_size + v_size, 2 * (x_size + v_size) + 1>
  generateAugmentedSigmaPoints(const Eigen::Matrix<T, x_size, 1> &x,
                               const Eigen::Matrix<T, x_size, x_size> &P,
                               const Eigen::Matrix<T, v_size, v_size> &Q) {
    Eigen::Matrix<T, x_size + v_size, x_size + v_size> PQ =
        Eigen::Matrix<T, x_size + v_size, x_size + v_size>::Zero();
    PQ.block(0, 0, x_size, x_size) = P;
    PQ.block(x_size, x_size, v_size, v_size) = Q;

    Eigen::Matrix<T, x_size + v_size, 1> x_aug =
        Eigen::Matrix<T, x_size + v_size, 1>::Zero();
    x_aug.block(0, 0, x_size, 1) = x;
    return generateSigmaPoints(x_aug, PQ);
  }

  template <int size, int cols>
  static Eigen::Matrix<T, size, 1> ukfMean(
      const Eigen::Matrix<T, size, cols> &x, const int n_aug = size + v_size) {
    const T lambda = defaultLambda(n_aug);
    const T mean_weight = lambda / (lambda + n_aug);
    const T other_weight = 1 / (2 * (lambda + n_aug));
    return other_weight * x.rowwise().sum() +
           (mean_weight - other_weight) * x.col(0);
  }

  template <int size, int cols>
  static Eigen::Matrix<T, size, size> ukfCovariance(
      const Eigen::Matrix<T, size, cols> &x,
      const Eigen::Matrix<T, size, 1> &mean, const int n_aug = size + v_size) {
    const T lambda = defaultLambda(n_aug);
    const T mean_weight = lambda / (lambda + n_aug);
    const T other_weight = 1 / (2 * (lambda + n_aug));
    const Eigen::Matrix<T, size, cols> X_x = x.colwise() - mean;
    return other_weight * X_x * X_x.transpose() +
           (mean_weight - other_weight) * X_x.col(0) * X_x.col(0).transpose();
  }

  template <int cols>
  static Eigen::Matrix<T, x_size, y_size> ukfCrossCovariance(
      const Eigen::Matrix<T, x_size, cols> &x,
      const Eigen::Matrix<T, x_size, 1> &xbar,
      const Eigen::Matrix<T, y_size, cols> &y,
      const Eigen::Matrix<T, y_size, 1> &ybar,
      const int n_aug = x_size + v_size) {
    const T lambda = defaultLambda(n_aug);
    const T mean_weight = lambda / (lambda + n_aug);
    const T other_weight = 1 / (2 * (lambda + n_aug));
    const Eigen::Matrix<T, x_size, cols> X_x = x.colwise() - xbar;
    const Eigen::Matrix<T, y_size, cols> Y_y = y.colwise() - ybar;
    return other_weight * X_x * Y_y.transpose() +
           (mean_weight - other_weight) * X_x.col(0) * Y_y.col(0).transpose();
  }

  /** update the internal state and state covariance predictions to the
   * specified time */
  Eigen::Matrix<T, x_size, 2 * (x_size + v_size) + 1> predict(const T &t) {
    const int n_aug = x_size + v_size;
    const int cols = 2 * n_aug + 1;
    const T dt = t - this->t;

    /* Generate the augmented sigma points */
    const Eigen::Matrix<T, n_aug, cols> aug_sigma_points =
        generateAugmentedSigmaPoints(x, P, dynamics.Pv);

    /* Now separate these points into the state and noise components. */
    const Eigen::Matrix<T, x_size, cols> states =
        aug_sigma_points.block(0, 0, x_size, cols);
    const Eigen::Matrix<T, v_size, cols> noises =
        aug_sigma_points.block(x_size, 0, v_size, cols);
    if (verbose) cout << "aug_sigma_points" << endl << aug_sigma_points << endl;

    /* For each pair, use the dynamics to predict the state at time t+dt */
    Eigen::Matrix<T, x_size, cols> predicted_states =
        Eigen::Matrix<T, x_size, cols>::Zero();
    for (int i = 0; i < cols; ++i) {
      predicted_states.col(i) =
          dynamics.predictState(t, dt, states.col(i), noises.col(i));
    }
    if (verbose) cout << "predicted_states" << endl << predicted_states << endl;

    /* Calculate the mean and covariance */
    this->x = ukfMean(predicted_states, n_aug);
    this->P = ukfCovariance(predicted_states, this->x, n_aug);
    this->t = t;
    return predicted_states;
  }

  /** update the internal state and state covariance predictions
   * given the measurement y at the specified time */
  void predictAndUpdate(const Event<T, y_size> &event) {
    const T &t = event.time;
    const Eigen::Matrix<T, y_size, 1> y = event.data;
    Measurement<T, x_size, y_size> *measurement_model =
        getMeasurementModel(event.type);
    // TODO Remove this after submitting for project. The get function should
    // have an assertion.
    if (!measurement_model) {
      return;
    }
    if (!initialized) {
      initialized = true;
      this->t = t;
      this->x = measurement_model->inv(t, y);
      /* Note that we are assuming that you initialized P externally. */
    } else {
      /* First, predict up to the current time */
      const int n_aug = x_size + v_size;
      const int cols = 2 * n_aug + 1;
      const Eigen::Matrix<T, x_size, cols> predicted_states = predict(t);

      /*
       * Next, predict the observed measurements.
       * Technically, we should recalculate the sigma points
       * to pass to the measurement model, but we will reuse the
       * values output from the prediction step.
       */
      Eigen::Matrix<T, y_size, cols> predicted_measurements =
          Eigen::Matrix<T, y_size, cols>::Zero(y.size(), cols);
      for (int i = 0; i < cols; ++i) {
        /* Note that this is a little bit of a roundabout way of calculating
         * the measurements, but if a measurement model has a
         * special way of calculating error, then this will be much more
         * accurate since it can compensate for things like angular errors. */
        predicted_measurements.col(i) =
            y - measurement_model->error(t, predicted_states.col(i), y);
      }
      if (verbose)
        cout << "predicted_measurements" << endl
             << predicted_measurements << endl;

      /* Now compute the mean and covariance of these predictions */
      Eigen::Matrix<T, y_size, 1> ybar = ukfMean(predicted_measurements, n_aug);
      if (verbose) cout << "ybar" << endl << ybar << endl;

      Eigen::Matrix<T, y_size, y_size> S(y.size(), y.size());
      S = measurement_model->Pw(t);
      S += ukfCovariance(predicted_measurements, ybar, n_aug);

      if (verbose) cout << "S: " << endl << S << endl;

      /* Finally, compute the cross covariance,  */
      Eigen::Matrix<T, x_size, y_size> cross_covariance = ukfCrossCovariance(
          predicted_states, this->x, predicted_measurements, ybar, n_aug);
      Eigen::Matrix<T, x_size, y_size> K = cross_covariance * S.inverse();
      this->x += K * (y - ybar);
      this->P -= K * S * K.transpose();
    }
  }
};