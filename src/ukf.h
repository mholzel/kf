#pragma once

#include "utils.h"
#include "dynamics.h"
#include "measurement.h"
#include "Eigen/Dense"

template<typename T = double, int x_size = Eigen::Dynamic, int v_size = Eigen::Dynamic, int y_size = Eigen::Dynamic>
class UKF {
private:

    Dynamics<T, x_size, v_size> &dynamics;
    Measurement<T, x_size, y_size> &measurement_model;

public:

    /** the time corresponding to the state vector */
    T &t;

    /** state vector */
    Eigen::Matrix<T, x_size, 1> &x;

    /** state covariance matrix */
    Eigen::Matrix<T, x_size, x_size> &P;

    /** A flag indicating whether we have received a measurement */
    bool initialized = false;

    UKF(T &t,
        Eigen::Matrix<T, x_size, 1> &x,
        Eigen::Matrix<T, x_size, x_size> &P,
        Dynamics<T, x_size, v_size> &dynamics,
        Measurement<T, x_size, y_size> &measurement_model)
            : t(t), x(x), P(P), dynamics(dynamics), measurement_model(measurement_model) {
    }

    ~UKF() {};

    static Eigen::Matrix<T, x_size + v_size, 2 * (x_size + v_size) + 1>
    generateAugmentedSigmaPoints(Eigen::Matrix<T, x_size, 1> &x,
                                 Eigen::Matrix<T, x_size, x_size> &P,
                                 Eigen::Matrix<T, v_size, v_size> &Q) {

        Eigen::Matrix<T, x_size + v_size, x_size + v_size> PQ = Eigen::Matrix<T,
                x_size + v_size, x_size + v_size>::Zero();
        PQ.block(0, 0, x_size, x_size) = P;
        PQ.block(x_size, x_size, v_size, v_size) = Q;

        Eigen::Matrix<T, x_size + v_size, 1> x_aug = Eigen::Matrix<T, x_size + v_size, 1>::Zero();
        x_aug.block(0, 0, x_size, 1) = x;
        return generateSigmaPoints(x_aug, PQ);
    }

    template<int cols>
    static Eigen::Matrix<T, x_size, 1>
    ukfMean(const Eigen::Matrix<T, x_size, cols> &predicted_states,
            const int n_aug = x_size + v_size) {
        const T lambda = 3 - n_aug;
        const T mean_weight = lambda / (lambda + n_aug);
        const T other_weight = 1 / (2 * (lambda + n_aug));
        return other_weight * predicted_states.rowwise().sum() +
               (mean_weight - other_weight) * predicted_states.col(0);
    }

    template<int cols>
    static Eigen::Matrix<T, x_size, x_size>
    ukfCovariance(const Eigen::Matrix<T, x_size, cols> &predicted_states,
                  const Eigen::Matrix<T, x_size, 1> &mean,
                  const int n_aug = x_size + v_size) {
        const T lambda = 3 - n_aug;
        const T mean_weight = lambda / (lambda + n_aug);
        const T other_weight = 1 / (2 * (lambda + n_aug));
        const Eigen::Matrix<T, x_size, cols> X_x = predicted_states.colwise() - mean;
        return other_weight * X_x * X_x.transpose() +
               (mean_weight - other_weight) * X_x.col(0) * X_x.col(0).transpose();
    }

    template<int cols>
    static Eigen::Matrix<T, x_size, y_size>
    ukfCrossCovariance(const Eigen::Matrix<T, x_size, cols> &x,
                       const Eigen::Matrix<T, x_size, 1> &xbar,
                       const Eigen::Matrix<T, y_size, cols> &y,
                       const Eigen::Matrix<T, y_size, 1> &ybar,
                       const int n_aug = x_size + v_size) {
        const T lambda = 3 - n_aug;
        const T mean_weight = lambda / (lambda + n_aug);
        const T other_weight = 1 / (2 * (lambda + n_aug));
        const Eigen::Matrix<T, x_size, cols> X_x = x.colwise() - xbar;
        const Eigen::Matrix<T, y_size, cols> Y_y = y.colwise() - ybar;
        return other_weight * X_x * Y_y.transpose() +
               (mean_weight - other_weight) * X_x.col(0) * Y_y.col(0).transpose();
    }

    /** update the internal state and state covariance predictions to the specified time */
    Eigen::Matrix<T, x_size, 2 * (x_size + v_size) + 1>
    predict(const T &t) {
        const int n_aug = x_size + v_size;
        const int cols = 2 * n_aug + 1;
        const T dt = t - this->t;

        /* Generate the augmented sigma points */
        const Eigen::Matrix<T, n_aug, cols> aug_sigma_points = generateAugmentedSigmaPoints(x, P, dynamics.Pv);

        /* Now separate these points into the state and noise components. */
        const Eigen::Matrix<T, x_size, cols> states = aug_sigma_points.block(0, 0, x_size, cols);
        const Eigen::Matrix<T, x_size, cols> noises = aug_sigma_points.block(x_size, 0, v_size, cols);

        /* For each pair, use the dynamics to predict the state at time t+dt */
        Eigen::Matrix<T, x_size, cols> predicted_states = Eigen::Matrix<T, x_size, cols>::Zero();
        for (int i = 0; i < cols; ++i) {
            predicted_states.col(i) = dynamics.predictState(t, dt, states.col(i), noises.col(i));
        }

        /* Calculate the mean and covariance */
        this->x = ukfMean(predicted_states, n_aug);
        this->P = ukfCovariance(predicted_states, this->x, n_aug);
        this->t = t;
        return predicted_states;
    }

    /** update the internal state and state covariance predictions
     * given the measurement y at the specified time */
    void
    predictAndUpdate(const T &t,
                     const Eigen::Matrix<T, y_size, 1> &y) {
        if (!initialized) {
            initialized = true;
            this->t = t;
            this->x = measurement_model.inv(t, y);
            /* Note that we are assuming that you initialized P externally. */
        } else {

            /* First, predict up to the current time */
            const int n_aug = x_size + v_size;
            const int cols = 2 * n_aug + 1;
            Eigen::Matrix<T, x_size, cols> predicted_states = predict(t);

            /*
             * Next, predict the observed measurements.
             * Technically, we should recalculate the sigma points
             * to pass to the measurement model, but we will reuse the
             * values output from the prediction step.
             */
            Eigen::Matrix<T, y_size, cols> predicted_measurements;
            for (int i = 0; i < cols; ++i) {
                /* Note that this is a little bit of a roundabout way of calculating
                 * the measurements, but if a measurement model has a
                 * special way of calculating error, then this will be much more
                 * accurate since it can compensate for things like angular errors. */
                predicted_measurements.col(i) = y - measurement_model.error(t, predicted_states.col(i), y);
            }

            /* Now compute the mean and covariance of these predictions */
            Eigen::Matrix<T, y_size, cols> ybar = ukfMean(predicted_measurements, n_aug);
            Eigen::Matrix<T, y_size, cols> S =
                    measurement_model.Pw(t) + ukfCovariance(predicted_measurements, ybar, n_aug);

            /* Finally, compute the cross covariance,  */
            Eigen::Matrix<T, x_size, y_size> cross_covariance =
                    ukfCrossCovariance(predicted_states,
                                       this->x,
                                       predicted_measurements,
                                       ybar,
                                       n_aug);
            Eigen::Matrix<T, x_size, y_size> K = cross_covariance * S.inverse();
            this->x += K * (y - ybar);
            this->P -= K * S * K.transpose();
        }
    }
};