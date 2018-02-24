#ifndef CTRV_HEADER
#define CTRV_HEADER

#include "Eigen/Dense"
#include "dynamics.h"

/**
 * Constant turn rate and velocity model.
 * This model assumes that the dynamics are following a circular path.
 * The state vector is of the form
 *
 * x = [ px, py, v, psi, dpsi ]^T
 *
 * where
 *
 * px and py denote the (x,y) positions,
 * v denotes the magnitude of the velocity (which is constant)
 * psi denotes the yaw angle
 * dpsi denotes the yaw rate (which is constant)
 *
 * In the continuous-time domain, this system looks like
 *
 * dx = [ v * cos(psi), v * sin(psi), 0, dpsi, 0  ]^T
 *
 * But we need the discrete-time form, which is given by
 *
 * s = v / dpsi
 * a = psi + dt * dpsi
 * x(t+dt) = x(t) + [ s * ( sin(a) - sin(psi)) ]
 *                  [ s * (-cos(a) + sin(psi)) ]
 *                  [       0                  ]
 *                  [     dt * dpsi            ]
 *                  [       0                  ]
 */
template<typename T = double>
class CTRV : public Dynamics<T, 5, 2> {

private :

    /** a matrix that we use when calculating the covariance. */
    Eigen::Matrix<T, 5, 2> g;

public:

    CTRV(const Eigen::Matrix<T, 2, 2> &Pv) : Dynamics<T, 5, 2>(Pv) {
        this->g = Eigen::Matrix<T, 5, 2>::Zero();
    }

    ~CTRV() {};

    /** predict the state x(t+dt) given the previous state x(t) and zero noise */
    Eigen::Matrix<T, 5, 1>
    predictState(const T &t,
                 const T &dt,
                 const Eigen::Matrix<T, 5, 1> &x) {
        Eigen::Matrix<T, 5, 1> dx;
        T v = x(2);
        T psi = x(3);
        T dpsi = x(4);
        if (abs(dpsi) > 1e-5) {
            T scale = v / dpsi;
            T a = psi + dt * dpsi;
            dx << scale * (sin(a) - sin(psi)), scale * (-cos(a) + cos(psi)), 0, dt * dpsi, 0;
        } else {
            dx << v * dt * cos(psi), v * dt * sin(psi), 0, dt * dpsi, 0;
        }
        return x + dx;
    }

    /** process noise multiplier G */
    Eigen::Matrix<T, 5, 2>
    G(const T &t,
      const T &dt,
      const Eigen::Matrix<T, 5, 1> &x) {
        T dt2 = dt * dt / 2;
        T psi = x(3);
        g(0, 0) = dt2 * cos(psi);
        g(1, 0) = dt2 * sin(psi);
        g(2, 0) = dt;
        g(3, 1) = dt2;
        g(4, 1) = dt;
        return g;
    }
};

#endif /* CTRV_HEADER */
