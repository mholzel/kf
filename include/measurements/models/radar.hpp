#pragma once

#include "Eigen/Dense"

#include "measurements/measurement_with_jacobian.hpp"

template <typename T = double, int x_size = 4, int y_size = Eigen::Dynamic>
class Radar : public MeasurementWithJacobian<T, x_size, y_size> {
 private:
  Eigen::Matrix<T, y_size, 1> e;
  Eigen::Matrix<T, y_size, x_size> Hj;
  const Eigen::Matrix<T, y_size, y_size> R;

 public:
  Radar(const Eigen::Matrix<T, y_size, y_size> &R)
      : MeasurementWithJacobian<T, x_size, y_size>(),
        Hj(Eigen::Matrix<T, y_size, x_size>::Zero(3, x_size)),
        R(R) {
    this->e = Eigen::Matrix<T, y_size, 1>::Zero(3);
  }

  ~Radar() {}

  Eigen::Matrix<T, y_size, 1> H(const T &t,
                                const Eigen::Matrix<T, x_size, 1> &x) {
    Eigen::Matrix<T, y_size, 1> y = Eigen::Matrix<T, y_size, 1>::Zero(3, 1);
    T px = x(0);
    T py = x(1);
    T norm = sqrt(px * px + py * py);
    y(0) = norm;
    y(1) = atan2(py, px);
    if (norm >= 1e-5) {
      T vx;
      T vy;
      if (x_size == 4) {
        /* x = [ px, py, vx, vy ] */
        vx = x(2);
        vy = x(3);
      } else if (x_size == 5) {
        /* x = [ px, py, v, psi, dpsi ] */
        T v = x(2);
        T psi = x(3);
        vx = v * cos(psi);
        vy = v * sin(psi);
      }
      y(2) = (px * vx + py * vy) / norm;
    }
    return e;
  }

  Eigen::Matrix<T, y_size, 1> error(const T &t,
                                    const Eigen::Matrix<T, x_size, 1> &x,
                                    const Eigen::Matrix<T, y_size, 1> &y) {
    /* Get the predicted measurement */
    Eigen::Matrix<T, y_size, 1> yhat = H(t, x);

    /* By default, this function would return y - yhat
     * However, the second entry of that vector is the bearing angle (in
     * radians). We need to be careful because we have no guarantee that the
     * angle y(1) is in the same range as our estimate yhat(1). For instance, if
     * y(1)= 0 and yhat(1) = 2 * pi + 0.1, the error should not be 2 * pi + 0.1,
     * but rather 0.1. To find the correct angle, we use the dot product and
     * then simply correct the sign by projecting onto the vector normal to
     * (px,py) */

    /* First, set the second component of yhat to 0 so that we can calculate the
     * normal errors */
    yhat(1) = 0;
    Eigen::Matrix<T, y_size, 1> e = y - yhat;

    /* Now find the correct bearing angle difference. */
    T px = x(0);
    T py = x(1);
    T norm = sqrt(px * px + py * py);
    px /= norm;
    py /= norm;
    T cosine = cos(y(1)) * px + sin(y(1)) * py;

    /* When using low precision variables (like T = float), I have seen cosines
     * slightly outside of [-1,1], yielding acos(cos) = Nan. So before we
     * compute the acos, we make sure that cos is strictly in this range. */
    if (cosine > 1)
      cosine = 1;
    else if (cosine < -1)
      cosine = -1;
    e(1) = acos(cosine);
    if (cos(y(1)) * py >= sin(y(1)) * px) {
      e(1) = -e(1);
    }
    return e;
  }

  Eigen::Matrix<T, y_size, x_size> dH(const T &t,
                                      const Eigen::Matrix<T, x_size, 1> &x) {
    T px = x(0);
    T py = x(1);
    T vx = x(2);
    T vy = x(3);
    T norm2 = px * px + py * py;
    if (norm2 < 1e-5) {
      Hj = Eigen::Matrix<T, y_size, x_size>::Zero(3, x_size);
    } else {
      T norm = sqrt(px * px + py * py);
      T norm3 = norm * norm2;
      T px_norm = px / norm;
      T py_norm = py / norm;
      T pv = px * vx + py * vy;
      Hj(0, 0) = px_norm;
      Hj(1, 0) = -py / norm2;
      Hj(2, 0) = vx / norm - (px * pv) / norm3;
      Hj(0, 1) = py_norm;
      Hj(1, 1) = px / norm2;
      Hj(2, 1) = vy / norm - (py * pv) / norm3;
      Hj(2, 2) = px_norm;
      Hj(2, 3) = py_norm;
    }
    /* TODO: Not implemented for x_Size = 5.
     * TODO: No point in doing this unless we will also compute jacobians
     * TODO: of the CTRV dynamics for use in an EKF */
    return Hj;
  }

  Eigen::Matrix<T, x_size, 1> inv(const T &t,
                                  const Eigen::Matrix<T, y_size, 1> &y) {
    /* y = [ range, bearing, range dot ] */
    Eigen::Matrix<T, x_size, 1> x = Eigen::Matrix<T, x_size, 1>::Zero();
    T c = cos(y(1));
    T s = sin(y(1));
    x(0) = y(0) * c;
    x(1) = y(0) * s;
    if (x_size == 4) {
      /* x = [ px, py, vx, vy ] */
      /* Assume that the bearing angle is constant */
      x(2) = y(2) * c;
      x(3) = y(2) * s;
    } else if (x_size == 5) {
      /* x = [ px, py, v, psi, dpsi ] */
      x(2) = y(2);
      x(3) = y(1);
      x(4) = 0;
    }
    return x;
  }

  Eigen::Matrix<T, y_size, y_size> Pw(const T &t) { return R; };
};