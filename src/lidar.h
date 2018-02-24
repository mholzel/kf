#ifndef LIDAR_HEADER
#define LIDAR_HEADER

#include "measurement_with_jacobian.h"
#include "Eigen/Dense"

template<typename T = double, int x_size = 4, int y_size = Eigen::Dynamic>
class Lidar : public MeasurementWithJacobian<T, x_size, y_size> {
private:

    const Eigen::Matrix<T, y_size, x_size> Hj;
    const Eigen::Matrix<T, y_size, y_size> R;

public:

    Lidar(const Eigen::Matrix<T, y_size, y_size> &R)
            : MeasurementWithJacobian<T, x_size, y_size>(),
              Hj(Eigen::Matrix<T, y_size, x_size>::Identity(2, x_size)),
              R(R) {
    }

    ~Lidar() {}

    Eigen::Matrix<T, y_size, 1>
    H(const T &t,
      const Eigen::Matrix<T, x_size, 1> &x) {
        return Hj * x;
    }

    Eigen::Matrix<T, y_size, x_size>
    dH(const T &t,
       const Eigen::Matrix<T, x_size, 1> &x) {
        return Hj;
    }

    Eigen::Matrix<T, x_size, 1>
    inv(const T &t,
        const Eigen::Matrix<T, y_size, 1> &y) {
        Eigen::Matrix<T, x_size, 1> x = Eigen::Matrix<T, x_size, 1>::Zero();
        x(0) = y(0);
        x(1) = y(1);
        return x;
    }

    Eigen::Matrix<T, y_size, y_size>
    Pw(const T &t) {
        return R;
    };
};

#endif /* LIDAR_HEADER */
