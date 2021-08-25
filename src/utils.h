#pragma once

#include <vector>
#include "Eigen/Dense"

class Utils {
public:

    /** A helper method to calculate RMSE. */
    template<typename T = double, int x_size = Eigen::Dynamic>
    static Eigen::Matrix<T, x_size, 1>
    rmse(const std::vector<Eigen::Matrix<T, x_size, 1>> &estimates,
         const std::vector<Eigen::Matrix<T, x_size, 1>> &ground_truth) {
        Eigen::Matrix<T, x_size, 1> rmse = Eigen::Matrix<T, x_size, 1>::Zero(estimates[0].size());
        int n = estimates.size();
        for (int i = 0; i < n; ++i) {
            Eigen::Matrix<T, x_size, 1> e2 = (estimates[i] - ground_truth[i]).array().square();
            rmse += e2;
        }
        return rmse / n;
    }

    /** Convert a unix timestamp in microseconds (long long) to seconds.
     * By default, we will return the seconds since GMT: Monday, January 1, 2018 12:00:00 AM,
     * (NOT the seconds since Unix epoch (in 1970)). */
    template<typename T = double>
    static T
    timestampToSeconds(long long timestamp,
                       long long since = 1514764800000000) {
        return (timestamp - since) / 1e6;
    }
};
