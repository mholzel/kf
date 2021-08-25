#pragma once

#include "Eigen/Dense"

enum EventType { LIDAR, RADAR };

template <typename T = double, int y_size = Eigen::Dynamic>
class Event {
 public:
  T time;
  EventType type;
  Eigen::Matrix<T, y_size, 1> data;
};