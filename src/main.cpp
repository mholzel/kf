#include <iostream>

#include <uWS/uWS.h>
#include "json.hpp"

#include "dynamics/models/ctrv.hpp"
#include "dynamics/models/cv.hpp"
#include "filters/ekf_fusion.hpp"
#include "filters/ukf_fusion.hpp"
#include "measurements/models/lidar.hpp"
#include "measurements/models/radar.hpp"
#include "utils.hpp"

using namespace Eigen;
using namespace std;
using T = double;
using json = nlohmann::json;

/* Checks if the SocketIO event has JSON data.
   If there is data the JSON object in string format will be returned,
   else the empty string "" will be returned.
*/
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.find_first_of("]");
  if (found_null != string::npos) {
    return "";
  } else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 1);
  }
  return "";
}

/* The goal is to get the final RMSE error under  [.09, .10, .40, .30] */
int main(int argc, char *argv[]) {
  uWS::Hub h;
  const double pscale = argc > 1 ? atof(argv[1]) : 1;
  const double vvar = argc > 2 ? atof(argv[2]) : 9;
  const double yawvar = argc > 3 ? atof(argv[3]) : 1;

  /* Note: TO use the CV model, change this to false, and rename the CV model
   * below to "dynamics" */
  const bool use_ctrv_model = true;
  const bool use_radar = true;
  const bool use_lidar = true;

  /* The state size is based on the model we are using.
   * The constant velocity model has 4 states,
   * the ctrv model has 5. */
  const int x_size = use_ctrv_model ? 5 : 4;

  /* Initial time, state, and state covariance */
  T t = 0;
  Matrix<T, x_size, 1> x = Matrix<T, x_size, 1>::Zero();
  Matrix<T, x_size, x_size> P = pscale * Matrix<T, x_size, x_size>::Identity();

  /* The dynamic model. We will use which model is called "dynamics" */
  Matrix<T, 2, 2> cv_covariance;
  cv_covariance << 9, 0, 0, 9;
  CV<T> cv(cv_covariance);

  Matrix<T, 2, 2> ctrv_covariance = 9. * Matrix<T, 2, 2>::Identity();
  ctrv_covariance << vvar, 0, 0, yawvar;
  CTRV<T> dynamics(ctrv_covariance);

  /* The lidar and radar measurement models */
  Matrix<T, Dynamic, Dynamic> R_lidar =
      0.0225 * Matrix<T, Dynamic, Dynamic>::Identity(2, 2);
  Lidar<T, x_size> lidar(R_lidar);

  Matrix<T, Dynamic, Dynamic> R_radar =
      0.09 * Matrix<T, Dynamic, Dynamic>::Identity(3, 3);
  R_radar(1, 1) = 0.0009;
  Radar<T, x_size> radar(R_radar);

  /* Fuse the car dynamics with the lidar and radar measurement models into a
   * UKF */
  //    unordered_map<int, MeasurementWithJacobian<T, x_size> *>
  //    models_with_jacobians; models_with_jacobians[EventType::LIDAR] = &lidar;
  //    models_with_jacobians[EventType::RADAR] = &radar;
  //    EKFFusion<T, x_size, 2> ekfFusion(t, x, P, dynamics,
  //    models_with_jacobians);

  unordered_map<int, Measurement<T, x_size> *> models;
  if (use_lidar) {
    models[EventType::LIDAR] = &lidar;
  }
  if (use_radar) {
    models[EventType::RADAR] = &radar;
  }
  UKFFusion<T, x_size, 2> ukfFusion(t, x, P, dynamics, models);

  IOFormat format(4, 0, ", ", "\n", "[", "]");
  vector<Matrix<T, 4, 1>> estimates;
  vector<Matrix<T, 4, 1>> fusedGroundTruth;
  long long since = 0;
  h.onMessage([&since, &format, &ukfFusion, &estimates, &fusedGroundTruth](
                  uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                  uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event

    bool verbose = true;
    if (length && length > 2 && data[0] == '4' && data[1] == '2') {
      auto s = hasData(string(data));
      if (s != "") {
        auto j = json::parse(s);

        string event = j[0].get<string>();

        if (event == "telemetry") {
          // j[1] is the data JSON object

          string sensor_measurement = j[1]["sensor_measurement"];

          Event<T> event;
          istringstream iss(sensor_measurement);
          long long timestamp;

          // reads first element from the current line
          string sensor_type;
          iss >> sensor_type;
          if (sensor_type.compare("L") == 0) {
            event.type = EventType::LIDAR;
            event.data = Matrix<T, Dynamic, 1>::Zero(2, 1);
            float px;
            float py;
            iss >> px;
            iss >> py;
            event.data << px, py;
            iss >> timestamp;
            if (since == 0) since = timestamp;
            event.time = Utils::timestampToSeconds(timestamp, since);

            if (verbose)
              cout << "lidar event" << endl
                   << event.data.transpose().format(format) << endl;

          } else if (sensor_type.compare("R") == 0) {
            event.type = EventType::RADAR;
            event.data = Matrix<T, Dynamic, 1>::Zero(3, 1);
            float ro;
            float theta;
            float ro_dot;
            iss >> ro;
            iss >> theta;
            iss >> ro_dot;
            event.data << ro, theta, ro_dot;
            iss >> timestamp;
            if (since == 0) since = timestamp;
            event.time = Utils::timestampToSeconds(timestamp, since);

            if (verbose)
              cout << "radar event" << endl
                   << event.data.transpose().format(format) << endl;
          }
          float x_gt;
          float y_gt;
          float vx_gt;
          float vy_gt;
          iss >> x_gt;
          iss >> y_gt;
          iss >> vx_gt;
          iss >> vy_gt;
          Matrix<T, 4, 1> groundTruth(4);
          groundTruth(0) = x_gt;
          groundTruth(1) = y_gt;
          groundTruth(2) = vx_gt;
          groundTruth(3) = vy_gt;
          fusedGroundTruth.push_back(groundTruth);

          /* Update the fused filter */
          ukfFusion.predictAndUpdate(event);

          /* Show our state estimate and the ground truth */
          if (verbose) {
            cout << "true state    " << endl
                 << groundTruth.transpose().format(format) << endl;
            cout << "state estimate" << endl
                 << ukfFusion.x.transpose().format(format) << endl;
            cout << "state covariance" << endl
                 << ukfFusion.P.format(format) << endl;
          }

          /* If the state estimator is predicting the (x,y) values
           * of the velocity vector, then we can continue as normal.
           * If, instead, the estimator is predicting the magnitude of the
           * velocity and the yaw angle, then we need a small conversion
           * since the ground truth values are reported as (px, py, vx, vy) */
          Matrix<T, 4, 1> estimate;
          T p_x = estimate(0) = ukfFusion.x(0);
          T p_y = estimate(1) = ukfFusion.x(1);
          if (x_size == 5) {
            /* If the state has five elements, then we are using the ctrv model.
             * That means that we need to convert the absolute velocity and
             * bearing angle to cartesian velocities vx and vy so that we can
             * compare with the ground truth */
            T v = ukfFusion.x(2);
            T psi = ukfFusion.x(3);
            estimate(2) = v * cos(psi);
            estimate(3) = v * sin(psi);

          } else {
            /* If the state has four elements, then we are using the constant
             * velocity model */
            estimate(2) = ukfFusion.x(2);
            estimate(3) = ukfFusion.x(3);
          }
          estimates.push_back(estimate);

          /* Just for fun, compute the root mean-squared error */
          Matrix<T, 4, 1> rmse = Utils::rmse(estimates, fusedGroundTruth);
          cout << "rmse" << rmse.transpose().format(format) << endl;

          json msgJson;
          msgJson["estimate_x"] = p_x;
          msgJson["estimate_y"] = p_y;
          msgJson["rmse_x"] = rmse(0);
          msgJson["rmse_y"] = rmse(1);
          msgJson["rmse_vx"] = rmse(2);
          msgJson["rmse_vy"] = rmse(3);
          auto msg = "42[\"estimate_marker\"," + msgJson.dump() + "]";
          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }
      } else {
        string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }
  });

  // We don't need this since we're not using HTTP but if it's removed the
  // program doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data,
                     size_t, size_t) {
    const string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1) {
      res->end(s.data(), s.length());
    } else {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    cout << "Connected!!!" << endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    cout << "Disconnected" << endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    cout << "Listening to port " << port << endl;
  } else {
    cerr << "Failed to listen to port" << endl;
    return -1;
  }
  h.run();
}
