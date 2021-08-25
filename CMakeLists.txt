cmake_minimum_required(VERSION 3.11)

project(kalman_filters)

include(FetchContent)

set(CMAKE_CXX_STANDARD 11)

set(EIGEN_BUILD_DOC OFF)
set(BUILD_TESTING OFF)
FetchContent_Declare(_eigen_
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0)
FetchContent_MakeAvailable(_eigen_)

add_executable(ukf src/main.cpp)
add_executable(ukf_test src/main_test.cpp)

target_link_libraries(ukf PRIVATE Eigen3::Eigen)
target_link_libraries(ukf_test PRIVATE Eigen3::Eigen)

target_link_libraries(ukf PRIVATE z ssl uv uWS)
