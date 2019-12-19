cmake_minimum_required(VERSION 2.8.3)
project(VIEKFSLAM2)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  tf
  geometry_msgs
  sensor_msgs
  nav_msgs
  std_msgs
  cv_bridge
  tf_conversions
  eigen_conversions
)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)


catkin_package(
)

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(vi_ekfslam_node src/vi_ekfslam_node.cpp src/vi_ekfslam.cpp)
target_link_libraries(vi_ekfslam_node ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(create_aruco_markers src/create_aruco_markers.cpp)
target_link_libraries(create_aruco_markers ${OpenCV_LIBRARIES})