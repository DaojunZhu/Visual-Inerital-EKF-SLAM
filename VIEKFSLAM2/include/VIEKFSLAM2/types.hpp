#pragma once

#include <vector>
#include <opencv2/core.hpp>
#include <eigen3/Eigen/Core>

namespace vi_ekfslam{

struct MarkerMeasData
{
    int id;
    std::vector<cv::Point2f> corners;
};

struct CameraIMUParameters{
    //camera intrinsics and distortion parameters
    cv::Mat camera_intrinsics;
    cv::Mat distortion_coeffs;
    //the transform from camera frame to imu frame
    Eigen::Vector3d t_imu_cam;
    Eigen::Matrix3d R_imu_cam;

    //Process noise
    double gyro_noise;   // gn
    double acc_noise;    // an
    double gyro_bias_noise;  // gw
    double acc_bias_noise;   // aw

    //Gravity vector in the world frame(NED)
    Eigen::Vector3d gravity;
};

extern CameraIMUParameters camera_imu_parameters;

struct MarkerParemeters{
    //The aruco marker size in metric unit
    double size;

    //the four coordinates of marker's four corners expressed in marker frame
    //started by top-left corner ,then top right...
    Eigen::Vector3d M_c0;
    Eigen::Vector3d M_c1;
    Eigen::Vector3d M_c2;
    Eigen::Vector3d M_c3;

    //the observation noise in nominal image plane
    double observation_noise;

    double initial_position_noise;
    double initla_orientation_noise;
};

extern MarkerParemeters marker_parameters;

}


