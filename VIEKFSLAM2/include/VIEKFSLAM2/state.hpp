#pragma once

#include <eigen3/Eigen/Dense>
#include <map>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>

#include "types.hpp"
#include "converter.hpp"

namespace vi_ekfslam{

//IMU state 
struct IMUState{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    //Time when the state is recorded
    double time;

    //Orientation
    //use hamilton quaternion representation(qw,qv)
    //Take a vector from the IMU(body) frame 
    //to the world frame
    Eigen::Quaterniond orientation;

    //Position of the IMU frame in the world frame
    Eigen::Vector3d position;

    //Velocity of the IMU frame in the world frame
    Eigen::Vector3d velocity;

    //gyroscope and acclerometer bias to be estimated
    Eigen::Vector3d gyro_bias;
    Eigen::Vector3d acc_bias;

    //constructor
    IMUState(): time(0),
        orientation(Eigen::Quaterniond::Identity()) ,
        position(Eigen::Vector3d::Zero()),
        velocity(Eigen::Vector3d::Zero()),
        gyro_bias(Eigen::Vector3d::Zero()),
        acc_bias(Eigen::Vector3d::Zero()) {}

};


struct FeatureState{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    //The marker feature id
    int id;

    FeatureState(int _id) : id(_id){}
    
    //Rotation 
    //Transform a vector from the marker frame to the world frame
    Eigen::Quaterniond orientation;

    //The marker position in the world frame
    Eigen::Vector3d position;

    //initialize the pose of the marker
    static inline void initializePose(const IMUState& imu_state, 
        const MarkerMeasData& meas,
        Eigen::Quaterniond& rotation, 
        Eigen::Vector3d& position);
};

typedef std::vector<FeatureState> FeatureMap;

void FeatureState::initializePose(const IMUState& imu_state, 
        const MarkerMeasData& meas,
        Eigen::Quaterniond& rotation, 
        Eigen::Vector3d& position){

    //opencv function
    cv::Vec3d rvec,tvec;
    //The corner point have been undistorted to nomalized plan
    std::vector<std::vector<cv::Point2f>> corners_in;
    std::vector<cv::Vec3d> rvecs,tvecs;
    corners_in.push_back(meas.corners);
    double fx = camera_imu_parameters.camera_intrinsics.at<double>(0,0);
    double fy = camera_imu_parameters.camera_intrinsics.at<double>(1,0);
    double cx = camera_imu_parameters.camera_intrinsics.at<double>(2,0);
    double cy = camera_imu_parameters.camera_intrinsics.at<double>(3,0);
    // cv::Mat K = (cv::Mat_<double>(3,3) << fx,0,cx,0,fy,cy,0,0,1 );
    cv::Mat K = cv::Mat::eye(3,3,CV_64F);
    cv::Mat distcoeffs = (cv::Mat_<double>(4,1) << 0.0,0.0,0.0,0.0,0.0);
    cv::aruco::estimatePoseSingleMarkers(corners_in,marker_parameters.size,
        K ,distcoeffs,rvecs,tvecs);

    rvec = rvecs[0];
    tvec = tvecs[0];
    Eigen::Vector3d t_cam_marker;
    vectorCv2Eigen(tvec,t_cam_marker);
    cv::Mat R;
    cv::Rodrigues(rvec,R);
    Eigen::Matrix3d R_cam_marker;
    matrixCv2Eigen(R,R_cam_marker);
    rotation = Eigen::Quaterniond( imu_state.orientation.toRotationMatrix() * 
        camera_imu_parameters.R_imu_cam * R_cam_marker );
    rotation.normalize();
    position = imu_state.orientation.toRotationMatrix()*(camera_imu_parameters.R_imu_cam*
        t_cam_marker+camera_imu_parameters.t_imu_cam) + 
        imu_state.position;

}

}
