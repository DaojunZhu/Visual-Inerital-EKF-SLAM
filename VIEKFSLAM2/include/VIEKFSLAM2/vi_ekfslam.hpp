#pragma once

#include <eigen3/Eigen/Dense>

#include <opencv2/aruco.hpp>
#include <opencv2/core.hpp>

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h>
#include <tf/transform_broadcaster.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include "state.hpp"
#include "types.hpp"

namespace vi_ekfslam{


class VIEkfSlam{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    VIEkfSlam(::ros::NodeHandle& nh);
    VIEkfSlam(const VIEkfSlam&) = delete;
    VIEkfSlam operator=(const VIEkfSlam&) = delete;
    
    ~VIEkfSlam(){}

    //initialize the vi-ekfslam
    bool initialize();
    
    typedef std::shared_ptr<VIEkfSlam> Ptr;
    typedef std::shared_ptr<const VIEkfSlam> ConstPtr;

private:

    struct StateServer
    {
        IMUState imu_state;
        FeatureMap fature_states;

        //State covariance matrix
        Eigen::MatrixXd state_cov;
        Eigen::Matrix<double,12,12> continuous_noise_cov;
    };

    // Load parameters from the parameter server
    bool loadParameters();

    //Create ros publisher and subscribers
    bool createRosIO();

    //Callback function for the imu message
    void imuCallback(const sensor_msgs::ImuConstPtr& imu_msg);

    //Callback function for camera message
    void camCallback(const sensor_msgs::ImageConstPtr& img_msg);

    //Publish the result of VIEKFSLAM
    void publish(const ros::Time& time);
    
    //Initialize the IMU bias and initial orientation
    //  based on the first few IMU readings.
    // static initialization
    void initializeGravityAndBias();

    //process IMU measurement
    void processIMUData(const double& time,
        const Eigen::Vector3d& m_gyro,const Eigen::Vector3d& m_acc);

    //predict the nominal IMU state
    void predictNominalState(const double& dt,
        const Eigen::Vector3d& gyro,const Eigen::Vector3d& acc);

    //augment a new detected marker to state
    void stateAugmentation(const MarkerMeasData& marker_data);

    //process image measurement
    void processMarkerData(const std::vector<MarkerMeasData>& marker_meas);

    // compute the marker feature measuement jacobian
    void measurementJacobian(int index, const MarkerMeasData& meas, 
        Eigen::MatrixXd& H_x_i, Eigen::VectorXd& r_i);

    //measurement update
    void measurementUpdate(const Eigen::MatrixXd& H_x, const Eigen::VectorXd& r);

    //update state
    void updateState(const Eigen::VectorXd& delta);

    //detect aruco marker 
    void detectArucoMarker(const cv::Mat& image,std::vector<MarkerMeasData>& markers);

    //undistort points
    void undistortPoints(const std::vector<cv::Point2f>& pts_in,
                        const cv::Vec4d& intrinsics,
                        const cv::Vec4d& distortion_coeffs,
                        std::vector<cv::Point2f>& pts_out);

    //draw aruco marker pose and covariance in the rviz
    void drawRosMarker(int id, const ros::Time& time, 
        const Eigen::Vector3d& position, const Eigen::Quaterniond& orientation,
        const Eigen::Matrix3d& cov_pp, 
        visualization_msgs::Marker& marker);


private:

    //State vector
    StateServer state_server;

    // Indicate if the gravity vector is set.
    bool is_gravity_set;

    //beginning IMU msgs for initialization
    std::vector<sensor_msgs::Imu> imu_buffer;

    //opencv aruco marker dictionary for detection
    cv::Ptr<cv::aruco::Dictionary> dictionary;

    //Ros node handle
    ros::NodeHandle nh;

    //subscribers and publishers
    ros::Subscriber imu_sub;
    ros::Subscriber img_sub;
    ros::Publisher odom_pub;
    ros::Publisher debug_img_pub;
    ros::Publisher feature_map_pub;
    ros::Publisher features_pose_pub;
    ros::Publisher trajectory_pub;
    tf::TransformBroadcaster tf_pub;

    //Frame id
    std::string world_frame_id;
    std::string body_frame_id;

    //Topic name
    std::string imu_topic;
    std::string img_topic;

    //debug image for visualization
    cv::Mat debug_image;
    
    
};


}

