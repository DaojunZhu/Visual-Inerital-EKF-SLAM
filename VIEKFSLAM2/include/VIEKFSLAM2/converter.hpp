#ifndef CONVERTER_H
#define CONVERTER_H

#include <geometry_msgs/Vector3.h>
#include <eigen3/Eigen/Dense>

#include <opencv2/core.hpp>

namespace vi_ekfslam{


inline Eigen::Vector3d vectorMsg2Eigen(const geometry_msgs::Vector3& msg){
    Eigen::Vector3d v;
    v(0) = msg.x;
    v(1) = msg.y;
    v(2) = msg.z;
    return v;
}

inline void vectorCv2Eigen(const cv::Mat& v_c, Eigen::Vector3d& v_e){
    v_e(0) = v_c.at<double>(0,0);
    v_e(1) = v_c.at<double>(1,0);
    v_e(2) = v_c.at<double>(2,0);
}

inline void vectorCv2Eigen(const cv::Vec3d& v_c, Eigen::Vector3d& v_e){
    v_e(0) = v_c(0);
    v_e(1) = v_c(1);
    v_e(2) = v_c(2);
}

inline void matrixCv2Eigen(const cv::Mat& m_c,Eigen::Matrix3d& m_e){
    m_e << m_c.at<double>(0,0),m_c.at<double>(0,1),m_c.at<double>(0,2),
        m_c.at<double>(1,0),m_c.at<double>(1,1),m_c.at<double>(1,2),
        m_c.at<double>(2,0),m_c.at<double>(2,1),m_c.at<double>(2,2);
}

}

#endif