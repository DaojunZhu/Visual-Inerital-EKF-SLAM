#pragma once

#include <eigen3/Eigen/Dense>

namespace vi_ekfslam{

inline Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d& v){
    Eigen::Matrix3d m = Eigen::Matrix3d::Zero();
    m(0,1) = -v(2);
    m(0,2) = v(1);
    m(1,0) = v(2);
    m(1,2) = -v(0);
    m(2,0) = -v(1);
    m(2,1) = v(0);
    return m;
}

//ref Kinematics (115)
inline Eigen::Matrix3d quaterion2Rotation(const Eigen::Vector4d& q){
    Eigen::Matrix3d R;
    double qw = q(0);
    double qx = q(1);
    double qy = q(2);
    double qz = q(3);
    R << qw*qw+qx*qx-qy*qy-qz*qz, 2*(qx*qy-qw*qz),2*(qx*qz+qw*qy),
        2*(qx*qy+qw*qz), qw*qw-qx*qx+qy*qy-qz*qz, 2*(qy*qz-qw*qx),
        2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), qw*qw-qx*qx-qy*qy+qz*qz;
    return R;
}

inline Eigen::Vector4d delta_q(const Eigen::Vector3d& w_dt){
    Eigen::Vector4d d_q;
    d_q(0) = 1;
    d_q.tail(3) = 0.5 * w_dt;
    return d_q;
}

}

