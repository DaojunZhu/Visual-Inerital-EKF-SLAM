#include "ros/ros.h"
#include "VIEKFSLAM2/vi_ekfslam.hpp"

int main(int argc,char** argv)
{
  
    ros::init(argc,argv,"vi_ekfslam");
    ros::NodeHandle nh;

    vi_ekfslam::VIEkfSlam viEkfSlam(nh);
    viEkfSlam.initialize();

    ros::spin();
    return 0;
}