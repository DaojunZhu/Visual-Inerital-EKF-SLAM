
#include "VIEKFSLAM2/vi_ekfslam.hpp"
#include "VIEKFSLAM2/converter.hpp"
#include "VIEKFSLAM2/math_utils.hpp"

#include <cv_bridge/cv_bridge.h>
//conversion between eigen and tf class
#include <tf_conversions/tf_eigen.h>
//conversion between eigen and ros msg
#include <eigen_conversions/eigen_msg.h>

#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseArray.h>

using namespace std;
using namespace Eigen;
using namespace cv;

namespace vi_ekfslam{


CameraIMUParameters camera_imu_parameters;
MarkerParemeters marker_parameters;


VIEkfSlam::VIEkfSlam(ros::NodeHandle& pnh)
    : is_gravity_set(false),
    nh(pnh){
        dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);

        
    }

bool VIEkfSlam::loadParameters(){
    nh.param<string>("world_frame_id",world_frame_id,"world");
    nh.param<string>("body_frame_id",body_frame_id,"robot");
    nh.param<string>("imu_topic",imu_topic,"imu");
    nh.param<string>("img_topic",img_topic,"image");
    
    nh.param<double>("noise/gyro",camera_imu_parameters.gyro_noise,0.001);
    nh.param<double>("noise/acc",camera_imu_parameters.acc_noise,0.01);
    nh.param<double>("noise/gyro_bias",camera_imu_parameters.gyro_bias_noise,0.001);
    nh.param<double>("noise/acc_bias",camera_imu_parameters.acc_bias_noise,0.01);

    camera_imu_parameters.gyro_noise *= camera_imu_parameters.gyro_noise;
    camera_imu_parameters.acc_noise *= camera_imu_parameters.acc_noise;
    camera_imu_parameters.gyro_bias_noise *= camera_imu_parameters.gyro_bias_noise;
    camera_imu_parameters.acc_bias_noise *= camera_imu_parameters.acc_bias_noise;

    nh.param<double>("initial_state/velocity/x",
      state_server.imu_state.velocity(0), 0.0);
    nh.param<double>("initial_state/velocity/y",
        state_server.imu_state.velocity(1), 0.0);
    nh.param<double>("initial_state/velocity/z",
        state_server.imu_state.velocity(2), 0.0);

    double gyro_bias_cov, acc_bias_cov, velocity_cov;
    nh.param<double>("initial_covariance/velocity",
        velocity_cov, 100);
    nh.param<double>("initial_covariance/gyro_bias",
        gyro_bias_cov, 1e-4);
    nh.param<double>("initial_covariance/acc_bias",
        acc_bias_cov, 1e-2);

    nh.param<double>("marker_size",marker_parameters.size,0.101);
    nh.param<double>("observation_noise",marker_parameters.observation_noise,0.0001);
    double s = 0.5 * marker_parameters.size;
    marker_parameters.M_c0 = Vector3d(-s,s,0);
    marker_parameters.M_c1 = Vector3d(s,s,0);
    marker_parameters.M_c2 = Vector3d(s,-s,0);
    marker_parameters.M_c3 = Vector3d(-s,-s,0);

    state_server.state_cov = MatrixXd::Zero(15,15);
    for(int i = 6; i < 9 ; ++i)
        state_server.state_cov(i,i) = velocity_cov;
    for(int i = 9; i < 12; ++i)
        state_server.state_cov(i,i) = acc_bias_cov;
    for(int i = 12; i < 15; ++i)
        state_server.state_cov(i,i) = gyro_bias_cov;

    //read parameters in the config file
    string config_file;
    if(!nh.getParam("config_file",config_file)){
        ROS_ERROR("The config_file parameter not set.");
        return false;
    }
    FileStorage fs;
    fs.open(config_file,FileStorage::READ);
    if(!fs.isOpened()){
        ROS_ERROR("Failed to open config file.");
        return false;
    }
    cv::Mat R,t;
    fs["R_imu_cam"] >> R;
    matrixCv2Eigen(R,camera_imu_parameters.R_imu_cam);
    fs["t_imu_cam"] >> t;
    vectorCv2Eigen(t,camera_imu_parameters.t_imu_cam);
    fs["intrinsics"] >> camera_imu_parameters.camera_intrinsics;
    fs["distortion_coeffs"] >> camera_imu_parameters.distortion_coeffs;

    ROS_INFO("Finish reading config file...");
    cout << "R_imu_cam: " << endl << camera_imu_parameters.R_imu_cam << endl;
    cout << "t_imu_cam: " << endl << camera_imu_parameters.t_imu_cam << endl;
    cout << "camera intrinsics: " << endl << camera_imu_parameters.camera_intrinsics << endl;
    cout << "distortion_coeffs: " << endl << camera_imu_parameters.distortion_coeffs << endl;

    fs.release();
    
    return true;
}

bool VIEkfSlam::createRosIO(){
    odom_pub = nh.advertise<nav_msgs::Odometry>("odom",10);
    debug_img_pub = nh.advertise<sensor_msgs::Image>("debug_img",10);
    feature_map_pub = nh.advertise<visualization_msgs::MarkerArray>("features",10);
    features_pose_pub = nh.advertise<geometry_msgs::PoseArray>("features_pose",10);
    trajectory_pub = nh.advertise<nav_msgs::Path>("traj",10);
    imu_sub = nh.subscribe(imu_topic,100,&VIEkfSlam::imuCallback,this);
    img_sub = nh.subscribe(img_topic,40,&VIEkfSlam::camCallback,this);
    return true;
}

bool VIEkfSlam::initialize(){
    if(!loadParameters()) return false;
    ROS_INFO("Finishing loading ROS parameters....");

    //Initialize state server
    state_server.continuous_noise_cov = Matrix<double,12,12>::Zero();
    state_server.continuous_noise_cov.block<3,3>(0,0) = 
        camera_imu_parameters.gyro_noise * Matrix3d::Identity();
    state_server.continuous_noise_cov.block<3,3>(3,3) = 
        camera_imu_parameters.acc_bias_noise * Matrix3d::Identity();
    state_server.continuous_noise_cov.block<3,3>(6,6) = 
        camera_imu_parameters.gyro_bias_noise * Matrix3d::Identity();
    state_server.continuous_noise_cov.block<3,3>(6,6) = 
        camera_imu_parameters.acc_bias_noise * Matrix3d::Identity();
    
    //imu initial position
    state_server.imu_state.position(0) = 0.0;
    state_server.imu_state.position(1) = 0.0;
    state_server.imu_state.position(2) = 0.0;

    // state_server.state_cov.block<3,3>(0,0) = 100.0 * Matrix3d::Identity();
    // state_server.state_cov.block<2,2>(3,3) = 10. * M_PI/180. * 10. * M_PI/180. * Matrix2d::Identity();
    // state_server.state_cov(5,5) = 100. *   M_PI/180. * 100. *   M_PI/180. ;


    if(!createRosIO()) return false;
    ROS_INFO("Finish creating ROS IO...");

    return true;
}

void VIEkfSlam::initializeGravityAndBias(){

    Vector3d sum_angular_vel = Vector3d::Zero();
    Vector3d sum_linear_acc = Vector3d::Zero();

    for(const auto& imu_msg : imu_buffer){
        Vector3d angular_vel = 
            vectorMsg2Eigen(imu_msg.angular_velocity);
        Vector3d linear_acc = 
            vectorMsg2Eigen(imu_msg.linear_acceleration);

        sum_angular_vel += angular_vel;
        sum_linear_acc += linear_acc;
    }

    state_server.imu_state.gyro_bias = sum_angular_vel / imu_buffer.size();
    //this is gravity in the imu frame
    Vector3d gravity_imu = sum_linear_acc / imu_buffer.size();

    double gravity_norm = gravity_imu.norm();
    camera_imu_parameters.gravity = Vector3d(0.0,0.0,-gravity_norm);

    //q_w_i0 convert a vector from IMU frame to world frame(NED)
    // initial IMU orientation estimation
    Quaterniond q_w_i0 = Quaterniond::FromTwoVectors(gravity_imu,-camera_imu_parameters.gravity);
    state_server.imu_state.orientation = q_w_i0;
}

void VIEkfSlam::imuCallback(const sensor_msgs::ImuConstPtr& imu_msg){
    //wait for the enough imu msg to initialize 
    //gravity and bias
    static int imu_msg_cntr = 0;
    if(!is_gravity_set){
        imu_buffer.push_back(*imu_msg);
        if(imu_msg_cntr++ < 200) return;
        initializeGravityAndBias();

        cout << "initial acc bias: " << endl << "  " << 
            state_server.imu_state.acc_bias.transpose() << endl; 
        cout << "initial gyro bias: " << endl << "  " << 
            state_server.imu_state.gyro_bias.transpose() << endl;
        cout << "initla gravity in world frame: " << endl << "  " <<
            camera_imu_parameters.gravity.transpose() << endl;
        cout << "initial imu orientation: " << endl << "  " << 
            state_server.imu_state.orientation.toRotationMatrix() << endl;
        
        is_gravity_set = true;
        ROS_INFO("[imuCallback] initialized done.");
        state_server.imu_state.time = imu_msg->header.stamp.toSec();
    }
    else{
        // process imu data
        double time = imu_msg->header.stamp.toSec();
        Vector3d m_gyro = vectorMsg2Eigen(imu_msg->angular_velocity);
        Vector3d m_acc = vectorMsg2Eigen(imu_msg->linear_acceleration);
        // cout << "process IMU measurement ..." << endl;
        processIMUData(time,m_gyro,m_acc);
    }
    
}

void VIEkfSlam::camCallback(const sensor_msgs::ImageConstPtr& img_msg){

    if(!is_gravity_set){
        ROS_INFO("[camCallback] gravity and bias not initialized...");
        return;
    }
    
    //get the current image
    cv_bridge::CvImageConstPtr image_ptr = 
        cv_bridge::toCvShare(img_msg,sensor_msgs::image_encodings::MONO8);
    
    vector<MarkerMeasData> marker_meas;

    ros::Time start_time = ros::Time::now();

    //detect aruco marker
    detectArucoMarker(image_ptr->image,marker_meas);

    double marker_detection_time = (ros::Time::now()-start_time).toSec();
    ROS_INFO("marker detection time: %f" ,marker_detection_time);
    
    //process marker measurement update
    processMarkerData(marker_meas);

    //publish ros topics and tf 
    publish(img_msg->header.stamp);
}


void VIEkfSlam::processIMUData(const double& time,
        const Eigen::Vector3d& m_gyro,
        const Eigen::Vector3d& m_acc){

    //nomial measurement
    IMUState& imu_state = state_server.imu_state;
    Vector3d gyro = m_gyro - imu_state.gyro_bias;
    Vector3d acc = m_acc - imu_state.acc_bias;
    double dtime = time - imu_state.time;
    double dtime2 = dtime * dtime;

    //ref: Kinematics P88
    //x_dot = F * x + G * n;
    //Phi = exp(F*dt) , approximate it using block wise, 2 order
    Matrix<double,15,15> Phi = Matrix<double,15,15>::Zero();
    Matrix<double,15,12> G = Matrix<double,15,12>::Zero();

    Matrix3d R = state_server.imu_state.orientation.toRotationMatrix();
    Phi.block<3,3>(0,0) = Matrix3d::Identity();
    Phi.block<3,3>(0,3) = -0.5 * R * skewSymmetric(acc) * dtime2;
    Phi.block<3,3>(0,6) = Matrix3d::Identity() * dtime;
    Phi.block<3,3>(0,9) = -0.5 * R * dtime2;
    Phi.block<3,3>(3,3) = quaterion2Rotation(delta_q(gyro*dtime)).transpose();
    Phi.block<3,3>(3,12) = -Matrix3d::Identity() * dtime;
    Phi.block<3,3>(6,3) = -R * skewSymmetric(acc) * dtime;
    Phi.block<3,3>(6,6) = Matrix3d::Identity();
    Phi.block<3,3>(6,9) = -R * dtime;
    Phi.block<3,3>(6,12) = 0.5 * R * skewSymmetric(acc) * dtime2;
    Phi.block<3,3>(9,9) = Matrix3d::Identity();
    Phi.block<3,3>(12,12) = Matrix3d::Identity();

    G.block<3,3>(3,0) = -Matrix3d::Identity();
    G.block<3,3>(6,3) = -R;
    G.block<3,3>(9,9) = Matrix3d::Identity();
    G.block<3,3>(12,6) = Matrix3d::Identity();

    //predict nominal state
    predictNominalState(dtime,gyro,acc);

    //Propagate the state covariance matrix
    Matrix<double,15,15> Q = Phi*G*state_server.continuous_noise_cov*
            G.transpose()*Phi.transpose()*dtime;
    state_server.state_cov.block<15,15>(0,0) = Phi*
        state_server.state_cov.block<15,15>(0,0)*Phi.transpose() + Q;

    if(state_server.fature_states.size() > 0){
        state_server.state_cov.topRightCorner(
            15,state_server.state_cov.cols()-15) = Phi * 
            state_server.state_cov.topRightCorner(15,state_server.state_cov.cols()-15);
        state_server.state_cov.bottomLeftCorner(
            state_server.state_cov.rows()-15,15) = state_server.state_cov.bottomLeftCorner(
                state_server.state_cov.rows()-15,15)*Phi.transpose();
    }

    //ensure the state covariance matrix to be symmetric
    MatrixXd state_cov_fixed = (state_server.state_cov +
      state_server.state_cov.transpose()) / 2.0;
    state_server.state_cov = state_cov_fixed;

    state_server.imu_state.time = time;

}


void VIEkfSlam::predictNominalState(const double& dt,
        const Eigen::Vector3d& gyro,
        const Eigen::Vector3d& acc){

    double gyro_norm = gyro.norm();

    //the imu nominal state in time point tn
    Quaterniond& q = state_server.imu_state.orientation;
    Vector3d& p = state_server.imu_state.position;
    Vector3d& v = state_server.imu_state.velocity;

    //The propagate orientation in tn+dt(tn+1) , tn+dt/2
    Quaterniond dq_dt , dq_dt2;
    if(gyro_norm > 1e-5){
        Vector3d dqv = gyro / gyro_norm * sin(0.5*dt*gyro_norm);
        dq_dt = q * Quaterniond(cos(gyro_norm*0.5*dt),dqv(0),dqv(1),dqv(2));
        Vector3d dqv2 = gyro / gyro_norm * sin(0.25*dt*gyro_norm);
        dq_dt2 = q * Quaterniond(cos(gyro_norm*0.25*dt),dqv2(0),dqv2(1),dqv2(2));
    }else{
        Vector3d dqv = 0.5 * gyro * dt;
        dq_dt = q * Quaterniond(1,dqv(0),dqv(1),dqv(2));
        Vector3d dqv2 = 0.25 * gyro * dt;
        dq_dt2 = q * Quaterniond(1,dqv2(0),dqv2(1),dqv2(2));
    }

    //Convert the quaternion to rotation matrix
    Matrix3d dR_dt = dq_dt.toRotationMatrix();
    Matrix3d dR_dt2 = dq_dt.toRotationMatrix();
    
    //k1 = f(tn,xn)
    Vector3d k1_v_dot = q*acc + camera_imu_parameters.gravity;
    Vector3d k1_p_dot = v;

    //k2 = f(tn+dt/2,xn+k1*dt/2)
    Vector3d k1_v = v + k1_v_dot * dt * 0.5;
    Vector3d k2_v_dot = dR_dt2 * acc + camera_imu_parameters.gravity;
    Vector3d k2_p_dot = k1_v;

    //k3 = f(tn+dt/2,xn+k2*dt/2)
    Vector3d k2_v = v + k2_v_dot * dt * 0.5;
    Vector3d k3_v_dot = dR_dt2 * acc + camera_imu_parameters.gravity;
    Vector3d k3_p_dot = k2_v;

    //k4 = f(tn+dt,xn+k3*dt)
    Vector3d k3_v = v + k3_v_dot * dt;
    Vector3d k4_v_dot = dR_dt * acc + camera_imu_parameters.gravity;
    Vector3d k4_p_dot = k3_v;

    q = dq_dt;
    q.normalize();
    v = v + dt/6*(k1_v_dot+2*k2_v_dot+2*k3_v_dot+k4_v_dot);
    p = p + dt/6*(k1_p_dot+2*k2_p_dot+2*k3_p_dot+k4_p_dot);
}

void VIEkfSlam::detectArucoMarker(const cv::Mat& image,
    std::vector<MarkerMeasData>& markers){

    markers.clear();
    // The number of aruco marker detected
    int detect_marker_num = 0;

    //store detected marker id map to observation times
    //a marker whos observation times greater than 5 times
    // will be considered a valid marker measurement
    static map<int,int> map_id2obstimes;

    vector<int> marker_ids;
    vector<vector<cv::Point2f>> marker_corners;

    image.copyTo(debug_image);

    //detection
    cv::aruco::detectMarkers(image,dictionary,marker_corners,marker_ids);
    detect_marker_num = marker_ids.size();
    cv::aruco::drawDetectedMarkers(debug_image,marker_corners,marker_ids);

    //Make marker measurement data
    for(int i = 0; i < detect_marker_num; ++i){

        if(map_id2obstimes.find(marker_ids[i]) == map_id2obstimes.end()){
            map_id2obstimes[marker_ids[i]] = 1;
        }else{
            // a valid marker measurement
            if(map_id2obstimes[marker_ids[i]] >= 5 ){
                MarkerMeasData marker_data;
                marker_data.id = marker_ids[i];
                // undistort points to normalized plane
                undistortPoints(marker_corners[i],camera_imu_parameters.camera_intrinsics,
                    camera_imu_parameters.distortion_coeffs,marker_data.corners);
                markers.push_back(marker_data);

            }else{
                ++map_id2obstimes[marker_ids[i]];
            }
        }

        
    }
}

void VIEkfSlam::processMarkerData(
    const std::vector<MarkerMeasData>& marker_meas){
    
    //return if empty marker detected
    if(marker_meas.empty()) return;
    //the marker counter that detected
    int marker_meas_cntr = marker_meas.size();

    //the marker measurement jacobian 
    //and residual for old markers
    vector<MatrixXd> v_H_x;
    vector<VectorXd> v_r;

    vector<MarkerMeasData> new_marker_datas;

    for(auto marker_data : marker_meas){
        
        // auto find_iter = state_server.fature_states.find(marker_data.id);
        auto find_iter = std::find_if(state_server.fature_states.begin(),
        state_server.fature_states.end(),[marker_data](const FeatureState& f){
            return f.id == marker_data.id;});
        //This is a new marker
        if(find_iter == state_server.fature_states.end()){
            cout << "new marker detected. marker id: " << marker_data.id << endl;
            //add the new marker to state 
            stateAugmentation(marker_data);
            // new_marker_datas.push_back(marker_data);
        }
        //This is an old marker
        else{
            cout << "old marker detected. marker id: " << marker_data.id << endl;
            //The ordered slam index of this marker
            int slam_index = std::distance(
                state_server.fature_states.begin(),find_iter);
            // cout << "slam_index : " << slam_index << endl;
            MatrixXd H_x_i = MatrixXd::Zero(8,
                    15+6*state_server.fature_states.size());
            VectorXd r_i = VectorXd::Zero(8);
            //compute jacobian
            measurementJacobian(slam_index,marker_data,H_x_i,r_i);
            v_H_x.push_back(H_x_i);
            v_r.push_back(r_i);

            // cout << "measurement jacobian : " << endl << 
            //     H_x_i << endl;
            // cout << "measurement error: " << endl << 
            //     r_i << endl;
            ros::Time start_time = ros::Time::now();

            measurementUpdate(H_x_i,r_i);

            double measurement_update_time = (ros::Time::now()-start_time).toSec();
            ROS_INFO("measurement update time: %f",measurement_update_time);

            // //debug
            // cout << "imu cov: " << endl;
            // cout << state_server.state_cov.block<15,15>(0,0) << endl;
            // for(int i = 0; i < state_server.fature_states.size(); ++i){
            //     cout << "marker " << state_server.fature_states[i].id << " cov: " << endl;
            //     cout << state_server.state_cov.block<6,6>(15+6*i,15+6*i) << endl;

            // }

        }
    }

    // for(auto marker_data : new_marker_datas){
    //     stateAugmentation(marker_data);
    // }

    // if(v_H_x.size() > 0){
    //     int cntr = v_H_x.size();
    //     cout << "v_H_x size : " << cntr << endl;
    //     MatrixXd H = MatrixXd::Zero(8*cntr,
    //             15+6*state_server.fature_states.size());
    //     VectorXd r = VectorXd::Zero(8*cntr);
    //     for(int i = 0; i < cntr; ++i){
    //         H.block(8*i,0,8,
    //             15+6*state_server.fature_states.size()) = v_H_x[i];
    //         r.segment(8*i,8) = v_r[i];
    //     }
    //     cout << "process measurement update..." << endl;
    //     measurementUpdate(H,r);
    //     cout << "measurement update done. " << endl;
    // }

    

// //debug
    // cout << "imu position cov: " << endl;
    // cout << state_server.state_cov.block<3,3>(0,0) << endl;
    // for(int i = 0; i < state_server.fature_states.size(); ++i){
    //     cout << "marker " << state_server.fature_states[i].id << " cov: " << endl;
    //     cout << state_server.state_cov.block<3,3>(15+6*i,15+6*i) << endl;
    // }
    
}

void VIEkfSlam::stateAugmentation(
    const MarkerMeasData& marker_data){
    
    //add new marker to state server
    FeatureState marker_feature(marker_data.id);
    FeatureState::initializePose(state_server.imu_state,marker_data,
        marker_feature.orientation,marker_feature.position);

    state_server.fature_states.push_back(marker_feature);

    //resize the covariance matrix
    size_t old_rows = state_server.state_cov.rows();
    size_t old_cols = state_server.state_cov.cols();
    state_server.state_cov.conservativeResize(old_rows+6,old_cols+6);
    state_server.state_cov.block<6,6>(old_rows,old_cols).setZero();
    //initialize the marker postion covariance 
    state_server.state_cov.block<3,3>(old_rows,old_cols) = 
        100*100*Matrix3d::Identity();
    // initialize the marker orientation covariance
    state_server.state_cov.bottomRightCorner(3,3) = 
        10*10*Matrix3d::Identity();
    state_server.state_cov.block(old_rows,0,6,old_cols).setZero();
    state_server.state_cov.block(0,old_cols,old_rows,6).setZero();
    

}

void VIEkfSlam::measurementJacobian(int index,const MarkerMeasData& meas,
         MatrixXd& H_x_i, VectorXd& r_i){

    const IMUState& imu_state = state_server.imu_state;
    const FeatureState& feature_state = state_server.fature_states[index];

    //the four corners expressed in  marker frame;
    vector<Vector3d> corners_in_marker(4);

    corners_in_marker[0] = marker_parameters.M_c0;
    corners_in_marker[1] = marker_parameters.M_c1;
    corners_in_marker[2] = marker_parameters.M_c2;
    corners_in_marker[3] = marker_parameters.M_c3;

    //current IMU pose and Marker pose estimation
    Matrix3d R_w_i = imu_state.orientation.toRotationMatrix();
    Vector3d t_w_i = imu_state.position;
    Matrix3d R_w_m = feature_state.orientation.toRotationMatrix();
    Vector3d t_w_m = feature_state.position;

    int total_state_size = 15 + 6 * state_server.fature_states.size();

    const Matrix3d& R_i_c = camera_imu_parameters.R_imu_cam;
    const Vector3d& t_i_c = camera_imu_parameters.t_imu_cam;

    for(int i =0; i < 4; ++i){

        //Corner in imu frame
        Vector3d p_i = R_w_i.transpose()*(
                R_w_m*corners_in_marker[i]+t_w_m-t_w_i); 
        //Corner in camera frame
        Vector3d p_c = R_i_c.transpose()*(p_i-t_i_c);

        Matrix<double,2,3> jacobian_uv_xyz;
        // const double& fx = camera_imu_parameters.camera_intrinsics.at<double>(0,0);
        // const double& fy = camera_imu_parameters.camera_intrinsics.at<double>(1,0);
        // jacobian_uv_xyz << fx/p_c(2), 0, -p_c(0)*fx/(p_c(2)*p_c(2)),
        //                     0, fy/p_c(2), -p_c(1)*fy/(p_c(2)*p_c(2));
        jacobian_uv_xyz << 1.0/p_c(2), 0, -p_c(0)/(p_c(2)*p_c(2)),
                             0, 1.0/p_c(2), -p_c(1)/(p_c(2)*p_c(2));
        MatrixXd jacobian_xyz_state= MatrixXd::Zero(3,total_state_size);
        jacobian_xyz_state.block<3,3>(0,0) = -R_i_c.transpose() * R_w_i.transpose();
        jacobian_xyz_state.block<3,3>(0,3) = R_i_c.transpose() * skewSymmetric(p_i);
        jacobian_xyz_state.block<3,3>(0,15+6*index) = R_i_c.transpose() * R_w_i.transpose();
        jacobian_xyz_state.block<3,3>(0,15+6*index+3) = -R_i_c.transpose() * R_w_i.transpose() * 
            R_w_m * skewSymmetric(corners_in_marker[i]);
        MatrixXd tmp_H = MatrixXd::Zero(2,total_state_size);
        tmp_H = jacobian_uv_xyz * jacobian_xyz_state;
        Vector2d tmp_r;
        tmp_r << meas.corners[i].x - p_c(0) / p_c(2),
                meas.corners[i].y - p_c(1) / p_c(2);
        H_x_i.block(2*i,0,2,total_state_size) = tmp_H;
        r_i.segment(2*i,2) = tmp_r;
    }

}

void VIEkfSlam::measurementUpdate(const MatrixXd& H, const VectorXd& r){
    if(H.rows() == 0 || r.rows() == 0) return;

    const MatrixXd& P = state_server.state_cov;
    MatrixXd V = marker_parameters.observation_noise * 
            MatrixXd::Identity(H.rows(),H.rows());
    MatrixXd S = H * P * H.transpose() + V;
    MatrixXd S_inv = S.inverse();
    //kalman gain
    MatrixXd K = P * H.transpose() * S_inv;
    // cout << "S: " << endl << S << endl;
    // cout << "S_inv : " << endl << S_inv << endl;

    VectorXd delta_x = K * r;

    //update state
    updateState(delta_x);
    //update state covariance
    int total_state_size = 15 + 6 * state_server.fature_states.size();
    MatrixXd I_KH = MatrixXd::Identity(
            total_state_size,total_state_size) - K * H;
    // This formular is not stable here , why??
    // state_server.state_cov = I_KH * P * I_KH.transpose() + K * V * K.transpose();
    state_server.state_cov = P - K * S * K.transpose();
    
    MatrixXd state_cov_fixed = (state_server.state_cov +
      state_server.state_cov.transpose()) / 2.0;
    state_server.state_cov = state_cov_fixed;
}

void VIEkfSlam::updateState(const VectorXd& delta){

    IMUState& imu_state = state_server.imu_state;
    imu_state.position += delta.segment(0,3);
    imu_state.orientation *= Quaterniond(AngleAxisd(delta.segment(3,3).norm(),
        delta.segment(3,3).normalized()));
    imu_state.orientation.normalize();
    imu_state.velocity += delta.segment(6,3);
    imu_state.acc_bias += delta.segment(9,3);
    imu_state.gyro_bias += delta.segment(12,3);

    FeatureMap& features = state_server.fature_states;
    int marker_state_size = delta.size() - 15;
    int marker_cntr = marker_state_size / 6;
    for(int i = 0; i < marker_cntr; ++i){
        features[i].position += delta.segment(15+6*i,3);
        features[i].orientation *= Quaterniond(AngleAxisd(delta.segment(15+6*i+3,3).norm(),
            delta.segment(15+6*i+3,3).normalized()));
        features[i].orientation.normalize();
    }
}

void VIEkfSlam::publish(const ros::Time& time){
    const IMUState& imu_state = state_server.imu_state;

    //publish tf
    tf::Vector3 tf_imu_position;
    tf::vectorEigenToTF(imu_state.position,tf_imu_position);
    tf::Quaternion tf_imu_orientation;
    tf::quaternionEigenToTF(imu_state.orientation,tf_imu_orientation);
    tf::Transform T_w_i_tf;
    T_w_i_tf.setOrigin(tf_imu_position);
    T_w_i_tf.setRotation(tf_imu_orientation);
    tf_pub.sendTransform(tf::StampedTransform(T_w_i_tf,time,world_frame_id,body_frame_id));

    //publish the odometry
    nav_msgs::Odometry odom_msg;
    odom_msg.header.stamp = time;
    odom_msg.header.frame_id = world_frame_id;
    odom_msg.child_frame_id = body_frame_id;
    tf::quaternionEigenToMsg(imu_state.orientation,odom_msg.pose.pose.orientation);
    tf::pointEigenToMsg(imu_state.position,odom_msg.pose.pose.position);
    tf::vectorEigenToMsg(imu_state.velocity,odom_msg.twist.twist.linear);
    //convert the covariance for pose
    Matrix3d P_pp = state_server.state_cov.block<3,3>(0,0);
    Matrix3d P_po = state_server.state_cov.block<3,3>(0,3);
    Matrix3d P_op = state_server.state_cov.block<3,3>(3,0);
    Matrix3d P_oo = state_server.state_cov.block<3,3>(3,3);
    Matrix<double,6,6> P_imu_pose = Matrix<double,6,6>::Zero();
    P_imu_pose << P_pp, P_po,P_op,P_oo;
    for(int i = 0; i < 6; ++i)
        for(int j = 0; j < 6; ++j)
            odom_msg.pose.covariance[6*i+j] = P_imu_pose(i,j);
    //convert the covariance for velocity
    Matrix3d P_imu_vel = state_server.state_cov.block<3,3>(6,6);
    for(int i = 0; i < 3; ++i)
        for(int j = 0; j < 3; ++j)
            odom_msg.twist.covariance[3*i+j] = P_imu_vel(i,j);
    //publish odom topic
    odom_pub.publish(odom_msg);

    // // publish debug image
    cv_bridge::CvImage debug_image_msg;
    debug_image_msg.header.stamp = time;
    debug_image_msg.encoding = sensor_msgs::image_encodings::MONO8;
    debug_image_msg.image = debug_image;
    debug_img_pub.publish(debug_image_msg.toImageMsg());
    
    //publish markers
    visualization_msgs::MarkerArray markers;
    geometry_msgs::PoseArray markers_pose_msg;
    markers_pose_msg.header.frame_id = world_frame_id;
    markers_pose_msg.header.stamp = time;
    const FeatureMap& features = state_server.fature_states;
    for(int i = 0; i < features.size(); ++i){
        visualization_msgs::Marker marker;
        drawRosMarker(features[i].id,time,features[i].position,features[i].orientation,
            state_server.state_cov.block<3,3>(15+6*i,15+6*i),marker);
        markers.markers.push_back(marker);
        markers_pose_msg.poses.push_back(marker.pose);
    }
    feature_map_pub.publish(markers);
    features_pose_pub.publish(markers_pose_msg);
    
    //publish the trajectory
    static nav_msgs::Path traj_msg;
    traj_msg.header.frame_id = world_frame_id;
    traj_msg.header.stamp = time;
    geometry_msgs::PoseStamped pose;
    pose.header.stamp = time;
    pose.header.frame_id = world_frame_id;
    pose.pose = odom_msg.pose.pose;
    traj_msg.poses.push_back(pose);
    trajectory_pub.publish(traj_msg);
    
}

void VIEkfSlam::drawRosMarker(int id, const ros::Time& time,
        const Eigen::Vector3d& position, const Eigen::Quaterniond& orientation,
        const Eigen::Matrix3d& cov_pp, 
        visualization_msgs::Marker& marker){

    marker.header.frame_id = world_frame_id;
    marker.header.stamp = time;
    marker.ns = "ArucoMarker";
    marker.id = id;
    marker.type = visualization_msgs::Marker::SPHERE;
    marker.action = visualization_msgs::Marker::ADD;
    marker.color.r = 0.0f;
    marker.color.g = 1.0f;
    marker.color.b = 0.0f;
    marker.color.a = 1.0f;
    marker.lifetime = ros::Duration();

    tf::pointEigenToMsg(position,marker.pose.position);
    tf::quaternionEigenToMsg(orientation,marker.pose.orientation);
    SelfAdjointEigenSolver<Matrix<double,3,3>> eigen_solver(
            (cov_pp + cov_pp.transpose())/2);
    VectorXd eigen_values = eigen_solver.eigenvalues();
    marker.scale.x = eigen_values(0) * 10;
    marker.scale.y = eigen_values(1) * 10;
    marker.scale.z = eigen_values(2) * 10;
}


void VIEkfSlam::undistortPoints(const std::vector<cv::Point2f>& pts_in,
                        const cv::Vec4d& intrinsics,
                        const cv::Vec4d& distortion_coeffs,
                        std::vector<cv::Point2f>& pts_out){
                        
    if(pts_in.size() == 0) return;

    const cv::Matx33d K(
        intrinsics[0], 0.0, intrinsics[2],
        0.0, intrinsics[1], intrinsics[3],
        0.0, 0.0, 1.0);
    
    cv::undistortPoints(pts_in,pts_out,K,distortion_coeffs);

}

}

