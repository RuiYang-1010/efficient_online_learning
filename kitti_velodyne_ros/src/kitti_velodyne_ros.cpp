// ROS
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/Point.h>
#include <visualization_msgs/MarkerArray.h>
// PCL
#include <pcl_conversions/pcl_conversions.h>
// C++
#include <dirent.h>

int main(int argc, char **argv) {
  double frequency;
  std::string velodyne_dir;
  std::string poses_file;
  double pose[12];
  geometry_msgs::PoseArray poses;
  visualization_msgs::MarkerArray markers;
  bool save_to_csv;
  std::string times_file;
  double timestamp;
  
  ros::init(argc, argv, "kitti_velodyne_ros");
  ros::NodeHandle private_nh("~");
  
  ros::Publisher velodyne_pub = private_nh.advertise<sensor_msgs::PointCloud2>("velodyne_points", 100, true);
  ros::Publisher poses_pub = private_nh.advertise<geometry_msgs::PoseArray>("poses", 100, true);
  ros::Publisher markers_pub = private_nh.advertise<visualization_msgs::MarkerArray>("markers", 100, true);
  
  private_nh.param<double>("frequency", frequency, 10);
  private_nh.param<std::string>("velodyne_dir", velodyne_dir, "velodyne_dir_path");
  private_nh.param<std::string>("poses_file", poses_file, "poses_file_path");
  private_nh.param<std::string>("times_file", times_file, "times_file_path");
  private_nh.param<bool>("save_to_csv", save_to_csv, false);
  
  ros::Rate loop_rate(frequency);
  
  struct dirent **filelist;
  int n_file = scandir(velodyne_dir.c_str(), &filelist, NULL, alphasort);
  if(n_file == -1) {
    ROS_ERROR_STREAM("[kitti_velodyne_ros] Could not open directory: " << velodyne_dir);
    return EXIT_FAILURE;
  } else {
    ROS_INFO_STREAM("[kitti_velodyne_ros] Load velodyne files in " << velodyne_dir);
    ROS_INFO_STREAM("[kitti_velodyne_ros] frequency (loop rate): " << frequency);
    ROS_INFO_STREAM("[kitti_velodyne_ros] save_to_csv: " << std::boolalpha << save_to_csv);
  }
  
  std::fstream poses_txt(poses_file.c_str(), std::ios::in);
  if(!poses_txt.good()) {
    ROS_WARN_STREAM("[kitti_velodyne_ros] Could not read poses file: " << poses_file);
  }
  
  std::fstream times_txt(times_file.c_str(), std::ios::in);
  if(!times_txt.good()) {
    ROS_WARN_STREAM("[kitti_velodyne_ros] Could not read times file: " << times_file);
  }
  
  int i_file = 2; // 0 = . 1 = ..
  while(ros::ok() && i_file < n_file) {
    /*** Timestamp ***/
    times_txt >> timestamp;
    ros::Time timestamp_ros(timestamp == 0 ? ros::TIME_MIN.toSec() : timestamp);
    
    /*** Velodyne ***/
    std::string s = velodyne_dir + filelist[i_file]->d_name;
    std::fstream velodyne_bin(s.c_str(), std::ios::in | std::ios::binary);
    //std::cerr << s.c_str() << std::endl;
    if(!velodyne_bin.good()) {
      ROS_ERROR_STREAM("[kitti_velodyne_ros] Could not read file: " << s);
      return EXIT_FAILURE;
    } else {
      pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
      
      velodyne_bin.seekg(0, std::ios::beg);
      for(int i = 0; velodyne_bin.good() && !velodyne_bin.eof(); i++) {
      	pcl::PointXYZI point;
      	velodyne_bin.read((char *) &point.x, 3 * sizeof(float));
      	velodyne_bin.read((char *) &point.intensity, sizeof(float));
      	cloud->push_back(point);
      }
      velodyne_bin.close();
      
      if(save_to_csv) {
	std::ofstream csv_file(s.substr(0, s.find_last_of('.')) + ".csv");
	if(csv_file.is_open()) {
	  csv_file << "seq, x, y, z, i\n";
	  for(size_t i = 0; i < cloud->size(); i++) {
	    csv_file << i << ", " << cloud->points[i].x << ", " << cloud->points[i].y << ", " << cloud->points[i].z << ", " <<  cloud->points[i].intensity << "\n";
	  }
	  csv_file.close();
	}
      }
      
      if(velodyne_pub.getNumSubscribers() > 0) {
    	sensor_msgs::PointCloud2 pc2;
    	pcl::toROSMsg(*cloud, pc2);
    	pc2.header.frame_id = "velodyne";
	if(!poses_txt.good()) {
	  pc2.header.stamp = ros::Time::now();
	} else {
	  pc2.header.stamp = timestamp_ros;
	}
    	velodyne_pub.publish(pc2);
      }
    }
    
    /*** Ground Truth Poses ***/
    poses_txt >> pose[0] >> pose[1] >> pose[2] >> pose[3] >> pose[4] >> pose[5] >> pose[6] >> pose[7] >> pose[8] >> pose[9] >> pose[10] >> pose[11];
    geometry_msgs::Pose p;
    p.position.x = pose[11];
    p.position.y = pose[3];
    p.position.z = 0;
    p.orientation.w = 1;
    poses.poses.push_back(p);
    if(!poses_txt.good()) {
      poses.header.stamp = ros::Time::now();
    } else {
      poses.header.stamp = timestamp_ros;
    }
    poses.header.frame_id = "base_link";
    poses_pub.publish(poses);
    
    /*** Rviz car's trajectory ***/
    visualization_msgs::Marker m;
    m.header.stamp = ros::Time::now();
    m.header.frame_id = "odom";
    m.type = visualization_msgs::Marker::LINE_STRIP;
    geometry_msgs::Point pp;
    for(int i = 0; i < poses.poses.size(); i++) {
      pp.x = poses.poses[i].position.x - poses.poses[poses.poses.size()-1].position.x;
      pp.y = poses.poses[i].position.y - poses.poses[poses.poses.size()-1].position.y;
      m.points.push_back(pp);
    }
    m.scale.x = 0.1;
    m.color.a = 1.0;
    m.color.r = 1.0;
    m.color.g = 0.5;
    m.color.b = 0.0;
    m.lifetime = ros::Duration(1.0);
    markers.markers.push_back(m);
    markers_pub.publish(markers);
    
    ros::spinOnce();
    loop_rate.sleep();
    i_file++;
  }
  
  for(int i = 2; i < n_file; i++) {
    free(filelist[i]);
  }
  free(filelist);
  
  return EXIT_SUCCESS;
}
