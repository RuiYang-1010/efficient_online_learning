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
// Rui
#include <vision_msgs/Detection2DArray.h>
#include <vision_msgs/ObjectHypothesisWithPose.h>

int main(int argc, char **argv) {
        double frequency;
        std::string camera_dir;
        double timestamp;

        ros::init(argc, argv, "kitti_camera_ros");
        ros::NodeHandle private_nh("~");

        ros::Publisher camera_pub = private_nh.advertise<vision_msgs::Detection2DArray>("/image_detections", 100, true);

        private_nh.param<double>("frequency", frequency, 10);
        private_nh.param<std::string>("camera_dir", camera_dir, "camera_dir_path");

        ros::Rate loop_rate(frequency);

        //vision_msgs::Detection2DArray detection_results;

        struct dirent **filelist;
        int n_file = scandir(camera_dir.c_str(), &filelist, NULL, alphasort);
        if(n_file == -1) {
                ROS_ERROR_STREAM("[kitti_camera_ros] Could not open directory: " << camera_dir);
                return EXIT_FAILURE;
        } else {
                ROS_INFO_STREAM("[kitti_camera_ros] Load camera files in " << camera_dir);
                ROS_INFO_STREAM("[kitti_camera_ros] frequency (loop rate): " << frequency);
        }

        int i_file = 2; // 0 = . 1 = ..
        while(ros::ok() && i_file < n_file) {
                vision_msgs::Detection2DArray detection_results;

                /*** Camera ***/
                std::string s = camera_dir + filelist[i_file]->d_name;
                std::fstream camera_txt(s.c_str(), std::ios::in | std::ios::binary);
                //std::cerr << "s: " << s.c_str() << std::endl;
                if(!camera_txt.good()) {
                        ROS_ERROR_STREAM("[kitti_camera_ros] Could not read file: " << s);
                        return EXIT_FAILURE;
                } else {
                        camera_txt >> timestamp;
                        ros::Time timestamp_ros(timestamp == 0 ? ros::TIME_MIN.toSec() : timestamp);
                        detection_results.header.stamp = timestamp_ros;

                        //camera_txt.seekg(0, std::ios::beg);

                        for(int i = 0; camera_txt.good() && !camera_txt.eof(); i++) {
                                vision_msgs::Detection2D detection;
                                vision_msgs::ObjectHypothesisWithPose result;
                                camera_txt >> detection.bbox.center.x;
                                camera_txt >> detection.bbox.center.y;
                                camera_txt >> detection.bbox.size_x;
                                camera_txt >> detection.bbox.size_y;
                                camera_txt >> result.id;
                                camera_txt >> result.score;
                                detection.results.push_back(result);
                                detection_results.detections.push_back(detection);
                        }
                        camera_txt.close();

                        camera_pub.publish(detection_results);
                        // ROS_INFO_STREAM("[kitti_camera_ros] detection_results.size " << detection_results.detections.size());
                        // ROS_INFO_STREAM("--------------------------------------------");
                        // for(int n = 0; n < detection_results.detections.size(); n++) {
                        //   ROS_INFO_STREAM("[kitti_camera_ros] detections.label " << detection_results.detections[n].results[0].id);
                        //   ROS_INFO_STREAM("[kitti_camera_ros] detections.score " << detection_results.detections[n].results[0].score);
                        // }
                }

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
