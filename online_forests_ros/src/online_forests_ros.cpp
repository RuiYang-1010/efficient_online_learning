// (c) 2020 Zhi Yan
// This code is licensed under MIT license (see LICENSE.txt for details)
#define GMM_USES_BLAS

// ROS
#include <ros/ros.h>
#include <std_msgs/String.h>
// Online Random Forests
#include "online_forests/onlinetree.h"
#include "online_forests/onlinerf.h"

int main(int argc, char **argv) {
  std::string conf_file_name;
  int mode; // 1 - train, 2 - test, 3 - train and test.
  int minimum_samples;
  
  ros::init(argc, argv, "online_forests_ros");
  ros::NodeHandle nh, private_nh("~");
  
  if(private_nh.getParam("conf_file_name", conf_file_name)) {
    ROS_INFO("Got param 'conf_file_name': %s", conf_file_name.c_str());
  } else {
    ROS_ERROR("Failed to get param 'conf_file_name'");
    exit(EXIT_SUCCESS);
  }
  
  if(private_nh.getParam("mode", mode)) {
    ROS_INFO("Got param 'mode': %d", mode);
  } else {
    ROS_ERROR("Failed to get param 'mode'");
    exit(EXIT_SUCCESS);
  }

  private_nh.param<int>("minimum_samples", minimum_samples, 1);
  
  Hyperparameters hp(conf_file_name);
  std_msgs::String::ConstPtr features;
  
  while (ros::ok()) {
    features = ros::topic::waitForMessage<std_msgs::String>("/point_cloud_features/features"); // process blocked waiting

    // Creating the train data
    DataSet dataset_tr;
    dataset_tr.loadLIBSVM2(features->data);
    
    // Creating the test data
    DataSet dataset_ts;
    // dataset_ts.loadLIBSVM(hp.testData);

    if(atoi(features->data.substr(0, features->data.find(" ")).c_str()) > minimum_samples) {
      OnlineRF model(hp, dataset_tr.m_numClasses, dataset_tr.m_numFeatures, dataset_tr.m_minFeatRange, dataset_tr.m_maxFeatRange); // TOTEST: OnlineTree
      
      time_t start_time = time(NULL);
      switch(mode) {
      case 1: // train only
	model.train(dataset_tr);
	break;
      case 2: // test only
	model.test(dataset_ts);
	break;
      case 3: // train and test
	model.trainAndTest(dataset_tr, dataset_ts);
	break;
      default:
	ROS_ERROR("Unknown 'mode'");
      }
      cout << "Time: " << time(NULL)-start_time << "s" << endl;
    }
    
    ros::spinOnce();
  }
  
  return EXIT_SUCCESS;
}
