// (c) 2020 Zhi Yan, Rui Yang
// This code is licensed under MIT license (see LICENSE.txt for details)
#define GMM_USES_BLAS

// ROS
#include <ros/ros.h>
#include <std_msgs/String.h>
// Online Random Forests
#include "online_forests/onlinetree.h"
#include "online_forests/onlinerf.h"

int main(int argc, char **argv) {
  std::ofstream icra_log;
  std::string log_name = "orf_time_log_"+std::to_string(ros::WallTime::now().toSec());

  std::string conf_file_name;
  std::string model_file_name;
  int mode; // 1 - train, 2 - test, 3 - train and test.
  int minimum_samples;
  int total_samples = 0;

  ros::init(argc, argv, "online_forests_ros");
  ros::NodeHandle nh, private_nh("~");

  if(private_nh.getParam("conf_file_name", conf_file_name)) {
    ROS_INFO("Got param 'conf_file_name': %s", conf_file_name.c_str());
  } else {
    ROS_ERROR("Failed to get param 'conf_file_name'");
    exit(EXIT_SUCCESS);
  }

  if(private_nh.getParam("model_file_name", model_file_name)) {
    ROS_INFO("Got param 'model_file_name': %s", model_file_name.c_str());
  } else {
    ROS_ERROR("Failed to get param 'model_file_name'");
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

    if(atoi(features->data.substr(0, features->data.find(" ")).c_str()) >= minimum_samples) {
      OnlineRF model(hp, dataset_tr.m_numClasses, dataset_tr.m_numFeatures, dataset_tr.m_minFeatRange, dataset_tr.m_maxFeatRange); // TOTEST: OnlineTree

      //string model_file_name = "";
      icra_log.open(log_name, std::ofstream::out | std::ofstream::app);
      time_t start_time = ros::WallTime::now().toSec();

      switch(mode) {
      case 1: // train only
        if(access( model_file_name.c_str(), F_OK ) != -1){
          model.loadForest(model_file_name);
        }
        model.train(dataset_tr);
        //model.writeForest(model_file_name);
        break;
      case 2: // test only
        model.loadForest(model_file_name);
        model.test(dataset_ts);
        break;
      case 3: // train and test
        model.trainAndTest(dataset_tr, dataset_ts);
        break;
      default:
        ROS_ERROR("Unknown 'mode'");
      }

      std::cout << "[online_forests_ros] Training time: " << ros::WallTime::now().toSec() - start_time << " s" << std::endl;
      icra_log << (total_samples+=dataset_tr.m_numSamples) << " " << ros::WallTime::now().toSec()-start_time << "\n";
      icra_log.close();
    }

    ros::spinOnce();
  }

  return EXIT_SUCCESS;
}
