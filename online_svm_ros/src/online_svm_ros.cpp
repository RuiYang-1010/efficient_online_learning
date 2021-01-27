// (c) 2020 Zhi Yan
// This code is licensed under GPLv3 license (see LICENSE for details)

#include <ros/ros.h>
#include <std_msgs/String.h>
#include <yaml-cpp/yaml.h>
#include <libsvm/svm.h>
#include <fstream>

#define __APP_NAME__ "online_svm_ros"

double s2d(std::string s) {
  std::stringstream ss;
  double d;
  ss << s;
  ss >> d;
  return d;
}

int main(int argc, char **argv) {
  std::ofstream icra_log;
  std::string log_name = "svm_time_log_"+std::to_string(ros::WallTime::now().toSec());
  
  std::string svm_train_params;
  int max_examples = 5000;
  struct svm_parameter parameter;
  struct svm_problem problem;
  struct svm_model *model = NULL;
  //struct svm_node *node = NULL;
  float *range[2];
  float x_lower = -1.0, x_upper = 1.0;
  bool best_params = false;
  bool save_data = false;
  
  ros::init(argc, argv, "online_svm_ros");
  ros::NodeHandle nh, private_nh("~");
  
  if(private_nh.getParam("svm_train_params", svm_train_params)) {
    ROS_INFO("[%s] Got param 'svm_train_params': %s", __APP_NAME__, svm_train_params.c_str());
    YAML::Node node = YAML::LoadFile(svm_train_params);
    
    x_lower = node["x_lower"].as<float>();
    x_upper = node["x_upper"].as<float>();
    max_examples = node["max_examples"].as<int>();
    
    parameter.svm_type = node["svm_type"].as<int>();
    parameter.kernel_type = node["kernel_type"].as<int>();
    parameter.degree = node["degree"].as<int>();
    parameter.gamma = node["gamma"].as<double>();
    parameter.coef0 = node["coef0"].as<double>();
    parameter.cache_size = node["cache_size"].as<double>();
    parameter.eps = node["eps"].as<double>();
    parameter.C = node["C"].as<double>();
    parameter.nr_weight = node["nr_weight"].as<int>();
    parameter.weight_label = NULL;
    parameter.weight = NULL;
    parameter.nu = node["nu"].as<double>();
    parameter.p = node["p"].as<double>();
    parameter.shrinking = node["shrinking"].as<int>();
    parameter.probability = node["probability"].as<int>();

    best_params = node["best_params"].as<bool>();
    save_data = node["save_data"].as<bool>();
  } else {
    ROS_ERROR("[%s] Failed to get param 'svm_train_params'", __APP_NAME__);
    exit(EXIT_SUCCESS);
  }
  
  std_msgs::String::ConstPtr features;
  int m_numFeatures = 0;
  std::string line;
  
  problem.l = 0;
  problem.y = (double *)malloc(max_examples * sizeof(double));
  problem.x = (struct svm_node **)malloc(max_examples * sizeof(struct svm_node *));
  
  while(ros::ok()) {
    features = ros::topic::waitForMessage<std_msgs::String>("/point_cloud_features/features"); // process blocked waiting
    std::istringstream iss(features->data);
    
    // Parsing the header
    std::getline(iss, line, ' ');
    int m_numSamples = atoi(line.c_str());
    std::getline(iss, line, ' ');
    if(m_numFeatures == 0 && atoi(line.c_str()) > 0) {
      m_numFeatures = atoi(line.c_str());
      for(int i = 0; i < max_examples; i++) {
	problem.x[i] = (struct svm_node *)malloc((m_numFeatures + 1) * sizeof(struct svm_node));
      }
      range[0] = (float *)malloc(m_numFeatures * sizeof(float));
      range[1] = (float *)malloc(m_numFeatures * sizeof(float)); 
      //node = (struct svm_node *)malloc((m_numFeatures + 1) * sizeof(struct svm_node)); // 1 more size for end index (-1) 
    }
    std::getline(iss, line, ' ');
    //m_numClasses = atoi(line.c_str());
    std::getline(iss, line, '\n');
    //startIndex = atoi(line.c_str());

    while(std::getline(iss, line)) {
      problem.y[problem.l] = atoi(line.substr(line.find(' ')).c_str()); // read label
      std::istringstream iss2(line);
      for(int i = 0; i < m_numFeatures; i++) {
    	std::getline(iss2, line, ':');
    	problem.x[problem.l][i].index = atoi(line.c_str());
	std::getline(iss2, line, ' ');
    	problem.x[problem.l][i].value = s2d(line);
      }
      problem.x[problem.l][m_numFeatures].index = -1;
      problem.l++;
      if(problem.l == max_examples) {
	ROS_INFO("[online_svm_ros] Reach the maximum number of examples %d.", max_examples);
	break;
      }
    }
    
    // train
    icra_log.open(log_name, std::ofstream::out | std::ofstream::app);
    time_t start_time = ros::WallTime::now().toSec();
    
    for(int i = 0; i < problem.l; i++) {
      for(int j = 0; j < m_numFeatures; j++) {
	if(std::fabs(range[0][j] - range[1][j]) < DBL_EPSILON) { // skip single-valued attribute
    	  continue;
    	}
	
    	if(std::fabs(problem.x[i][j].value - x_lower) < DBL_EPSILON) {
    	  problem.x[i][j].value = range[0][j];
    	} else if(std::fabs(problem.x[i][j].value - x_upper) < DBL_EPSILON) {
    	  problem.x[i][j].value = range[1][j];
    	} else {
    	  problem.x[i][j].value = range[0][j] + (problem.x[i][j].value - x_lower) * (range[1][j] - range[0][j]) / (x_upper - x_lower);
    	}
      }
    }
    
    // save data to file
    if(save_data) {
      std::ofstream ofs;
      ofs.open("svm_training_data_"+std::to_string(start_time));
      for(int i = 0; i < problem.l; i++) {
	ofs << problem.y[i];
	for(int j = 0; j < m_numFeatures; j++)
	  ofs << " " << problem.x[i][j].index << ":" <<  problem.x[i][j].value;
	ofs << "\n";
      }
      ofs.close();
    }
    
    // scale the current data
    for(int i = 0; i < m_numFeatures; i++) {
      range[0][i] = FLT_MAX; // min range
      range[1][i] = -FLT_MAX; // max range
    }
    for(int i = 0; i < problem.l; i++) {
      for(int j = 0; j < m_numFeatures; j++) {
    	range[0][j] = std::min(range[0][j], (float)problem.x[i][j].value);
    	range[1][j] = std::max(range[1][j], (float)problem.x[i][j].value);
      }
    }
    // for(int i = 0; i < m_numFeatures; i++) {
    //   std::cout << "svm scale range [attribute " << i << "]: " << range[0][i] << ", " << range[1][i] << std::endl;
    // }
    for(int i = 0; i < problem.l; i++) {
      for(int j = 0; j < m_numFeatures; j++) {
	if(std::fabs(range[0][j] - range[1][j]) < DBL_EPSILON) { // skip single-valued attribute
    	  continue;
	}
    	if(std::fabs(problem.x[i][j].value - range[0][j]) < DBL_EPSILON) {
    	  problem.x[i][j].value = x_lower;
    	} else if(std::fabs(problem.x[i][j].value - range[1][j]) < DBL_EPSILON) {
    	  problem.x[i][j].value = x_upper;
    	} else {
    	  problem.x[i][j].value = x_lower + (x_upper - x_lower) * (problem.x[i][j].value - range[0][j]) / (range[1][j] - range[0][j]);
	}
    	//std::cout << "training data " << i << " [attribute " << j << "]: " << problem.x[i][j].value << std::endl;
      }
    }
    
    if(best_params) {
      std::ofstream ofs;
      ofs.open("svm_training_data_scaled");
      for(int i = 0; i < problem.l; i++) {
      	ofs << problem.y[i];
      	for(int j = 0; j < m_numFeatures; j++)
      	  ofs << " " << problem.x[i][j].index << ":" <<  problem.x[i][j].value;
      	ofs << "\n";
      }
      ofs.close();
      
      std::cout << "Finding the best training parameters ..." << std::endl;
      if(svm_check_parameter(&problem, &parameter) == NULL) {
    	char result[100];
    	FILE *fp = popen("svm-grid svm_training_data_scaled", "r");
    	if(fp == NULL) {
    	  std::cerr << "Can not run cross validation!" << std::endl;
    	} else {
    	  if(fgets(result, 100, fp) != NULL) {
    	    char *pch = strtok(result, " ");
    	    parameter.C = s2d(pch);
	    pch = strtok(NULL, " ");
    	    parameter.gamma = s2d(pch);
	    pch = strtok(NULL, " ");
    	    float rate = s2d(pch);
    	    std::cout << "Best c=" << parameter.C << ", g=" << parameter.gamma << " CV rate=" << rate << std::endl;
    	  }
    	}
    	pclose(fp);
      }
    }
    model = svm_train(&problem, &parameter);
     
    if(save_data) {
      if(svm_save_model("svm_model", model) == 0) {
	ROS_INFO("[online_svm_ros] A model has been generated here: ~/.ros/svm.model");
      }
    }
    
    std::cout << "[online_svm_ros] Training time: " << ros::WallTime::now().toSec() - start_time << " s" << std::endl;
    icra_log << problem.l << " " << ros::WallTime::now().toSec()-start_time << "\n";
    icra_log.close();
    
    ros::spinOnce();
    
    if(problem.l == max_examples) {
      ROS_INFO("[online_svm_ros] Exit the node.");
      break;
    }
  }

  // memory tidy-up
  svm_destroy_param(&parameter);
  free(problem.y);
  if(m_numFeatures > 0) {
    for(int i = 0; i < max_examples; i++) {
      free(problem.x[i]);
    }
  }
  free(problem.x);
  if(model != NULL) {
    svm_free_and_destroy_model(&model);
  }
  free(range[0]);
  free(range[1]);
  // if(node != NULL) {
  //   free(node);
  // }
  
  return EXIT_SUCCESS;
}
