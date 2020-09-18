cmake_minimum_required(VERSION 2.8.3)
project(point_cloud_features)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  pcl_conversions
  pcl_ros
  std_msgs
  )

find_package(PCL REQUIRED)

include_directories(include ${catkin_INCLUDE_DIRS})

catkin_package(INCLUDE_DIRS include)

add_executable(point_cloud_features
  src/point_cloud_feature_extractor.cpp
  src/${PROJECT_NAME}/point_cloud_features.cpp
  )

target_link_libraries(point_cloud_features
  ${catkin_LIBRARIES}
  )

if(catkin_EXPORTED_TARGETS)
  add_dependencies(point_cloud_features
    ${catkin_EXPORTED_TARGETS}
    )
endif()
