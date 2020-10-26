cmake_minimum_required(VERSION 2.8.3)
project(online_svm_ros)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  )

find_package(yaml-cpp REQUIRED)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  )

catkin_package(
  INCLUDE_DIRS include
  )

add_executable(online_svm_ros
  src/online_svm_ros.cpp
  )

target_link_libraries(online_svm_ros
  ${catkin_LIBRARIES}
  yaml-cpp
  svm
  )

if(catkin_EXPORTED_TARGETS)
  add_dependencies(online_svm_ros
    ${catkin_EXPORTED_TARGETS}
    )
endif()
