# Efficient Online Transfer Learning for 3D Object Classification in Autonomous Driving #

This is a ROS-based efficient online learning framework for object classification in 3D LiDAR scans, taking advantage of robust multi-target tracking to avoid the need for data annotation by a human expert.
Please watch the videos below for more details.

[![YouTube Video 1](https://img.youtube.com/vi/wl5ehOFV5Ac/0.jpg)](https://www.youtube.com/watch?v=wl5ehOFV5Ac)

# Install & Build
Please read the readme file of each sub-package first and install the corresponding dependencies.

# Run
#### 1. Prepare dataset
     # Raw Data from KITTI Benchmark
     # Image Pre-detection Data by running efficient_det_node.py (optional)

#### 2. Manual set specific path parameters
     # launch/efficient_online_learning
     # autoware_tracker/config/params.yaml

#### 3. Run the project
```bash
$ cd catkin_ws
$ source devel/setup.bash
$ roslaunch src/efficient_online_learning/launch/efficient_online_learning.launch
```
