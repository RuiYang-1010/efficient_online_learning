# Efficient Online Transfer Learning for 3D Object Classification in Autonomous Driving #

*We are actively updating this repository (especially removing hard code and adding comments) to make it easy to use. If you have any questions, please open an issue. Thanks!*

This is a ROS-based efficient online learning framework for object classification in 3D LiDAR scans, taking advantage of robust multi-target tracking to avoid the need for data annotation by a human expert.
The system is only tested in Ubuntu 18.04 and ROS Melodic (compilation fails on Ubuntu 20.04 and ROS Noetic).

Please watch the videos below for more details.

[![YouTube Video 1](https://img.youtube.com/vi/wl5ehOFV5Ac/0.jpg)](https://www.youtube.com/watch?v=wl5ehOFV5Ac)

## Install & Build
Please read the readme file of each sub-package first and install the corresponding dependencies.

## Run
### 1. Prepare dataset
* (Optional) Download the [raw data](http://www.cvlibs.net/datasets/kitti/raw_data.php)  from KITTI.

* (Optional) Download the [sample data](https://github.com/epan-utbm/efficient_online_learning/releases/download/sample_data/2011_09_26_drive_0005_sync.tar) for testing.

* (Optional) Prepare a customized dataset according to the format of the sample data.

### 2. Manual set specific path parameters
     # launch/efficient_online_learning
     # autoware_tracker/config/params.yaml

### 3. Run the project
```sh
cd catkin_ws
source devel/setup.bash
roslaunch src/efficient_online_learning/launch/efficient_online_learning.launch
```

## Citation

If you are considering using this code, please reference the following:

```
@inproceedings{efficient_online_learning,
   author = {Rui Yang and Zhi Yan and Tao Yang and Yassine Ruichek},
   title = {Efficient Online Transfer Learning for 3D Object Classification in Autonomous Driving},
   booktitle = {Proceedings of the 2021 IEEE International Intelligent Transportation Systems Conference (ITSC)},
   pages = {2950--2957},
   year = {2021}
}
```
