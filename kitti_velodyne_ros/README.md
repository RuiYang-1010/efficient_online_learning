# kitti_velodyne_ros

[![Build Status](https://travis-ci.org/epan-utbm/kitti_velodyne_ros.svg?branch=melodic)](https://travis-ci.org/epan-utbm/kitti_velodyne_ros) [![Codacy Badge](https://api.codacy.com/project/badge/Grade/24e89caa1d40456f966e039145f64edf)](https://app.codacy.com/gh/epan-utbm/kitti_velodyne_ros?utm_source=github.com&utm_medium=referral&utm_content=epan-utbm/kitti_velodyne_ros&utm_campaign=Badge_Grade_Dashboard) [![License](https://img.shields.io/badge/License-BSD%203--Clause-gree.svg)](https://opensource.org/licenses/BSD-3-Clause)

Load KITTI velodyne data, play in ROS.

## Usage

```console
$ roslaunch kitti_velodyne_ros kitti_velodyne_ros.launch
```

If you want to save the point cloud as a csv file, simply activate in [kitti_velodyne_ros.launch](launch/kitti_velodyne_ros.launch) :

```console
<param name="save_to_csv" type="bool" value="true"/>
```

In case you want to play with [LOAM](https://github.com/laboshinl/loam_velodyne):

```console
$ roslaunch kitti_velodyne_ros kitti_velodyne_ros_loam.launch
```
