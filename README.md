# Frontier Exploration

## Installation

Clone the package and dependencies to ROS workspace:
```bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
git clone https://github.com/RuslanAgishev/frontier_exploration.git
git clone https://github.com/ros/geometry2
git clone https://github.com/ros/geometry
git clone https://github.com/ros-perception/vision_opencv
```
Setup python3 environment (called `dl` in this example) for the package with
[conda](https://docs.conda.io/en/latest/miniconda.html):
```
conda create -n dl
conda activate dl
cd ~/catkin_ws/src/frontier_exploration/
pip install -r requirements.txt
```
Follow [pytorch3d](https://github.com/facebookresearch/pytorch3d/blob/master/INSTALL.md)
installation instructions.

Build the ROS package (tested with Ubuntu 18.04):
```
cd ~/catkin_ws
catkin config -DPYTHON_EXECUTABLE=$HOME/miniconda3/envs/dl/bin/python3  \
              -DPYTHON_INCLUDE_DIR=$HOME/miniconda3/envs/dl/include/python3.6m \
              -DPYTHON_LIBRARY=$HOME/miniconda3/envs/dl/lib/libpython3.6m.so
catkin build frontier_exploration
source ~/catkin_ws/devel/setup.bash
```

## Running

### Point cloud Visibility Estimation

<img src="./data/hpr.gif">

Ones the package is installed, run the launch file and specify the bag file location:
```bash
roslaunch frontier_exploration pointcloud_processor.launch
rosbag play PATH_TO_BAG_FILE -r 5 --clock
```
Replace `PATH_TO_BAG_FILE` with the path to the bag file, for example: `./data/josef_2019-06-06-13-58-12_proc_0.1m.bag`

In this example, the purple points are the whole cloud in a camera frame,
grey ones are only the visible points (not occluded by other points from the camera perspective).
The hidden points removal (HPR) algorithm implementation is based on the article
[Katz et al](http://www.weizmann.ac.il/math/ronen/sites/math.ronen/files/uploads/katz_tal_basri_-_direct_visibility_of_point_sets.pdf
).
The resultant point cloud rendering on an image plane is done with
[pytorch3d](https://github.com/facebookresearch/pytorch3d) library.

### Multiple cameras:

<img src="./data/hpr_cams_01234.gif">

In this example, the point cloud visibility is estimated for each individual camera
(in its field of view and the distance range) separately.
The combined point cloud is then visualized in the robot `base_link` frame.

### Examples

The [./notebooks](https://github.com/RuslanAgishev/frontier_exploration/tree/master/notebooks)
folder contains the following examples:
- [HPR](https://github.com/RuslanAgishev/frontier_exploration/blob/master/notebooks/hidden_points_removal.ipynb):
    hidden points removal example with different input point clouds as well as from different camera poses.
- [Camera position Optimization](https://github.com/RuslanAgishev/frontier_exploration/blob/master/notebooks/camera_pose_optimization_3d.ipynb):
    camera position optimization loop based on the point cloud visibility estimation.
- [HPR, Open3D](https://github.com/RuslanAgishev/frontier_exploration/blob/master/notebooks/open3d.ipynb):
    hidden points removal with [Open3D](http://www.open3d.org/html/tutorial/Basic/pointcloud.html#Hidden-point-removal) library.
- [Point cloud Rendering on Image plane](https://github.com/RuslanAgishev/frontier_exploration/blob/master/notebooks/pytorch3d.ipynb):
    point cloud hidden points removal with Open3D and rendering on an image plane with [pytorch3d](https://github.com/facebookresearch/pytorch3d) library.
