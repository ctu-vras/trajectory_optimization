# frontier_exploration

## Installation

Clone the package to ROS workspace and build it:
```bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
git clone https://github.com/RuslanAgishev/frontier_exploration.git
cd ~/catkin_ws
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
The hidden points removal algorithm implementation is based on the article
[Katz et al](http://www.weizmann.ac.il/math/ronen/sites/math.ronen/files/uploads/katz_tal_basri_-_direct_visibility_of_point_sets.pdf
).
The resultant point cloud rendering on an image plane is done with
[pytorch3d](https://github.com/facebookresearch/pytorch3d) library.
