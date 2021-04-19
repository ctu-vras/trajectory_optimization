# Perception aware Trajectory Optimization

Perception aware trajectory optimization based on point cloud visibility estimation in a camera frustum.
The package is implemented as a ROS node with
[examples](https://github.com/RuslanAgishev/trajectory_optimization/tree/master/notebooks) in jupyter-notebooks.

## Installation

Please, follow installation instructions in
[INSTALL.md](https://github.com/RuslanAgishev/trajectory_optimization/blob/master/INSTALL.md)

## Running

### Point cloud Visibility Estimation

<img src="./demos/hpr.gif">

Ones the package is installed, run the launch file and specify the bag file location:
```bash
roslaunch trajectory_optimization pointcloud_processor.launch
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

<img src="./demos/hpr_cams_01234.gif">

In this example, the point cloud visibility is estimated for each individual camera
(in its field of view and the distance range) separately.
The combined point cloud is then visualized in the robot `base_link` frame.

### Camera Position Optimization

<img src="./demos/cam_pose_opt.gif">

Ego-pose optimization based on the observed in camera frustum point cloud visibility estimation.
In this example, the points color encodes a distance-based (to camera frame) reward.
The white points are currently observed ones by camera.

```bash
roslaunch trajectory_optimization pose_optimization.launch
```

### Camera Waypoints Optimization

<img src="./demos/cam_wps_opt.gif">

Camera pose (X, Y and Yaw) optimization is consequently applied here for each separate sampled way-point
of an initial trajectory.

### Camera Trajectory Evaluation

<img src="./demos/cam_traj_eval.gif">

A camera trajectory could be evaluated by a number of observed voxels (points in the cloud).
Single pose visibility estimation rewards are combined using log odds representation as it
is done in [OctoMap](https://www.researchgate.net/publication/235008236_OctoMap_A_Probabilistic_Flexible_and_Compact_3D_Map_Representation_for_Robotic_Systems) paper.

### Camera Trajectory Optimization

<img src="./demos/cam_traj_opt.gif">

Based on the evaluation result, the trajectory (consisting of several waypoints)
is optimized with the goal to increase overal visibility score.

```bash
roslaunch trajectory_optimization trajecory_optimization.launch
```

## Examples

The [./notebooks](https://github.com/RuslanAgishev/trajectory_optimization/tree/master/notebooks)
folder contains the following examples:
- [HPR](https://github.com/RuslanAgishev/trajectory_optimization/blob/master/notebooks/hidden_points_removal.ipynb):
    hidden points removal example with different input point clouds as well as from different camera poses.
- Camera position Optimization,
[in 2D](https://github.com/RuslanAgishev/trajectory_optimization/blob/master/notebooks/camera_pose_optimization_2d.ipynb),
[in 3D](https://github.com/RuslanAgishev/trajectory_optimization/blob/master/notebooks/camera_pose_optimization_3d.ipynb):
    camera position optimization loop based on the point cloud visibility estimation.
- [HPR, Open3D](https://github.com/RuslanAgishev/trajectory_optimization/blob/master/notebooks/open3d.ipynb):
    hidden points removal with [Open3D](http://www.open3d.org/html/tutorial/Basic/pointcloud.html#Hidden-point-removal) library.
- [Point cloud Rendering on Image plane](https://github.com/RuslanAgishev/trajectory_optimization/blob/master/notebooks/pytorch3d.ipynb):
    point cloud hidden points removal with Open3D and rendering on an image plane with [pytorch3d](https://github.com/facebookresearch/pytorch3d) library.
