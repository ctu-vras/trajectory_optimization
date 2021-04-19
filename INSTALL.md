# Installation

## Locally (tested with Ubuntu 18.04)

Clone the package and dependencies to ROS workspace:
```bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
git clone https://github.com/RuslanAgishev/trajectory_optimization.git
git clone -b melodic-devel https://github.com/ros/geometry2
git clone -b melodic-devel https://github.com/ros/geometry
git clone -b melodic https://github.com/ros-perception/vision_opencv
```
Setup python3 environment (called `dl` in this example) for the package with
[conda](https://docs.conda.io/en/latest/miniconda.html):
```
conda create -n dl python=3.6
conda activate dl
pip install catkin_pkg empy rospkg
```
Follow [pytorch3d](https://github.com/facebookresearch/pytorch3d/blob/master/INSTALL.md)
installation instructions.

Build the ROS package:
```
cd ~/catkin_ws
rosdep install --from-paths src --ignore-src -y -r

catkin config -DPYTHON_EXECUTABLE=$HOME/miniconda3/envs/dl/bin/python3  \
              -DPYTHON_INCLUDE_DIR=$HOME/miniconda3/envs/dl/include/python3.6m \
              -DPYTHON_LIBRARY=$HOME/miniconda3/envs/dl/lib/libpython3.6m.so
catkin build trajectory_optimization

source ~/catkin_ws/devel/setup.bash
```

## [Singularity](https://singularity.lbl.gov/)

Please, download the prebuilt writable singularity image from
[https://drive.google.com/drive/folders/1oOvIDpldfv30HqijHu1xupgO8EdTED6K?usp=sharing](https://drive.google.com/drive/folders/1oOvIDpldfv30HqijHu1xupgO8EdTED6K?usp=sharing)

Run the image mounting your catkin workspace with the package:
```bash
sudo singularity shell -w --nv --bind $HOME/trajopt_ws/:/root/catkin_ws trajopt/
```

From the singularity container execute:
```bash
source /root/.bashrc
roslaunch trajectory_optimization trajecory_optimization.launch
```
