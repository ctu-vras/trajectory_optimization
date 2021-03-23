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