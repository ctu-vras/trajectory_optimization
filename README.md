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

Ones the package is installed, run the launch file and specify the bag file location:
```bash
roslaunch frontier_exploration frontier_exploration.launch bag_filename:=<PATH_TO_BAG_FILE>
```
Replace `<PATH_TO_BAG_FILE>` with the full path to the bag file.


## Assignment

The task is to implement a planning algorithm that chooses the frontier
and plans the path between the robot's actual position and the chosen frontier.
The closest frontier should be chosen as the one to which leads the closest path,
not the closest in the euclidean metrics. The plan should be as smooth as possible,
the path should not suffer by map discretization (changes of yaw angle should be smooth).

## Implementation details

In order to meet the assignment requirements, I decided to do the following steps:

1. Convert the elevation map to 2.5D occupancy grid representation.

   The grid cell could have one of the free values:
      - `0` - means the cell is free (traversable)
      - `1` - the cell is occupied (obstacle, untraversable)
      - `0.5` - the cell is not explored yet. This value could be adjusted for the step 3 of the algorithm.
      
   Traversability requirement: `cell_z - robot_z > height_margin` defines if the cell is free (`0`) or
   contains an obstacle (`1`).
   The `height_margin` parameter could be ajusted. It defines the maximum height of the local elevation map
   which is considered as traversable for the robot. However, a more sofisticated way to define it,
   could be calculation of the mean height of the neighbouing cells.
   Implementation details are available at
   [grid.py](https://github.com/RuslanAgishev/frontier_exploration/blob/path_planning/src/grid.py).
   
2. Plan a path on the generated occupancy grid with Breadth First Search algorithm (BFS)
   to find the closest frontier from robot location.
   
   After local occupancy map is defined and built, I convert robot location to grid coordinates,
   and run BFS planning until a first unexplored cell is found. This would be the closest frontier to
   navigate to. The BFS algorithm provides also a path to the frontier. However,
   the trajectory is almost always jerky and lies too close to obstacles.
   That is why I decided to perform the next step. The implementation details coud be found at
   [planning.py, line 127](https://github.com/RuslanAgishev/frontier_exploration/blob/849a6671dff3f8be2594badab1409c5915002e59/src/planning.py#L127).
   
3. Ones the rough plan (from BFS) and a goal (nearest frontier to explore) is found,
   run Artificial Potential Field (APF) algorithm to generate smooth collision free trajectory.
   
   The main goal of this planning layer is to construct a feasible trajectory along the traverable area
   to the nearest frontier to explore. Several parameters could be tweaked here to acieve good performance.
   The main of them are:
      - `repulsive_coef`: defines obstacles repulsive force,
      - `attractive_coef`: is responsible for the frontier attractive force for the robot,
      - `unexplored_value`: parameter relevant to occupancy grid creation. It also defines with
        which force the unexplored area affects the robot. It could even attract the robot if the
        parameter is set with negative value.
      - `max_apf_iters`: defines the maximum APF trajectory length.
   Implementation details are available at
   [planning.py, line 250](https://github.com/RuslanAgishev/frontier_exploration/blob/849a6671dff3f8be2594badab1409c5915002e59/src/planning.py#L250)


## Appendix

Experiments with the data could be found in the
[planning.ipynb](https://github.com/RuslanAgishev/frontier_exploration/blob/path_planning/src/planning.ipynb)
jupyter-notebook. Among them is, for example, and A* algorithm approach for the task.
