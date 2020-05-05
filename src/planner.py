#!/usr/bin/env python

import rospy
from sensor_msgs.msg import PointCloud2
import tf
from nav_msgs.msg import Path
from utils import publish_path

import numpy as np
from pointcloud_utils import pointcloud2_to_xyzrgb_array
from planning import breadth_first_search
from planning import prune_path, smooth_path
from planning import apf_planner, apf_path_to_map
from grid import create_grid


class Planner:
    def __init__(self, elev_map_topic='/exploration/local_elev', pc_topic='/dynamic_point_cloud'):
        self.elev_map_topic = rospy.get_param('~pointcloud_topic', elev_map_topic)
        print("Subscribed to " + self.elev_map_topic)
        local_map_sub = rospy.Subscriber(elev_map_topic, PointCloud2, self.local_map_callback)

        self.pc_topic = rospy.get_param('~pointcloud_topic', pc_topic)
        print("Subscribed to " + self.pc_topic)
        pc_sub = rospy.Subscriber(pc_topic, PointCloud2, self.pc_callback)

        self.tl = tf.TransformListener()
        self.goal = None
        self.local_map = None

    def get_robot_pose(self, origin_frame='map', robot_frame='base_link'):
        position, quaternion = None, None
        if self.tl.frameExists(origin_frame) and self.tl.frameExists(robot_frame):
            t = self.tl.getLatestCommonTime(origin_frame, robot_frame)
            position, quaternion = self.tl.lookupTransform(origin_frame, robot_frame, t)
        return position, quaternion

    @property
    def robot_pose(self):
        return self.get_robot_pose()[0]

    @property
    def robot_orient(self):
        return self.get_robot_pose()[1]
    
    def local_map_callback(self, elev_map_pc_msg):
        elev_map = pointcloud2_to_xyzrgb_array(elev_map_pc_msg)
        self.local_map = elev_map

    def pc_callback(self, pc_msg):
        self.dynamic_pc = pointcloud2_to_xyzrgb_array(pc_msg)


# define main parameters here
height_margin = 0.1 # traversable height margin: elevation map cells, higher than this value, are considered as untraversable
safety_distance = 0.3 # to keep away from obstacles (for grid creation)
map_res = 0.15 # map resolution
unexplored_value = 0.25 # value of unknown cells in a constructed grid
num_apf_iters = 50 # max num of iterations to plan an APF trajectory


if __name__ == '__main__':
    rospy.init_node('path_planner')
    planner = Planner()
    
    rate = rospy.Rate(0.33)
    while not rospy.is_shutdown():

        if planner.local_map is not None and planner.robot_pose is not None:
            elev_map = planner.local_map; robot_pose = np.array(planner.robot_pose)
            grid, elev_grid = create_grid(elev_map, map_res, safety_distance, height_margin, unexplored_value)
            x_min, y_min = np.min(elev_map[:, 0]), np.min(elev_map[:, 1])

            # define start on a grid
            robot_grid_pose = (robot_pose - [x_min, y_min, 0]) // map_res
            start_grid = (int(robot_grid_pose[0]), int(robot_grid_pose[1]))

            # BFS
            path_grid, goal_grid = breadth_first_search(grid, start_grid, unexplored_value)
            # transform path to map coordintes (m)
            bfs_path = [(np.array(point)*map_res+[x_min, y_min]).tolist()+[elev_grid[point]] for point in path_grid]

            if len(bfs_path)>0:
                bfs_path = prune_path(bfs_path, 1e-3)
                # bfs_path = smooth_path(np.array(bfs_path), vis=1)
            bfs_path = np.array(bfs_path)

            # APF
            # if BFS found a frontier then do APF trajectory planning
            if goal_grid is not None:
                apf_path_grid = apf_planner(grid, [start_grid[1], start_grid[0]], [goal_grid[1], goal_grid[0]], num_iters=num_apf_iters)
                # transform path to map coordintes (m)
                apf_path = apf_path_to_map(apf_path_grid, elev_map, elev_grid, map_res)
                apf_path = np.array(apf_path)

                apf_path = apf_path - apf_path[0,:] + robot_pose # start path exactly from robot location
                # publish paths here
                apf_path[:,2] += map_res # for better path visuaization with elevation map
                publish_path(apf_path, orient=[0,0,0,1], topic_name='/exploration/apf_path')
                bfs_path[:,2] += map_res # for better path visuaization with elevation map
                publish_path(bfs_path, orient=[0,0,0,1], topic_name='/exploration/bfs_path')
            
        rate.sleep()
    