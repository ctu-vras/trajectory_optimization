#!/usr/bin/env python

import rospy
from sensor_msgs.msg import PointCloud2
import tf
from nav_msgs.msg import Path
from utils import publish_path

import numpy as np
from pointcloud_utils import pointcloud2_to_xyzrgb_array
from planning import breadth_first_search
from planning import prune_path
from planning import smooth_path
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


# main parameters
map_res = 0.15
margin = 0.2

if __name__ == '__main__':
    rospy.init_node('path_planner')
    planner = Planner()
    
    rate = rospy.Rate(0.33)
    while not rospy.is_shutdown():
        if planner.local_map is not None and planner.robot_pose is not None:
            elev_map = planner.local_map; robot_pose = np.array(planner.robot_pose)
            grid, elev_grid = create_grid(elev_map, map_res=0.15, safety_distance=0.2, margin=margin)
            x_min, y_min = np.min(elev_map[:, 0]), np.min(elev_map[:, 1])

            # define start on a grid
            robot_grid_pose = (robot_pose - [x_min, y_min, 0]) // map_res
            start_grid = (int(robot_grid_pose[0]), int(robot_grid_pose[1]))

            path_grid, goal_grid = breadth_first_search(grid, start_grid)
            # transform path to map coordintes (m)
            path = [(np.array(point)*0.15+[x_min, y_min]).tolist()+[elev_grid[point]] for point in path_grid]
            if len(path)>0:
                path = prune_path(path, 1e-3)
                # path = smooth_path(np.array(path), vis=1)
            path = np.array(path) - path[0,:] + robot_pose # start path exactly from robot location

            # publish path here
            publish_path(path, orient=[0,0,0,1], topic_name='resultant_path')
        rate.sleep()
    