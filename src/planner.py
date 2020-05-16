#!/usr/bin/env python

import rospy
from sensor_msgs.msg import PointCloud2
import tf
from nav_msgs.msg import Path
from utils import publish_path

import numpy as np
import matplotlib.pyplot as plt
from pointcloud_utils import pointcloud2_to_xyzrgb_array
from planning import breadth_first_search
from planning import prune_path, smooth_path
from planning import apf_planner, apf_path_to_map, draw_gradient
from grid import create_grid


class Planner:
    def __init__(self, elev_map_topic='/exploration/local_elev', pc_topic='/dynamic_point_cloud'):
        self.elev_map_topic = rospy.get_param('~elevation_map_topic', elev_map_topic)
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

def save_data(ind, folder_path='/home/ruslan/Desktop/CTU/catkin_ws/src/frontier_exploration/src/data/'):
    np.save(folder_path+'elev_map{}.npy'.format(ind), planner.local_map)
    np.save(folder_path+'dynamic_pc{}.npy'.format(ind), planner.dynamic_pc)
    np.save(folder_path+'robot_pose{}.npy'.format(ind), planner.robot_pose)
    print("saved data")


# define main parameters here
height_margin = 0.1 # traversable height margin: elevation map cells, higher than this value, are considered as untraversable
map_res = 0.15 # map resolution
safety_distance = 1*map_res # to keep away from obstacles (for grid creation)
unexplored_value = 0.5 # value of unknown cells in a constructed grid

# APF params
max_apf_iters = 200 # max num of iterations to plan an APF trajectory
influence_r = 0.15
repulsive_coef = 200
attractive_coef = 1./100


if __name__ == '__main__':
    rospy.init_node('path_planner')
    planner = Planner()
    
    ind = 0
    rate = rospy.Rate(1)
    plt.figure(figsize=(10,10))
    while not rospy.is_shutdown():

        if planner.local_map is not None and planner.robot_pose is not None:
            # save_data(ind)

            elev_map = planner.local_map; robot_pose = np.array(planner.robot_pose)
            grid, elev_grid = create_grid(elev_map, robot_pose[2], map_res, height_margin, safety_distance, unexplored_value)
            x_min, y_min = np.min(elev_map[:, 0]), np.min(elev_map[:, 1])

            # define start on a grid
            robot_grid_pose = (robot_pose - [x_min, y_min, 0]) // map_res
            start_grid = (int(robot_grid_pose[0]), int(robot_grid_pose[1]))

            # BFS
            bfs_path_grid, goal_grid = breadth_first_search(grid, start_grid, unexplored_value)
            # transform path to map coordintes (m)
            bfs_path = [(np.array(point)*map_res+[x_min, y_min]).tolist()+[elev_grid[point]] for point in bfs_path_grid]

            if len(bfs_path)>0:
                bfs_path = prune_path(bfs_path, 1e-3)
                # bfs_path = smooth_path(np.array(bfs_path), vis=False)
            bfs_path = np.array(bfs_path)

            # APF
            # if BFS found a frontier then do APF trajectory planning
            if goal_grid is not None:
                apf_path_grid, total_potential = apf_planner(grid, [start_grid[1], start_grid[0]], [goal_grid[1], goal_grid[0]],
                                                             max_apf_iters, influence_r, repulsive_coef, attractive_coef)
                # transform path to map coordintes (m)
                apf_path = apf_path_to_map(apf_path_grid, elev_map, elev_grid, map_res)
                apf_path = np.array(apf_path)

                apf_path = apf_path - apf_path[0,:] + robot_pose # start path exactly from robot location
                # publish paths here
                apf_path[:,2] += map_res # for better path visuaization with elevation map
                publish_path(apf_path, topic_name='/exploration/apf_path')
                bfs_path[:,2] += map_res # for better path visuaization with elevation map
                publish_path(bfs_path, topic_name='/exploration/bfs_path')

                # visualize APF and trajectories
                plt.cla()
                plt.imshow(1-grid, cmap='gray')
                plt.plot(goal_grid[1], goal_grid[0], 'ro', label='frontier goal')
                plt.plot(start_grid[1], start_grid[0], 'ro', color='g', label='current position')
                apf_path_grid = np.array(apf_path_grid); bfs_path_grid = np.array(bfs_path_grid)
                plt.plot(apf_path_grid[:,0], apf_path_grid[:,1], label='APF path')
                plt.plot(bfs_path_grid[:,1], bfs_path_grid[:,0], '--', label='BFS path')
                draw_gradient(total_potential)
                plt.legend()
                plt.draw()
                plt.pause(0.01)
        ind += 1        
        rate.sleep()
    plt.show()