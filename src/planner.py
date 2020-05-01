#!/usr/bin/env python

import rospy
from sensor_msgs.msg import PointCloud2, PointField
import tf

import numpy as np
from utils import pointcloud2_to_xyzrgb_array
from utils import pc_to_voxel
from utils import array_to_pointcloud2
import time


class Planner:
    def __init__(self, elev_map_topic='/exploration/local_elev'):
        self.elev_map_topic = rospy.get_param('~pointcloud_topic', elev_map_topic)
        print("Subscribed to " + self.elev_map_topic)
        pc_sub = rospy.Subscriber(elev_map_topic, PointCloud2, self.pc_callback)

        self.tl = tf.TransformListener()

        self.goal = None
        self.local_map = None

    def get_robot_pose(self):
        position, quaternion = None, None
        if self.tl.frameExists("base_link") and self.tl.frameExists("map"):
            t = self.tl.getLatestCommonTime("base_link", "map")
            position, quaternion = self.tl.lookupTransform("base_link", "map", t)
        return position, quaternion

    @property
    def robot_pose(self):
        return self.get_robot_pose()[0]

    @property
    def robot_orient(self):
        return self.get_robot_pose()[1]
    

    def pc_callback(self, elev_map_pc_msg):
        pc_numpy = pointcloud2_to_xyzrgb_array(elev_map_pc_msg)
        self.local_map = pc_numpy
        # print('X', np.min(pc_numpy[:,0]), np.max(pc_numpy[:,0]))
        # print('Y', np.min(pc_numpy[:,1]), np.max(pc_numpy[:,1]))
        # print('Z', np.min(pc_numpy[:,2]), np.max(pc_numpy[:,2]))
        # voxel = pc_to_voxel(pc_numpy, resolution=0.15, x=(np.min(pc_numpy[:,0]), np.max(pc_numpy[:,0])),
        #                                                y=(np.min(pc_numpy[:,1]), np.max(pc_numpy[:,1])),
        #                                                z=(np.min(pc_numpy[:,2]), np.max(pc_numpy[:,2])))
        # print(voxel.shape)
        # voxel_msg = array_to_pointcloud2(voxel, stamp=pc_msg.header.stamp, frame_id=pc_msg.header.frame_id)


if __name__ == '__main__':
    rospy.init_node('elev_map_subscriber')
    planner = Planner()
    
    while not rospy.is_shutdown():
        if planner.local_map is not None:
            print(np.mean(planner.local_map))
        time.sleep(0.1)
    