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
        # print('X', np.min(pc_numpy[:,0]), np.max(pc_numpy[:,0]))
        # print('Y', np.min(pc_numpy[:,1]), np.max(pc_numpy[:,1]))
        # print('Z', np.min(pc_numpy[:,2]), np.max(pc_numpy[:,2]))
        # voxel = pc_to_voxel(pc_numpy, resolution=0.15, x=(np.min(pc_numpy[:,0]), np.max(pc_numpy[:,0])),
        #                                                y=(np.min(pc_numpy[:,1]), np.max(pc_numpy[:,1])),
        #                                                z=(np.min(pc_numpy[:,2]), np.max(pc_numpy[:,2])))
        # print(voxel.shape)
        # voxel_msg = array_to_pointcloud2(voxel, stamp=pc_msg.header.stamp, frame_id=pc_msg.header.frame_id)

    def pc_callback(self, pc_msg):
        self.dynamic_pc = pointcloud2_to_xyzrgb_array(pc_msg)


if __name__ == '__main__':
    rospy.init_node('elev_map_subscriber')
    planner = Planner()
    
    while not rospy.is_shutdown():
        # if planner.robot_pose is not None:
        #     print(planner.robot_pose[:2])
        if planner.local_map is not None and planner.dynamic_pc is not None:
            path = '/home/ruslan/Desktop/CTU/data/'
            np.save(path+'elev_map{}.npy'.format(time.time()), planner.local_map)
            np.save(path+'dynamic_pc{}.npy'.format(time.time()), planner.dynamic_pc)
            np.save(path+'robot_pose{}.npy'.format(time.time()), planner.robot_pose)
            print("saved data")
        time.sleep(3)
    