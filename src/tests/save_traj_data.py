#!/usr/bin/env python

import sys
import rospkg
import os
FE_PATH = rospkg.RosPack().get_path('frontier_exploration')
sys.path.append(os.path.join(FE_PATH, 'src/'))
import rospy
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Path
import tf, tf2_ros

import torch
from pytorch3d.transforms import quaternion_apply, quaternion_invert
from pointcloud_utils import pointcloud2_to_xyz_array, xyz_array_to_pointcloud2
import numpy as np
from pyquaternion import Quaternion
import time


class PointsProcessor:
    def __init__(self,
                 pc_topic='/final_cost_cloud',
                 path_topic='/path',
                 ):
        self.pc_frame = None
        self.points = None
        self.path = {'poses': [], 'orients': []}
        self.path_frame = None
        self.cam_frame = 'camera_0'

        self.device = torch.device("cuda:0")

        self.pc_topic = rospy.get_param('~pointcloud_topic', pc_topic)
        print("Subscribed to " + self.pc_topic)
        rospy.Subscriber(pc_topic, PointCloud2, self.pc_callback)

        self.path_topic = rospy.get_param('~path_topic', path_topic)
        print("Subscribed to " + self.path_topic)
        rospy.Subscriber(path_topic, Path, self.path_callback)

        self.tl = tf.TransformListener()

    def points_to_cam_frame(self, points):
        """Transform points (N x 3) to camera frame
        """
        # find transformation between lidar and camera
        t = self.tl.getLatestCommonTime(self.pc_frame, self.cam_frame)
        trans, quat = self.tl.lookupTransform(self.pc_frame, self.cam_frame, t)

        # transform point cloud to camera frame
        quat = torch.tensor([quat[3], quat[0], quat[1], quat[2]]).to(self.device)
        points = torch.from_numpy(points).to(self.device)
        trans = torch.unsqueeze(torch.tensor(trans), 0).to(self.device)

        points = points - trans
        quat_inv = quaternion_invert(quat)
        points = quaternion_apply(quat_inv, points)
        return points.T.float()

    def pc_callback(self, pc_msg):
        points = pointcloud2_to_xyz_array(pc_msg)
        self.pc_frame = pc_msg.header.frame_id
        self.points = points

    @staticmethod
    def publish_pointcloud(points, topic_name, stamp, frame_id):
        # create PointCloud2 msg
        pc_msg = xyz_array_to_pointcloud2(points, stamp=stamp, frame_id=frame_id)
        pub = rospy.Publisher(topic_name, PointCloud2, queue_size=1)
        pub.publish(pc_msg)

    def path_callback(self, path_msg):
        self.path_frame = path_msg.header.frame_id
        self.path['poses'] = []
        self.path['orients'] = []
        msg = Path()
        msg.header.stamp = path_msg.header.stamp
        msg.header.frame_id = self.cam_frame
        msg.poses = path_msg.poses
        path_pub = rospy.Publisher('/path_cam', Path, queue_size=1)
        for i in range(len(path_msg.poses)):
            pose = path_msg.poses[i]
            # transform poses to camera frame
            cam_pose, cam_quat = self.pose(self.cam_frame)
            # self.path['poses'].append(cam_quat.inverse.rotate(np.array([pose.pose.position.x,
            #                                                             pose.pose.position.y,
            #                                                             pose.pose.position.z]) - cam_pose))
            self.path['poses'].append(np.array([pose.pose.position.x,
                                                pose.pose.position.y,
                                                pose.pose.position.z]))
            msg.poses[i].pose.position.x = self.path['poses'][-1][0]
            msg.poses[i].pose.position.y = self.path['poses'][-1][1]
            msg.poses[i].pose.position.z = self.path['poses'][-1][2]
            # transform orients to camera frame
            # self.path['orients'].append(Quaternion(x=pose.pose.orientation.x,
            #                                        y=pose.pose.orientation.y,
            #                                        z=pose.pose.orientation.z,
            #                                        w=pose.pose.orientation.w) * cam_quat.inverse)
            self.path['orients'].append(Quaternion(x=pose.pose.orientation.x,
                                                   y=pose.pose.orientation.y,
                                                   z=pose.pose.orientation.z,
                                                   w=pose.pose.orientation.w))
            msg.poses[i].pose.orientation.x = self.path['orients'][-1].x
            msg.poses[i].pose.orientation.y = self.path['orients'][-1].y
            msg.poses[i].pose.orientation.z = self.path['orients'][-1].z
            msg.poses[i].pose.orientation.w = self.path['orients'][-1].w

        if self.points is not None:
            # transform point cloud to camera frame
            # self.points = self.points_to_cam_frame(self.points).cpu().numpy().T
            points_np = self.points

            self.publish_pointcloud(points_np,
                                    '/pc_cam',
                                    rospy.Time.now(),
                                    self.cam_frame)

            index = time.time()
            np.savez(f'../traj_data/paths/path_poses_{index}.npz', poses=np.asarray(self.path['poses']))
            np.savez(f'../traj_data/points/point_cloud_{index}.npz', pts=points_np)
        path_pub.publish(msg)

    def pose(self, frame, world_frame='map'):
        # find frame location in world_frame
        t = self.tl.getLatestCommonTime(world_frame, frame)
        trans, quat = self.tl.lookupTransform(world_frame, frame, t)
        return np.asarray(trans), Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])


if __name__ == '__main__':
    rospy.init_node('pc_processor_node')
    proc = PointsProcessor()
    rospy.spin()
