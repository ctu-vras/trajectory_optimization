#!/usr/bin/env python

import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import CameraInfo
import tf, tf2_ros

import numpy as np
import torch
import pytorch3d
from pointcloud_utils import pointcloud2_to_xyz_array, xyz_array_to_pointcloud2
from pyquaternion import Quaternion
import copy

class Test:
    def __init__(self, pc_topic='/dynamic_point_cloud', cam_info_topic='/viz/camera_0/camera_info'):
        self.K = np.zeros((3, 3))
        self.pc_frame = None
        self.cam_frame = None
        self.points = None

        self.pc_topic = rospy.get_param('~pointcloud_topic', pc_topic)
        print("Subscribed to " + self.pc_topic)
        pc_sub = rospy.Subscriber(pc_topic, PointCloud2, self.pc_callback)

        self.cam_info_topic = rospy.get_param('~cam_info_topic', cam_info_topic)
        print("Subscribed to " + self.cam_info_topic)
        cam_info_sub = rospy.Subscriber(cam_info_topic, CameraInfo, self.cam_info_callback)

        self.tl = tf.TransformListener()

    @staticmethod
    def ego_to_cam(points, trans, pyquat):
        """Transform points (3 x N) from ego frame into a pinhole camera
        """
        points = points - np.expand_dims(trans, 1)
        rot = pyquat.rotation_matrix
        points = rot.T @ points
        return points

    @staticmethod
    def get_only_in_img_mask(pts, H, W, intrins):
        """pts should be 3 x N
        """
        pts = intrins @ pts
        pts[:2] /= pts[2:3]
        return (pts[2] > 0) & \
               (pts[0] > 1) & (pts[0] < W - 1) & \
               (pts[1] > 1) & (pts[1] < H - 1)

    @staticmethod
    def publish_pointcloud(points, topic_name, stamp, frame_id):
        # create PointCloud2 msg
        pc_msg = xyz_array_to_pointcloud2(points, stamp=stamp, frame_id=frame_id)
        pub = rospy.Publisher(topic_name, PointCloud2, queue_size=1)
        pub.publish(pc_msg)

    def pc_callback(self, pc_msg):
        dynamic_pc = pointcloud2_to_xyz_array(pc_msg)
        self.pc_frame = pc_msg.header.frame_id
        self.points = dynamic_pc.T

    def cam_info_callback(self, cam_info_msg):
        W = cam_info_msg.width
        H = cam_info_msg.height
        self.cam_frame = cam_info_msg.header.frame_id
        self.K[0][0] = cam_info_msg.K[0]
        self.K[0][2] = cam_info_msg.K[2]
        self.K[1][1] = cam_info_msg.K[4]
        self.K[1][2] = cam_info_msg.K[5]
        self.K[2][2] = 1

        if self.pc_frame is not None:  # and self.tl.frameExists("map"):
            # find transformation between lidar and camera
            t = self.tl.getLatestCommonTime(self.pc_frame, self.cam_frame)
            trans, quat = self.tl.lookupTransform(self.pc_frame, self.cam_frame, t)
            pyquat = Quaternion(w=quat[3], x=quat[0], y=quat[1], z=quat[2]).normalised
            trans = np.array(trans)

            # project points to camera coordinate frame
            ego_pts = self.ego_to_cam(copy.deepcopy(self.points), trans, pyquat)
            # find points that are observed by the camera (in its FOV)
            frame_mask = self.get_only_in_img_mask(ego_pts, H, W, self.K)
            cam_pts = ego_pts[:, frame_mask]

            # clip points between 1.0 and 5.0 meters distance from the camera
            dist_mask = (cam_pts[2] > 1.0) & (cam_pts[2] < 5.0)
            cam_pts = cam_pts[:, dist_mask]

            self.publish_pointcloud(cam_pts.T, '/tmp_pointcloud', rospy.Time.now(), self.cam_frame)


if __name__ == '__main__':
    rospy.init_node('test_node')
    test = Test()
    
    rospy.spin()
