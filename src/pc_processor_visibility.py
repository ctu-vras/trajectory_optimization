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
import open3d as o3d
from time import time


class PointCloudProcessor:
    def __init__(self,
                 pc_topic='/final_cost_cloud',
                 cam_info_topics=['/viz/camera_0/camera_info',
                                  # '/viz/camera_1/camera_info',
                                  # '/viz/camera_2/camera_info',
                                  # '/viz/camera_3/camera_info',
                                  # '/viz/camera_4/camera_info',
                                  # '/viz/camera_5/camera_info',
                                  ],
                 min_dist=1.0,
                 max_dist=10.0):
        self.pc_frame = None
        self.points = None
        self.tl = tf.TransformListener()
        self.pc_clip_limits = [rospy.get_param('~min_dist', min_dist),
                               rospy.get_param('~max_dist', max_dist)]

        self.pc_topic = rospy.get_param('~pointcloud_topic', pc_topic)
        print("Subscribed to " + self.pc_topic)
        pc_sub = rospy.Subscriber(pc_topic, PointCloud2, self.pc_callback)

        for topic in cam_info_topics:
            print("Subscribed to " + topic)
            cam_info_sub = rospy.Subscriber(topic, CameraInfo, self.cam_info_callback)

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

    @staticmethod
    def remove_hidden_pts(pts):
        # transformations from ROS coord system to Open3d
        angle = np.pi
        # Rx = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
        Ry = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
        Rz = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        pts = Ry @ Rz @ pts.T
        pts = pts.T
        # define Open3d point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        diameter = np.linalg.norm(
            np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
        if diameter > 0:
            # print("Define parameters used for hidden_point_removal")
            # camera = [0, 0, diameter]
            camera = [0, 0, 0.]
            radius = diameter * 100
            # print("Get all points that are visible from given view point")
            _, pt_map = pcd.hidden_point_removal(camera, radius)
            # print("Visualize result")
            pcd_res = pcd.select_by_index(pt_map)
            pts_visible = np.asarray(pcd_res.points)
        else:
            # print('All the pts are visible here')
            pts_visible = pts
        # back to ROS coord system
        pts_visible = Rz.T @ Ry.T @ pts_visible.T
        pts_visible = pts_visible.T
        return pts_visible

    def pc_callback(self, pc_msg):
        points = pointcloud2_to_xyz_array(pc_msg)
        self.pc_frame = pc_msg.header.frame_id
        self.points = points.T

    def cam_info_callback(self, cam_info_msg):
        fovW = cam_info_msg.width
        fovH = cam_info_msg.height
        cam_frame = cam_info_msg.header.frame_id
        K = np.zeros((3, 3))
        K[0][0] = cam_info_msg.K[0]
        K[0][2] = cam_info_msg.K[2]
        K[1][1] = cam_info_msg.K[4]
        K[1][2] = cam_info_msg.K[5]
        K[2][2] = 1

        if self.pc_frame is not None:  # and self.tl.frameExists(self.pc_frame):
            self.run(fovH, fovW, K, cam_frame, output_pc_topic=f'/{cam_frame}/pointcloud')

    def run(self, fovH, fovW, K, cam_frame, output_pc_topic):
        # find transformation between lidar and camera
        t = self.tl.getLatestCommonTime(self.pc_frame, cam_frame)
        trans, quat = self.tl.lookupTransform(self.pc_frame, cam_frame, t)
        pyquat = Quaternion(w=quat[3], x=quat[0], y=quat[1], z=quat[2]).normalised
        trans = np.array(trans)

        # project points to camera coordinate frame
        pts_in_cam_frame = self.ego_to_cam(copy.deepcopy(self.points), trans, pyquat)

        # find points that are observed by the camera (in its FOV)
        frame_mask = self.get_only_in_img_mask(pts_in_cam_frame, fovH, fovW, K)
        cam_pts = pts_in_cam_frame[:, frame_mask]

        # clip points between 1.0 and 10.0 meters distance from the camera
        dist_mask = (cam_pts[2] > self.pc_clip_limits[0]) & \
                    (cam_pts[2] < self.pc_clip_limits[1])
        cam_pts = cam_pts[:, dist_mask]
        # np.savez(f'cam_pts_{cam_frame}.npz', pts=cam_pts)

        # remove hidden points from current camera FOV
        cam_pts_visible = self.remove_hidden_pts(cam_pts.T)

        self.publish_pointcloud(cam_pts.T,
                                output_pc_topic,
                                rospy.Time.now(),
                                cam_frame)
        self.publish_pointcloud(cam_pts_visible,
                                output_pc_topic + '_visible',
                                rospy.Time.now(),
                                cam_frame)


if __name__ == '__main__':
    rospy.init_node('pc_processor_node')
    proc = PointCloudProcessor()
    rospy.spin()
