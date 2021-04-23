#!/usr/bin/env python

import sys
import rospkg
import os
FE_PATH = rospkg.RosPack().get_path('trajectory_optimization')
sys.path.append(os.path.join(FE_PATH, 'src/'))
import torch
import torch.nn.functional as F
from model import ModelPose
import numpy as np
from time import time
from tqdm import tqdm
import cv2
# ROS libs
import rospy
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped
import tf, tf2_ros
import message_filters
from tools import publish_odom
from tools import publish_pointcloud
from tools import publish_tf_pose
from tools import publish_camera_info
from tools import publish_image
from tools import load_intrinsics
from tools import render_pc_image
from pointcloud_utils import pointcloud2_to_xyz_array


class PoseOpt:
    def __init__(self,
                 pc_topic='/pts',
                 input_pose_topic='/pose',
                 device=torch.device("cuda:0")
                 ):
        self.device = device
        self.pc_frame = None
        self.points = None
        self.pose = {'trans': None, 'quat': None}
        self.pose_frame = None
        self.model = None
        self.rate = rospy.Rate(rospy.get_param('pose_opt/rate', 0.5))

        self.K, self.img_width, self.img_height = load_intrinsics(device=self.device)

        ## Get trajectory optimization parameters values
        self.n_opt_steps = rospy.get_param('pose_opt/opt_steps', 10)
        self.lr_pose = rospy.get_param('pose_opt/lr_pose', 0.1)
        self.lr_quat = rospy.get_param('pose_opt/lr_quat', 0.0)

        self.pc_topic = pc_topic
        print("Subscribing to " + self.pc_topic)

        self.input_pose_topic = input_pose_topic
        print("Subscribing to " + self.input_pose_topic)

        points_sub = message_filters.Subscriber(self.pc_topic, PointCloud2)
        pose_sub = message_filters.Subscriber(self.input_pose_topic, PoseStamped)

        ts = message_filters.ApproximateTimeSynchronizer([points_sub, pose_sub], 10, slop=0.5)
        ts.registerCallback(self.callback)

    def get_data(self, pc_msg, pose_msg):
        # get point cloud tensor from ros msg
        pts_np = pointcloud2_to_xyz_array(pc_msg)

        # get path poses and orients tensors from ros msg
        trans_np = np.array([[pose_msg.pose.position.x,
                              pose_msg.pose.position.y,
                              pose_msg.pose.position.z]], dtype=np.float32)

        quat_wxyz_np = np.array([[pose_msg.pose.orientation.w,
                                  pose_msg.pose.orientation.x,
                                  pose_msg.pose.orientation.y,
                                  pose_msg.pose.orientation.z,
                                  ]], dtype=np.float32)

        return pts_np, trans_np, quat_wxyz_np

    def init_model(self):
        # Initialize a model
        model = ModelPose(points=self.points,
                          trans0=self.pose['trans'],
                          q0=self.pose['quat'],
                          intrins=self.K,
                          img_width=self.img_width, img_height=self.img_height,
                          min_dist=1.0, max_dist=5.0,
                          device=self.device).to(self.device)

        # Create an optimizer. Here we are using Adam and we pass in the parameters of the model
        optimizer = torch.optim.Adam([
            {'params': list([model.trans]), 'lr': self.lr_pose},
            {'params': list([model.quat]), 'lr': self.lr_quat},
        ])
        return model, optimizer

    def publish_data(self, pts_np=None, points_visible=None):
        # publish optimized position and orientation
        trans = self.model.trans.squeeze()
        quat = F.normalize(self.model.quat).squeeze()
        quat = (quat[1], quat[2], quat[3], quat[0])
        publish_odom(trans, quat, frame=self.pose_frame, topic='/odom')
        publish_tf_pose(trans, quat, "camera_frame", frame_id=self.pose_frame)
        publish_camera_info(topic_name="/camera/camera_info", frame_id="camera_frame")

        if pts_np is not None:
            intensity = self.model.observations.unsqueeze(1).detach().cpu().numpy()
            pts_rewards = np.concatenate([pts_np, intensity],
                                          axis=1)  # add observations for pts intensity visualization
            publish_pointcloud(pts_rewards, self.pc_topic+'/rewards', rospy.Time.now(), self.pc_frame)

        if points_visible is not None:
            # render point cloud image
            if points_visible.size()[0] > 0:
                image = render_pc_image(points_visible,
                                        self.K,
                                        self.img_height, self.img_width,
                                        device=self.device)

                image_vis = cv2.resize(image.detach().cpu().numpy(), (600, 800))
                publish_image(image_vis, topic='/pc_image')

            points_visible_np = points_visible.detach().cpu().numpy()
            publish_pointcloud(points_visible_np, '/pts_visible', rospy.Time.now(), 'camera_frame')

    def callback(self, pc_msg, pose_msg):
        # convert ros msgs to tensors
        pts_np, trans_np, quat_np = self.get_data(pc_msg, pose_msg)
        self.points = torch.from_numpy(pts_np).float().to(self.device)
        self.pose['trans'] = torch.from_numpy(trans_np).to(self.device)
        self.pose['quat'] = torch.from_numpy(quat_np).to(self.device)

        # initialize a model
        self.model, optimizer = self.init_model()

        self.pc_frame = pc_msg.header.frame_id
        self.pose_frame = pose_msg.header.frame_id
        # optimization loop
        for i in tqdm(range(self.n_opt_steps)):
            t0 = time()
            # optimization step: ~10 msec
            optimizer.zero_grad()
            points_visible, loss = self.model()
            loss.backward()
            optimizer.step()

            if i % int(self.n_opt_steps // 10) == 0:  # publish 10 times
                # print(f'Optimization step took {1000*(time()-t0)} msec')
                # self.publish_data(pts_np, points_visible)
                self.publish_data(pts_np=None, points_visible=None)

        self.rate.sleep()

if __name__ == '__main__':
    rospy.init_node('pose_opt_node')
    proc = PoseOpt(pc_topic=rospy.get_param('pose_opt/point_cloud_topic', '/pts'),
                   input_pose_topic=rospy.get_param('pose_opt/pose_topic', '/pose'))
    rospy.spin()
