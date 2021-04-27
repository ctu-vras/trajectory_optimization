#!/usr/bin/env python

import sys
import rospkg
import os
FE_PATH = rospkg.RosPack().get_path('trajectory_optimization')
sys.path.append(os.path.join(FE_PATH, 'src/'))
import torch
from model import ModelTraj
import numpy as np
from time import time
from tqdm import tqdm
# ROS libs
import rospy
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Path
import tf, tf2_ros
import message_filters
from tools import publish_path
from tools import publish_pointcloud
from tools import load_intrinsics
from pointcloud_utils import pointcloud2_to_xyz_array


class TrajOpt:
    def __init__(self,
                 pc_topic='/final_cost_cloud',
                 input_path_topic='/path',
                 publish_rewards_cloud=False,
                 device=torch.device("cuda:0")
                 ):
        self.device = device
        self.pc_frame = None
        self.points = None
        self.path = {'poses': None, 'orients': None}
        self.path_frame = None
        self.publish_rewards_cloud = publish_rewards_cloud

        self.K, self.img_width, self.img_height = load_intrinsics(device=self.device)

        ## Get trajectory optimization parameters values
        self.n_opt_steps = rospy.get_param('traj_opt/opt_steps', 10)
        self.smooth_weight = rospy.get_param('traj_opt/smooth_weight', 14.0)
        self.length_weight = rospy.get_param('traj_opt/length_weight', 0.02)
        self.lr_pose = rospy.get_param('traj_opt/lr_pose', 0.1)
        self.lr_quat = rospy.get_param('traj_opt/lr_quat', 0.0)

        self.pc_topic = pc_topic
        print("Subscribing to " + self.pc_topic)

        self.input_path_topic = input_path_topic
        print("Subscribing to " + self.input_path_topic)

        points_sub = message_filters.Subscriber(self.pc_topic, PointCloud2)
        path_sub = message_filters.Subscriber(self.input_path_topic, Path)

        ts = message_filters.ApproximateTimeSynchronizer([points_sub, path_sub], 10, slop=0.5)
        ts.registerCallback(self.callback)

    def get_data(self, pc_msg, path_msg):
        # get point cloud tensor from ros msg
        pts_np = pointcloud2_to_xyz_array(pc_msg)
        points = torch.from_numpy(pts_np).float().to(self.device)

        # get path poses and orients tensors from ros msg
        poses = []
        orients = []
        for i in range(len(path_msg.poses)):
            pose = path_msg.poses[i]
            poses.append(np.array([pose.pose.position.x,
                                   pose.pose.position.y,
                                   pose.pose.position.z]))

            orients.append(np.array([pose.pose.orientation.w,
                                     pose.pose.orientation.x,
                                     pose.pose.orientation.y,
                                     pose.pose.orientation.z]))

        poses = torch.from_numpy(np.asarray(poses)).float().to(self.device)
        orients = torch.from_numpy(np.asarray(orients)).float().to(self.device)
        return points, poses, orients

    def init_model(self):
        model = ModelTraj(points=self.points,
                          wps_poses=self.path['poses'],
                          wps_quats=self.path['orients'],
                          intrins=self.K,
                          img_width=self.img_width, img_height=self.img_height,
                          device=self.device).to(self.device)

        optimizer = torch.optim.Adam([
            {'params': list([model.poses]), 'lr': self.lr_pose},
            {'params': list([model.quats]), 'lr': self.lr_quat},
        ])
        return model, optimizer

    def quat_wxyz_to_xyzw(self, quat):
        return torch.tensor([quat[1], quat[2], quat[3], quat[0]], device=self.device)

    def run(self, model, optimizer, rewards_th=1.25, smoothness_th=0.9):
        # optimization loop
        FIRST_RECORD = True
        reward0 = 1e-6
        smooth_loss0 = 0.0
        t_step = 0.0
        for i in tqdm(range(self.n_opt_steps)):
            t0 = time()
            # optimization step: ~125 msec
            optimizer.zero_grad()
            loss = model()
            if FIRST_RECORD:
                reward0 = torch.mean(model.rewards)
                smooth_loss0 = model.loss['smooth']
                FIRST_RECORD = False
            loss.backward()
            optimizer.step()
            t_step += 1000 * (time() - t0)

            visibility_gain = torch.mean(model.rewards) / reward0
            smooth_gain = smooth_loss0 / model.loss['smooth']
            if visibility_gain > rewards_th and \
                    smooth_gain > smoothness_th:
                print(f'Found optimal trajectory with rewards coef={visibility_gain} after {i} steps')
                break

        print(f'Optimization step took {t_step / i} msec')
        print(f'Input point cloud size: {self.points.size()}')

    def callback(self, pc_msg, path_msg):
        # convert ros msgs to tensors
        self.points, self.path['poses'], self.path['orients'] = self.get_data(pc_msg, path_msg)

        # initialize a model
        model, optimizer = self.init_model()

        self.run(model, optimizer, rewards_th=1.25, smoothness_th=0.9)

        # publish optimized path with positions and orientations
        self.path_frame = path_msg.header.frame_id  # map
        print(f'Publishing optimized path in frame {self.path_frame}')
        poses_to_pub = model.poses.detach()
        quats_to_pub = [self.quat_wxyz_to_xyzw(quat / torch.linalg.norm(quat)) for quat in model.quats.detach()]
        publish_path(poses_to_pub, quats_to_pub,
                     topic_name=self.input_path_topic+'/optimized',
                     frame_id=self.path_frame)

        if self.publish_rewards_cloud:
            # publish colored point cloud for debugging
            self.pc_frame = pc_msg.header.frame_id  # map
            intensity = model.rewards.detach().unsqueeze(1).cpu().numpy()
            # print(np.min(intensity), np.mean(intensity), np.max(intensity))
            pts_np = self.points.cpu().numpy()
            points = np.concatenate([pts_np, intensity], axis=1)  # add rewards for pts intensity visualization
            publish_pointcloud(points,
                               topic_name=self.pc_topic+'/rewards',
                               stamp=rospy.Time.now(),
                               frame_id=self.pc_frame)


if __name__ == '__main__':
    rospy.init_node('trajopt_node')
    proc = TrajOpt(pc_topic=rospy.get_param('traj_opt/point_cloud_topic', '/final_cost_cloud'),
                   input_path_topic=rospy.get_param('traj_opt/input_path_topic', '/path'),
                   publish_rewards_cloud=rospy.get_param('traj_opt/publish_rewards_cloud', False))
    rospy.spin()
