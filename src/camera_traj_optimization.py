#!/usr/bin/env python

import sys
import os
import rospkg
FE_PATH = rospkg.RosPack().get_path('frontier_exploration')
sys.path.append(os.path.join(FE_PATH, 'src/'))
import torch
from pytorch3d.transforms import matrix_to_quaternion
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import cv2
from copy import deepcopy

# ROS libraries
import rospy
from tools import load_intrinsics
from tools import publish_odom
from tools import publish_pointcloud
from tools import publish_tf_pose
from tools import publish_camera_info
from tools import publish_image
from tools import publish_path
from tools import to_pose_stamped
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped


class Model(nn.Module):
    def __init__(self,
                 points: torch.tensor,
                 traj_wps: list,  # [np.array([x0, y0, z0]), np.array([x1, y1, z1]), ...]
                 min_dist=1.0, max_dist=10.0,
                 dist_rewards_mean=3.0, dist_rewards_sigma=2.0):
        super().__init__()
        self.points = points
        self.rewards = None
        self.observations = None
        self.device = points.device
        self.lo_sum = 0.0  # log odds sum for the entire point cloud for the whole trajectory

        # Create an optimizable parameter for the x, y, z position of the camera.
        self.rot0 = torch.eye(3, device=self.device).unsqueeze(0)
        for i in range(len(traj_wps) - 1):
            self.rot0 = torch.cat([self.rot0, torch.eye(3, device=self.device).unsqueeze(0)])  # (N, 3, 3)
        self.traj0 = torch.from_numpy(np.array(traj_wps, dtype=np.float32)).to(self.device)  # (N, 3)

        self.traj = nn.Parameter(deepcopy(self.traj0))
        self.rots = nn.Parameter(self.rot0)

        self.K, self.width, self.height = load_intrinsics(device=self.device)
        self.eps = 1e-6
        self.pc_clip_limits = [min_dist, max_dist]  # [m]
        self.dist_rewards = {'mean': dist_rewards_mean, 'dist_rewards_sigma': dist_rewards_sigma}

        self.loss_vis = None
        self.loss_traj = {'l2': None, 'vel': None, 'acc': None, 'jerk': None}

    @staticmethod
    def get_dist_mask(points, min_dist=1.0, max_dist=5.0):
        # clip points between MIN_DIST and MAX_DIST meters distance from the camera
        dist_mask = (points[2] > min_dist) & (points[2] < max_dist)
        return dist_mask

    @staticmethod
    def get_fov_mask(points, img_height, img_width, intrins):
        # find points that are observed by the camera (in its FOV)
        pts_homo = intrins[:3, :3] @ points
        pts_homo[:2] /= pts_homo[2:3]
        fov_mask = (pts_homo[2] > 0) & (pts_homo[0] > 1) & \
                   (pts_homo[0] < img_width - 1) & (pts_homo[1] > 1) & \
                   (pts_homo[1] < img_height - 1)
        return fov_mask

    def to_camera_frame(self, verts, R, T):
        R_inv = torch.transpose(torch.squeeze(R, 0), 0, 1)
        verts = torch.transpose(verts - torch.repeat_interleave(T, len(verts), dim=0).to(self.device), 0, 1)
        verts = torch.matmul(R_inv, verts)
        verts = torch.transpose(verts, 0, 1)
        return verts

    @staticmethod
    def gaussian(x, mu=3.0, sigma=100.0, normalize=False):
        # https://en.wikipedia.org/wiki/Normal_distribution
        g = torch.exp(-0.5 * ((x - mu) / sigma) ** 2)
        if normalize:
            g /= (sigma * torch.sqrt(torch.tensor(2 * np.pi)))
        return g

    def distance_visibility(self, verts, pose):
        # compute observations based on distance of the surrounding points
        dists = torch.linalg.norm(pose - verts, dim=1)
        rewards = self.gaussian(dists, mu=self.dist_rewards['mean'], sigma=self.dist_rewards['dist_rewards_sigma'])
        return rewards

    def forward(self):
        """
        Trajectory evaluation based on visibility estimation from its waypoints.
        traj_score = log_odds_sum([visibility_estimation(wp) for wp in traj_waypoints])
        """
        N_wps = self.traj.size()[0]
        lo_sum = 0.0
        for i in range(N_wps):
            # transform points to camera frame
            verts = self.to_camera_frame(self.points, self.rots[i].unsqueeze(0), self.traj[i].unsqueeze(0))
            verts = torch.transpose(verts, 0, 1)

            # get masks of points that are inside of the camera FOV
            dist_mask = self.get_dist_mask(verts, self.pc_clip_limits[0], self.pc_clip_limits[1])
            fov_mask = self.get_fov_mask(verts, self.height, self.width, self.K.squeeze(0))

            mask = torch.logical_and(dist_mask, fov_mask)

            p = self.distance_visibility(self.points,
                                         self.traj[i].unsqueeze(0)) * mask  # local observations reward (visibility)

            # apply log odds conversion for global voxel map observations update
            p = torch.clip(p, 0.5, 1 - self.eps)
            lo = torch.log(p / (1 - p))
            lo_sum = lo_sum + lo

        self.rewards = 1 / (1 + torch.exp(-lo_sum))  # total trajectory observations
        loss = self.criterion(self.rewards)
        return loss

    @staticmethod
    def smoothness_est(traj):
        vel = torch.linalg.norm(traj[1:] - traj[:-1], dim=1)
        acc = vel[1:] - vel[:-1]
        jerk = acc[1:] - acc[:-1]
        return torch.mean(torch.abs(vel)), torch.mean(torch.abs(acc)), torch.mean(torch.abs(jerk))

    def criterion(self, rewards, l2_lambda=0.004):  # , l2_lambda=0.0004):
        # transform observations to loss function: loss = 1 / sum(prob(observed))
        self.loss_vis = len(self.points) / (torch.sum(rewards) + self.eps)
        # add penalties for close waypoints
        self.loss_traj['l2'] = 0.0
        for i in [0, -1]:  # range(len(self.traj0)):
            self.loss_traj['l2'] += torch.linalg.norm(self.traj[i] - self.traj0[i])
        self.loss_traj['vel'], self.loss_traj['acc'], self.loss_traj['jerk'] = self.smoothness_est(self.traj)
        for key in self.loss_traj: self.loss_traj[key] *= l2_lambda
        return self.loss_vis + self.loss_traj['l2'] + self.loss_traj['vel'] + self.loss_traj['acc'] + self.loss_traj[
            'jerk']


if __name__ == "__main__":
    rospy.init_node('camera_traj_optimization')
    # Load the point cloud and initial trajectory to optimize
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    # Set paths
    index = 1612893730.3432848
    points_filename = os.path.join(FE_PATH, f"data/points/point_cloud_{index}.npz")
    pts_np = np.load(points_filename)['pts'].transpose()

    # make sure the point cloud is of (N x 3) shape:
    if pts_np.shape[1] > pts_np.shape[0]:
        pts_np = pts_np.transpose()
    points = torch.tensor(pts_np, dtype=torch.float32).to(device)

    poses_filename = os.path.join(FE_PATH, f"data/paths/path_poses_{index}.npz")
    traj_0 = np.load(poses_filename)['poses'].tolist()

    # Initialize a model
    model = Model(points=points,
                  traj_wps=traj_0).to(device)
    # Create an optimizer. Here we are using Adam and we pass in the parameters of the model
    optimizer = torch.optim.Adam(model.parameters(), lr=0.5)

    # Run optimization loop
    loop = tqdm(range(100))
    for i in loop:
        if rospy.is_shutdown():
            break
        optimizer.zero_grad()
        loss = model()
        loss.backward()
        optimizer.step()

        if i % 4 == 0:
            print(f"Loss: {loss.item()}")
            print(f"Trajectory visibility score: {torch.sum(model.rewards)}")

            # publish ROS msgs
            intensity = model.rewards.detach().unsqueeze(1).cpu().numpy()
            # print(np.min(intensity), np.mean(intensity), np.max(intensity))
            points = np.concatenate([pts_np, intensity], axis=1)  # add observations for pts intensity visualization
            publish_pointcloud(points, '/pts', rospy.Time.now(), 'world')
            publish_path(traj_0, topic_name='/path/initial', frame_id='world')
            publish_path(model.traj.detach(), topic_name='/path/optimized', frame_id='world')
