import sys
import os
FE_PATH = '/home/ruslan/Documents/CTU/catkin_ws/src/frontier_exploration/'
sys.path.append(os.path.join(FE_PATH, 'src/'))
import torch
import torch.nn as nn
import numpy as np
from pytorch3d.transforms import euler_angles_to_matrix
from tools import load_intrinsics, hidden_pts_removal


def rewards_from_pose(x, y, z,
                      roll, pitch, yaw,
                      verts,
                      min_dist=1.0, max_dist=10.0,
                      device=torch.device('cuda'),
                      hpr=False,  # whether to use hidden points removal algorithm
                      ):
    K, img_width, img_height = load_intrinsics()
    intrins = K.squeeze(0)
    R = euler_angles_to_matrix(torch.tensor([roll, pitch, yaw]), "XYZ").unsqueeze(0).to(device)
    T = torch.tensor([x, y, z], device=device).unsqueeze(0)

    # transform points to camera frame
    R_inv = torch.transpose(torch.squeeze(R, 0), 0, 1)
    verts = torch.transpose(verts - torch.repeat_interleave(T, len(verts), dim=0).to(device), 0, 1)
    verts = torch.matmul(R_inv, verts)

    # HPR: remove occluded points (currently works only on CPU)
    if hpr:
        verts, occl_mask = hidden_pts_removal(torch.transpose(verts, 0, 1).detach(), device=device)
        verts = torch.transpose(verts, 0, 1)

    # get masks of points that are inside of the camera FOV
    dist_mask = (verts[2] > min_dist) & (verts[2] < max_dist)

    pts_homo = intrins[:3, :3].to(device) @ verts
    pts_homo[:2] /= pts_homo[2:3]
    fov_mask = (pts_homo[2] > 0) & \
               (pts_homo[0] > 1) & (pts_homo[0] < img_width - 1) & \
               (pts_homo[1] > 1) & (pts_homo[1] < img_height - 1)

    mask = torch.logical_and(dist_mask, fov_mask).to(device)
    reward = torch.sum(mask, dtype=torch.float)
    return reward


class FrustumVisibilityEst(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                x, y, z,
                roll, pitch, yaw,
                verts,
                delta=0.05,  # small position and angular change for gradient estimation
                ):
        pose_reward = rewards_from_pose(x, y, z,
                                    roll, pitch, yaw,
                                    verts)

        # calculate how the small displacement dx=delta affects the amount of rewards, i.e. dr/dx = ?
        reward_dx = rewards_from_pose(x + delta, y, z,
                                       roll, pitch, yaw,
                                       verts) - pose_reward

        # calculate how the small displacement dy=delta affects the amount of rewards, i.e. dr/dy = ?
        reward_dy = rewards_from_pose(x, y + delta, z,
                                       roll, pitch, yaw,
                                       verts) - pose_reward

        # calculate how the small displacement dz=delta affects the amount of rewards, i.e. dr/dz = ?
        reward_dz = rewards_from_pose(x, y, z + delta,
                                       roll, pitch, yaw,
                                       verts) - pose_reward

        # calculate how the small rotation droll=delta affects the amount of rewards, i.e. dr/droll = ?
        reward_droll = rewards_from_pose(x, y, z,
                                          roll + delta, pitch, yaw,
                                          verts) - pose_reward

        # calculate how the small rotation dpitch=delta affects the amount of rewards, i.e. dr/dpitch = ?
        reward_dpitch = rewards_from_pose(x, y, z,
                                           roll, pitch + delta, yaw,
                                           verts) - pose_reward

        # calculate how the small rotation dyaw=delta affects the amount of rewards, i.e. dr/dyaw = ?
        reward_dyaw = rewards_from_pose(x, y, z,
                                         roll, pitch, yaw + delta,
                                         verts) - pose_reward

        ctx.save_for_backward(reward_dx, reward_dy, reward_dz,
                              reward_droll, reward_dpitch, reward_dyaw)
        return pose_reward

    @staticmethod
    def backward(ctx, grad_output):
        reward_dx, reward_dy, reward_dz,\
            reward_droll, reward_dpitch, reward_dyaw = ctx.saved_tensors

        device = reward_dx.device

        dx = (grad_output.clone() * reward_dx).to(device)
        dy = (grad_output.clone() * reward_dy).to(device)
        dz = (grad_output.clone() * reward_dz).to(device)

        droll = (grad_output.clone() * reward_droll).to(device)
        dpitch = (grad_output.clone() * reward_dpitch).to(device)
        dyaw = (grad_output.clone() * reward_dyaw).to(device)

        return dx, dy, dz, droll, dpitch, dyaw, None


class Model(nn.Module):
    def __init__(self,
                 points,
                 x, y, z,
                 roll, pitch, yaw,
                 min_dist=1.0, max_dist=5.0,
                 dist_rewards_mean=3.0, dist_rewards_sigma=2.0):
        super().__init__()
        self.points = points
        self.device = points.device
        self.rewards = None

        # Create optimizable parameters for pose of the camera.
        self.x = nn.Parameter(torch.as_tensor(x, dtype=torch.float32).to(self.device))
        self.y = nn.Parameter(torch.as_tensor(y, dtype=torch.float32).to(self.device))
        self.z = nn.Parameter(torch.as_tensor(z, dtype=torch.float32).to(self.device))
        self.roll = nn.Parameter(torch.as_tensor(roll, dtype=torch.float32).to(self.device))
        self.pitch = nn.Parameter(torch.as_tensor(pitch, dtype=torch.float32).to(self.device))
        self.yaw = nn.Parameter(torch.as_tensor(yaw, dtype=torch.float32).to(self.device))

        self.R = euler_angles_to_matrix(torch.tensor([self.roll,
                                                      self.pitch,
                                                      self.yaw]), "XYZ").unsqueeze(0).to(self.device)
        self.T = torch.tensor([self.x, self.y, self.z], device=self.device).unsqueeze(0)

        self.K, self.width, self.height = load_intrinsics()
        self.eps = 1e-6
        self.pc_clip_limits = torch.tensor([min_dist, max_dist])
        self.dist_rewards = {'mean': dist_rewards_mean, 'sigma': dist_rewards_sigma}

        self.frustum_visibility = FrustumVisibilityEst.apply

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
        R_inv = R.squeeze().T
        verts_cam = R_inv @ (verts - torch.repeat_interleave(T, len(verts), dim=0).to(self.device)).T
        verts_cam = verts_cam.T
        return verts_cam

    @staticmethod
    def gaussian(x, mu=3.0, sigma=5.0):
        # https://en.wikipedia.org/wiki/Normal_distribution
        g = torch.exp(-0.5 * ((x - mu) / sigma) ** 2)
        return g / (sigma * torch.sqrt(torch.tensor(2 * np.pi)))

    def distance_rewards(self, verts):
        # compute rewards based on distance of the surrounding points
        dists = torch.linalg.norm(verts, dim=1)
        rewards = self.gaussian(dists, mu=self.dist_rewards['mean'], sigma=self.dist_rewards['sigma'])
        return rewards

    def forward(self):
        pose_reward = self.frustum_visibility(self.x, self.y, self.z,
                                              self.roll, self.pitch, self.yaw,
                                              self.points)
        loss = 1. / (pose_reward + self.eps)

        self.R = euler_angles_to_matrix(torch.tensor([self.roll,
                                                      self.pitch,
                                                      self.yaw]), "XYZ").unsqueeze(0).to(self.device)
        self.T = torch.tensor([self.x, self.y, self.z], device=self.device).unsqueeze(0)
        verts = self.to_camera_frame(self.points, self.R, self.T)

        # calculate gaussian distance based rewards
        self.rewards = self.distance_rewards(verts)

        # get masks of points that are inside of the camera FOV
        dist_mask = self.get_dist_mask(verts.T, self.pc_clip_limits[0], self.pc_clip_limits[1])
        fov_mask = self.get_fov_mask(verts.T, self.height, self.width, self.K.squeeze(0))

        mask = torch.logical_and(dist_mask, fov_mask)

        # remove points that are outside of camera FOV
        verts = verts[mask, :]
        return verts, loss
