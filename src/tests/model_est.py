import sys
import os
import rospkg
FE_PATH = rospkg.RosPack().get_path('trajectory_optimization')
sys.path.append(os.path.join(FE_PATH, 'src/'))
import torch
import torch.nn as nn
import numpy as np
from pytorch3d.transforms import euler_angles_to_matrix
from tools import load_intrinsics, hidden_pts_removal


def observations_from_pose(x, y, z,
                           roll, pitch, yaw,
                           verts,
                           min_dist=1.0, max_dist=5.0,
                           mu=3.0, sigma=2.0,
                           device=torch.device('cuda'),
                           hpr=False,  # whether to use hidden points removal algorithm
                           ):
    K, img_width, img_height = load_intrinsics(device=device)

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

    # compute visibility based on distance of the surrounding points
    dists = torch.linalg.norm(verts, dim=0)
    # https://en.wikipedia.org/wiki/Normal_distribution
    observations = torch.exp(-0.5 * ((dists - mu) / sigma) ** 2) * mask
    return torch.sum(observations, dtype=torch.float)


class NumericalFrustumVisibility(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                x, y, z,
                roll, pitch, yaw,
                verts,
                cfg,
                device=torch.device('cuda'),
                ):
        observations = observations_from_pose(x, y, z,
                                             roll, pitch, yaw,
                                             verts,
                                             min_dist=cfg['frustum_min_dist'], max_dist=cfg['frustum_max_dist'],
                                             mu=cfg['dist_rewards_mean'], sigma=cfg['dist_rewards_sigma'],
                                             device=device
                                              )

        # calculate how the small displacement dx=delta affects the amount of observations, i.e. dr/dx = ?
        observations_dx = observations_from_pose(x + cfg['delta'], y, z,
                                           roll, pitch, yaw,
                                           verts,
                                           min_dist=cfg['frustum_min_dist'], max_dist=cfg['frustum_max_dist'],
                                           mu=cfg['dist_rewards_mean'], sigma=cfg['dist_rewards_sigma'],
                                           device=device) - observations

        # calculate how the small displacement dy=delta affects the amount of observations, i.e. dr/dy = ?
        observations_dy = observations_from_pose(x, y + cfg['delta'], z,
                                           roll, pitch, yaw,
                                           verts,
                                           min_dist=cfg['frustum_min_dist'], max_dist=cfg['frustum_max_dist'],
                                           mu=cfg['dist_rewards_mean'], sigma=cfg['dist_rewards_sigma'],
                                           device=device) - observations

        # calculate how the small displacement dz=delta affects the amount of observations, i.e. dr/dz = ?
        observations_dz = observations_from_pose(x, y, z + cfg['delta'],
                                           roll, pitch, yaw,
                                           verts,
                                           min_dist=cfg['frustum_min_dist'], max_dist=cfg['frustum_max_dist'],
                                           mu=cfg['dist_rewards_mean'], sigma=cfg['dist_rewards_sigma'],
                                           device=device) - observations

        # calculate how the small rotation droll=delta affects the amount of observations, i.e. dr/droll = ?
        observations_droll = observations_from_pose(x, y, z,
                                              roll + cfg['delta'], pitch, yaw,
                                              verts,
                                              min_dist=cfg['frustum_min_dist'], max_dist=cfg['frustum_max_dist'],
                                              mu=cfg['dist_rewards_mean'], sigma=cfg['dist_rewards_sigma'],
                                              device=device) - observations

        # calculate how the small rotation dpitch=delta affects the amount of observations, i.e. dr/dpitch = ?
        observations_dpitch = observations_from_pose(x, y, z,
                                               roll, pitch + cfg['delta'], yaw,
                                               verts,
                                               min_dist=cfg['frustum_min_dist'], max_dist=cfg['frustum_max_dist'],
                                               mu=cfg['dist_rewards_mean'], sigma=cfg['dist_rewards_sigma'],
                                               device=device) - observations

        # calculate how the small rotation dyaw=delta affects the amount of observations, i.e. dr/dyaw = ?
        observations_dyaw = observations_from_pose(x, y, z,
                                             roll, pitch, yaw + cfg['delta'],
                                             verts,
                                             min_dist=cfg['frustum_min_dist'], max_dist=cfg['frustum_max_dist'],
                                             mu=cfg['dist_rewards_mean'], sigma=cfg['dist_rewards_sigma'],
                                             device=device) - observations

        ctx.save_for_backward(observations_dx, observations_dy, observations_dz,
                              observations_droll, observations_dpitch, observations_dyaw)
        return observations

    @staticmethod
    def backward(ctx, grad_output):
        observations_dx, observations_dy, observations_dz,\
            observations_droll, observations_dpitch, observations_dyaw = ctx.saved_tensors

        device = observations_dx.device

        dx = (grad_output.clone() * observations_dx).to(device)
        dy = (grad_output.clone() * observations_dy).to(device)
        dz = (grad_output.clone() * observations_dz).to(device)

        droll = (grad_output.clone() * observations_droll).to(device)
        dpitch = (grad_output.clone() * observations_dpitch).to(device)
        dyaw = (grad_output.clone() * observations_dyaw).to(device)

        return dx, dy, dz, droll, dpitch, dyaw, None, None, None


class Model(nn.Module):
    def __init__(self,
                 points,
                 x, y, z,
                 roll, pitch, yaw,
                 cfg,
                 ):
        super().__init__()
        self.points = points
        self.device = points.device
        self.rewards = None
        self.observations = None
        self.lo_sum = 0.0  # log odds sum for the entire point cloud for the whole trajectory
        self.cfg = cfg

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

        self.K, self.width, self.height = load_intrinsics(device=self.device)

        self.frustum_visibility = NumericalFrustumVisibility.apply

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
    def gaussian(x, mu=3.0, sigma=5.0, normalize=False):
        # https://en.wikipedia.org/wiki/Normal_distribution
        g = torch.exp(-0.5 * ((x - mu) / sigma) ** 2)
        if normalize:
            g /= (sigma * torch.sqrt(torch.tensor(2 * np.pi)))
        return g

    def visibility_estimation(self, verts, mask):
        # compute visibility based on distance of the surrounding points
        dists = torch.linalg.norm(verts, dim=1)
        observations = self.gaussian(dists, mu=self.cfg['dist_rewards_mean'], sigma=self.cfg['dist_rewards_sigma'])
        return observations * mask

    def log_odds_conversion(self, p):
        # apply log odds conversion for global voxel map observations update
        p = torch.clip(p, 0.5, 1 - self.cfg['eps'])
        lo = torch.log(p / (1 - p))
        self.lo_sum += lo
        rewards = 1 / (1 + torch.exp(-self.lo_sum))
        return rewards

    def forward(self):
        pose_reward = self.frustum_visibility(self.x, self.y, self.z,
                                              self.roll, self.pitch, self.yaw,
                                              self.points,
                                              self.cfg,
                                              self.device)
        loss = 1. / (pose_reward + self.cfg['eps'])

        self.R = euler_angles_to_matrix(torch.tensor([self.roll,
                                                      self.pitch,
                                                      self.yaw]), "XYZ").unsqueeze(0).to(self.device)
        self.T = torch.tensor([self.x, self.y, self.z], device=self.device).unsqueeze(0)
        verts = self.to_camera_frame(self.points, self.R, self.T)

        # get masks of points that are inside of the camera FOV
        dist_mask = self.get_dist_mask(verts.T, self.cfg['frustum_min_dist'], self.cfg['frustum_max_dist'])
        fov_mask = self.get_fov_mask(verts.T, self.height, self.width, self.K.squeeze(0))

        mask = torch.logical_and(dist_mask, fov_mask)

        # calculate gaussian distance based observations
        self.observations = self.visibility_estimation(verts, mask)
        self.rewards = self.log_odds_conversion(self.observations)

        # remove points that are outside of camera FOV
        verts = verts[mask, :]
        # loss = 1. / (torch.sum(self.observations) + self.cfg['eps'])
        return verts, loss
