import torch
import torch.nn as nn
import numpy as np
from pytorch3d.transforms import quaternion_invert, quaternion_apply
from tools import load_intrinsics, hidden_pts_removal
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self,
                 points,
                 trans0=torch.tensor([[0., 0., 0.]]),  # t = (x, y, z)
                 q0=torch.tensor([[1., 0., 0., 0.]]),  # q = (w, x, y, z)
                 min_dist=1.0, max_dist=10.0,
                 dist_rewards_mean=3.0, dist_rewards_sigma=2.0):
        super().__init__()
        self.points = points
        self.rewards = None
        self.observations = None
        self.device = points.device
        self.lo_sum = 0.0  # log odds sum for the entire point cloud for the whole trajectory

        # Create an optimizable parameter for the x, y, z position and quaternion orientation of the camera.
        trans = torch.as_tensor(trans0, dtype=torch.float32).to(self.device)
        self.trans = nn.Parameter(trans)
        quat = torch.as_tensor(q0, dtype=torch.float32).to(self.device)
        self.quat = nn.Parameter(quat)

        self.K, self.width, self.height = load_intrinsics(device=self.device)
        self.eps = 1e-6
        self.pc_clip_limits = [min_dist, max_dist]  # [m]
        self.dist_rewards = {'mean': dist_rewards_mean, 'dist_rewards_sigma': dist_rewards_sigma}

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

    def to_camera_frame(self, verts):
        q = F.normalize(self.quat)
        q_inv = quaternion_invert(q)
        verts = verts - self.trans
        verts = quaternion_apply(q_inv, verts)
        return verts

    @staticmethod
    def gaussian(x, mu=3.0, sigma=100.0, normalize=False):
        # https://en.wikipedia.org/wiki/Normal_distribution
        g = torch.exp(-0.5 * ((x - mu) / sigma) ** 2)
        if normalize:
            g /= (sigma * torch.sqrt(torch.tensor(2 * np.pi)))
        return g

    def visibility_estimation(self, verts, mask):
        # compute observations based on distance of the surrounding points
        dists = torch.linalg.norm(verts, dim=1)
        visibility = self.gaussian(dists, mu=self.dist_rewards['mean'],
                                   sigma=self.dist_rewards['dist_rewards_sigma'])
        return visibility * mask

    def log_odds_conversion(self, p):
        # apply log odds conversion for global voxel map observations update
        p = torch.clip(p, 0.5, 1 - self.eps)
        lo = torch.log(p / (1 - p))
        self.lo_sum += lo
        rewards = 1 / (1 + torch.exp(-self.lo_sum))
        return rewards

    def forward(self):
        # transform points to camera frame
        verts = self.to_camera_frame(self.points)

        # get masks of points that are inside of the camera FOV
        dist_mask = self.get_dist_mask(verts.T, self.pc_clip_limits[0], self.pc_clip_limits[1])
        fov_mask = self.get_fov_mask(verts.T, self.height, self.width, self.K.squeeze(0))

        # HPR: remove occluded points
        # occlusion_mask = hidden_pts_removal(points.detach(), device=self.device)[1]

        # mask = torch.logical_and(occlusion_mask, torch.logical_and(dist_mask, fov_mask))
        mask = torch.logical_and(dist_mask, fov_mask)

        self.observations = self.visibility_estimation(verts, mask)  # local observations reward (visibility)
        self.rewards = self.log_odds_conversion(self.observations)  # total trajectory observations
        loss = self.criterion(self.observations)
        return verts[mask, :], loss

    def criterion(self, observations):
        # transform observations to loss function
        loss = 1. / (torch.sum(observations) + self.eps)
        return loss
