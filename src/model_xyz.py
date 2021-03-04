import torch
import torch.nn as nn
import numpy as np
from pytorch3d.transforms import euler_angles_to_matrix
from tools import load_intrinsics, hidden_pts_removal


class FrustumVisibility(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rewards, fov_mask):
        rewards_fov = rewards * fov_mask

        ctx.save_for_backward(fov_mask)
        return torch.sum(rewards_fov)

    @staticmethod
    def backward(ctx, grad_output):
        fov_mask, = ctx.saved_tensors
        d_rewards = grad_output.clone() * fov_mask
        return d_rewards, None


class Model(nn.Module):
    def __init__(self,
                 points,
                 x0=0.0, y0=0.0, z0=0.0,
                 min_dist=1.0, max_dist=10.0,
                 dist_rewards_mean=3.0, dist_rewards_sigma=2.0):
        super().__init__()
        self.points = points
        self.rewards = None
        self.observations = None
        self.device = points.device
        self.lo_sum = 0.0  # log odds sum for the entire point cloud for the whole trajectory

        # Create an optimizable parameter for the x, y, z position of the camera.
        self.camera_position = torch.from_numpy(np.array([x0, y0, z0], dtype=np.float32)).to(self.device)
        # Based on the new position of the
        # camera we calculate the rotation and translation matrices
        # Create optimizable parameters for pose of the camera.
        # R = look_at_rotation(self.camera_position[None, :], device=self.device)  # (1, 3, 3)
        # T = -torch.bmm(R.transpose(1, 2), self.camera_position[None, :, None])[:, :, 0]  # (1, 3)
        R = torch.eye(3, device=self.device).unsqueeze(0)
        T = -self.camera_position.unsqueeze(0)

        # TODO: include yaw rotation as an optimizable model parameter
        self.T = nn.Parameter(T)
        self.R = nn.Parameter(R)

        self.K, self.width, self.height = load_intrinsics()
        self.eps = 1e-6
        self.pc_clip_limits = [min_dist, max_dist]  # [m]
        self.dist_rewards = {'mean': dist_rewards_mean, 'sigma': dist_rewards_sigma}

        self.frustum_visibility = FrustumVisibility.apply

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

    def distance_visibility(self, verts):
        # compute rewards based on distance of the surrounding points
        dists = torch.linalg.norm(self.T - verts, dim=1)
        rewards = self.gaussian(dists, mu=self.dist_rewards['mean'], sigma=self.dist_rewards['sigma'])
        return rewards

    def log_odds_conversion(self, p):
        # apply log odds conversion for global voxel map rewards update
        p = torch.clip(p, 0.5, 1 - self.eps)
        lo = torch.log(p / (1 - p))
        self.lo_sum += lo
        rewards = 1 / (1 + torch.exp(-self.lo_sum))
        return rewards

    def forward(self):
        # transform points to camera frame
        verts = self.to_camera_frame(self.points, self.R, self.T)

        # get masks of points that are inside of the camera FOV
        dist_mask = self.get_dist_mask(verts.T, self.pc_clip_limits[0], self.pc_clip_limits[1])
        fov_mask = self.get_fov_mask(verts.T, self.height, self.width, self.K.squeeze(0))

        # HPR: remove occluded points
        # occlusion_mask = hidden_pts_removal(verts.detach(), device=self.device)[1]

        # mask = torch.logical_and(occlusion_mask, torch.logical_and(dist_mask, fov_mask))
        mask = torch.logical_and(dist_mask, fov_mask)
        # mask = torch.logical_and(occlusion_mask, dist_mask)

        # remove points that are outside of camera FOV
        verts = verts[mask, :]

        self.observations = self.distance_visibility(self.points)
        self.rewards = self.log_odds_conversion(self.observations)
        loss = self.criterion(self.rewards, mask.to(self.device))
        return verts, loss

    def criterion(self, rewards, mask):
        # transform rewards to loss function
        # loss = 1. / (torch.sum(rewards) + self.eps)
        loss = 1. / (self.frustum_visibility(rewards, mask) + self.eps)
        return loss
