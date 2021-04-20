import torch
import torch.nn as nn
import numpy as np
from pytorch3d.transforms import quaternion_invert, quaternion_apply
from tools import load_intrinsics, hidden_pts_removal
import torch.nn.functional as F
from copy import deepcopy
from time import time


# Helper functions
def get_dist_mask(points, min_dist=1.0, max_dist=5.0):
    # clip points between MIN_DIST and MAX_DIST meters distance from the camera
    # points.size() = N x 3
    dist_mask = (points[:, 2] > min_dist) & (points[:, 2] < max_dist)
    return dist_mask


def get_fov_mask(points, img_height, img_width, intrins):
    # find points that are observed by the camera (in its FOV)
    # points.size() = N x 3
    pts_homo = torch.matmul(intrins[:3, :3], torch.transpose(points, 0, 1))
    pts_homo[:2] /= pts_homo[2:3]
    fov_mask = (pts_homo[2] > 0) & (pts_homo[0] > 1) & \
               (pts_homo[0] < img_width - 1) & (pts_homo[1] > 1) & \
               (pts_homo[1] < img_height - 1)
    return fov_mask


def to_camera_frame(verts, quat, trans):
    q = F.normalize(quat)
    q_inv = quaternion_invert(q)
    verts = verts - trans
    verts = quaternion_apply(q_inv, verts)
    return verts


def gaussian(x, mu=3.0, sigma=100.0, normalize=False):
    # https://en.wikipedia.org/wiki/Normal_distribution
    g = torch.exp(-0.5 * ((x - mu) / sigma) ** 2)
    if normalize:
        g /= (sigma * torch.sqrt(torch.tensor(2 * np.pi)))
    return g


def visiblity_estimation(verts, mu, sigma, rewards_gain=50.0, mask=None):
    # compute observations based on distance of the surrounding points
    dists = torch.linalg.norm(verts, dim=1)
    rewards = gaussian(dists, mu, sigma)
    if mask is not None:
        # TODO: replace mask addition (+) with smth more sophisticated (Gaussian weights)
        # do not use multiplication (*) on binary mask as it zeros out the gradients
        rewards = rewards + rewards_gain * mask
    return rewards


"""
Model for a single pose optimization
"""


class ModelPose(nn.Module):
    def __init__(self,
                 points,
                 trans0=torch.tensor([[0., 0., 0.]]),  # t = (x, y, z)
                 q0=torch.tensor([[1., 0., 0., 0.]]),  # q = (w, x, y, z)
                 min_dist=1.0, max_dist=5.0,
                 dist_rewards_mean=3.0, dist_rewards_sigma=2.0,
                 device=torch.device('cuda')):
        super().__init__()
        self.device = device
        self.points = points.to(self.device)
        self.rewards = None
        self.observations = None
        self.lo_sum = 0.0  # log odds sum for the entire point cloud for the whole trajectory

        # Create an optimizable parameter for the x, y, z position and quaternion orientation of the camera.
        trans = torch.as_tensor(trans0, dtype=torch.float32).to(self.device)
        self.trans = nn.Parameter(trans)
        quat = torch.as_tensor(q0, dtype=torch.float32).to(self.device)
        self.quat = nn.Parameter(quat)

        self.K, self.width, self.height = load_intrinsics(device=self.device)
        self.eps = 1e-6
        self.pc_clip_limits = [min_dist, max_dist]  # [m]
        self.dist_rewards = {'mean': dist_rewards_mean, 'sigma': dist_rewards_sigma}

    def forward(self, hpr=False, debug=False):
        # transform points to camera frame
        t0 = time()
        verts = to_camera_frame(self.points, self.quat, self.trans)
        if debug:
            print(f'\nPoint cloud transformation took: {1000*(time() - t0)} msec')

        # get masks of points that are inside of the camera FOV
        t1 = time()
        dist_mask = get_dist_mask(verts, self.pc_clip_limits[0], self.pc_clip_limits[1])
        fov_mask = get_fov_mask(verts, self.height, self.width, self.K.squeeze(0))

        if hpr:
            # HPR: remove occluded points
            occlusion_mask = hidden_pts_removal(verts.detach(), device=self.device)[1]
            mask = torch.logical_and(occlusion_mask, torch.logical_and(dist_mask, fov_mask))
        else:
            mask = torch.logical_and(dist_mask, fov_mask)
        if debug:
            print(f'Masks computation took: {1000 * (time() - t1)} msec')

        t2 = time()
        self.observations = visiblity_estimation(verts,
                                                 self.dist_rewards['mean'],
                                                 self.dist_rewards['sigma'],
                                                 mask=mask)
        if debug:
            print(f'Visibility estimation took: {1000 * (time() - t2)} msec')

        t3 = time()
        loss = self.criterion(self.observations)
        if debug:
            print(f'Loss calculation took: {1000 * (time() - t3)} msec \n')
        return verts[mask, :], loss

    def criterion(self, observations):
        # transform observations to loss function
        loss = 1. / (torch.sum(observations) + self.eps)
        return loss


"""
Model for trajectory optimization
"""


def length_calc(traj: torch.tensor):
    l = 0.0
    for i in range(len(traj) - 1):
        l += torch.linalg.norm(traj[i + 1] - traj[i])
    return l


def mean_angle_calc(traj_wps, eps=1e-6):
    phi_mean = 0.0
    N_wps = len(traj_wps)
    for i in range(1, N_wps - 1):
        p1, p2, p3 = torch.as_tensor(traj_wps[i - 1]), torch.as_tensor(traj_wps[i]), torch.as_tensor(
            traj_wps[i + 1])

        AB = p1 - p2
        AC = p3 - p2

        phi = torch.arccos(torch.dot(AB, AC) / (torch.linalg.norm(AB) * torch.linalg.norm(AC) + eps))
        phi_mean += phi
    return phi_mean / (N_wps - 2)


class ModelTraj(nn.Module):
    def __init__(self,
                 points: torch.tensor,
                 wps_poses: torch.tensor,  # (N, 3): [[x0, y0, z0], [x1, y1, z1], ...]
                 wps_quats: torch.tensor,  # (N, 4): torch.tensor: [w, x, y, z]-format
                 min_dist=1.0, max_dist=5.0,
                 dist_rewards_mean=3.0, dist_rewards_sigma=2.0,
                 smoothness_weight=14.0, traj_length_weight=0.02,
                 device=torch.device('cuda')):
        super().__init__()
        self.device = device
        self.points = torch.as_tensor(points, dtype=torch.float32).to(self.device)
        self.rewards = None
        self.observations = None
        self.lo_sum = 0.0  # log odds sum for the entire point cloud for the whole trajectory

        # Create optimizable parameters for the poses and orients of the camera
        self.poses0 = torch.as_tensor(wps_poses, dtype=torch.float32).to(self.device)  # (N, 3)
        self.quats0 = torch.as_tensor(wps_quats, dtype=torch.float32).to(self.device)  # (N, 4)

        self.poses = nn.Parameter(deepcopy(self.poses0))
        self.quats = nn.Parameter(deepcopy(self.quats0))

        self.K, self.width, self.height = load_intrinsics(device=self.device)
        self.eps = 1e-6
        self.pc_clip_limits = [min_dist, max_dist]  # [m]
        self.dist_rewards = {'mean': dist_rewards_mean, 'sigma': dist_rewards_sigma}

        self.loss = {'vis': float('inf'),
                     'length': float('inf'),
                     'l2': float('inf'),
                     'smooth': float('inf')}
        self.smoothness_weight = smoothness_weight
        self.traj_length_weight = traj_length_weight

    def forward(self, hpr=False, debug=False):
        """
        Trajectory evaluation based on visibility estimation from its waypoints.
        traj_score = log_odds_sum([visibility_estimation(wp) for wp in traj_waypoints])
        """
        t0 = time()
        N_wps = len(self.poses)
        lo_sum = 0.0
        for i in range(N_wps):
            # transform points to camera frame
            verts = to_camera_frame(self.points, self.quats[i].unsqueeze(0), self.poses[i].unsqueeze(0))

            # get masks of points that are inside of the camera FOV
            dist_mask = get_dist_mask(verts, self.pc_clip_limits[0], self.pc_clip_limits[1])
            fov_mask = get_fov_mask(verts, self.height, self.width, self.K.squeeze(0))

            if hpr:
                # HPR: remove occluded points
                occlusion_mask = hidden_pts_removal(verts.detach(), device=self.device)[1]
                mask = torch.logical_and(occlusion_mask, torch.logical_and(dist_mask, fov_mask))
            else:
                mask = torch.logical_and(dist_mask, fov_mask)

            p = visiblity_estimation(verts,
                                     self.dist_rewards['mean'],
                                     self.dist_rewards['sigma'],
                                     mask=mask)  # local observations reward (visibility)

            # apply log odds conversion for global voxel map observations update
            p = torch.clip(p, 0.5, 1.0 - self.eps)
            lo = torch.log(p / (1.0 - p))
            lo_sum = lo_sum + lo
        if debug:
            print(f'Trajectory evaluation took {1000*(time() - t0)} msec')

        t1 = time()
        self.rewards = 1.0 / (1.0 + torch.exp(-lo_sum))  # total trajectory observations
        loss = self.criterion(self.rewards)
        if debug:
            print(f'Loss calculation took {1000*(time() - t1)} msec')
        return loss

    def criterion(self, rewards):
        # transform observations to loss function: loss = 1 / mean(prob(observed))
        self.loss['vis'] = 1.0 / torch.mean(rewards)

        # penalties for being far from initial waypoints
        self.loss['l2'] = torch.linalg.norm(self.poses[0] - self.poses0[0])
        # for i in range(1, len(self.traj)):
        #     self.loss['l2'] += 0.0003*torch.linalg.norm(self.traj[i] - self.traj0[i])

        # smoothness estimation based on average angles between waypoints:
        # the bigger the angle the better
        self.loss['smooth'] = self.smoothness_weight / (mean_angle_calc(self.poses, self.eps) + self.eps)

        # penalty for trajectory length (compared to initial one)
        self.loss['length'] = self.traj_length_weight * torch.abs(length_calc(self.poses) - length_calc(self.poses0))

        return self.loss['vis'] + self.loss['l2'] + self.loss['length'] + self.loss['smooth']
