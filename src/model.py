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
    assert isinstance(points, torch.Tensor)
    assert points.size()[1] == 3  # points.size() = N x 3
    # dist_mask = (points[:, 2] > min_dist) & (points[:, 2] < max_dist)

    # smooth and differentiable implementation of the above logic
    mean = torch.tensor((min_dist + max_dist) / 2.)
    std = torch.tensor(max_dist - min_dist)
    dist = torch.linalg.norm(points - mean, dim=1)
    dist_mask = torch.exp(-0.5 * (dist / std) ** 2)
    return dist_mask


def get_fov_mask(points, img_height, img_width, intrins):
    # find points that are observed by the camera (in its FOV)
    assert isinstance(points, torch.Tensor)
    assert points.size()[1] == 3  # points.size() = N x 3
    assert isinstance(intrins, torch.Tensor)
    assert intrins.size() == torch.Size([3, 3])

    # pts_homo = torch.matmul(intrins, torch.transpose(points, 0, 1))
    # pts_homo[:2] /= pts_homo[2]
    # fov_mask = (pts_homo[2] > 0) & \
    #            (pts_homo[0] > 1) & (pts_homo[0] < img_width - 1) & \
    #            (pts_homo[1] > 1) & (pts_homo[1] < img_height - 1)

    # smooth and differentiable implementation of the above logic
    pts_homo = torch.matmul(intrins, torch.transpose(points, 0, 1))
    depth_tanh = torch.tanh(pts_homo[2])
    width_gaussian = torch.exp(-0.5*((pts_homo[0]/pts_homo[2] - img_width/2) / img_width)**2)
    height_gaussian = torch.exp(-0.5 * ((pts_homo[1]/pts_homo[2] - img_height/2) / img_height) ** 2)

    fov_mask = depth_tanh * width_gaussian * height_gaussian
    return fov_mask


def visibility_estimation(points, img_height, img_width, intrins, pc_clip_limits, hpr=False):
    dist_mask = get_dist_mask(points, pc_clip_limits[0], pc_clip_limits[1])
    fov_mask = get_fov_mask(points, img_height, img_width, intrins)
    if hpr:
        # HPR: remove occluded points
        occlusion_mask = hidden_pts_removal(verts.detach(), device=self.device)[1]
        mask = occlusion_mask * (dist_mask * fov_mask)
    else:
        mask = dist_mask * fov_mask
    return mask


def to_camera_frame(verts, quat, trans):
    q = F.normalize(quat)
    q_inv = quaternion_invert(q)
    verts = verts - trans
    verts = quaternion_apply(q_inv, verts)
    return verts


"""
Model for a single pose optimization
"""


class ModelPose(nn.Module):
    def __init__(self,
                 points,
                 trans0,  # t = (x, y, z), example: torch.tensor([[0., 0., 0.]])
                 q0,  # q = (w, x, y, z), example: torch.tensor([[1., 0., 0., 0.]])
                 intrins,  # torch.tensor, size=(3, 3)
                 img_width, img_height,
                 min_dist=1.0, max_dist=5.0,
                 device=torch.device('cuda:0')):
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

        self.K = torch.as_tensor(intrins, dtype=torch.float32).to(self.device)
        self.img_width, self.img_height = float(img_width), float(img_height)
        self.eps = 1e-6
        self.pc_clip_limits = [min_dist, max_dist]  # [m]

        self.to(self.device)

    def forward(self, hpr=False, debug=False):
        # transform points to camera frame
        t0 = time()
        points = to_camera_frame(self.points, self.quat, self.trans)
        if debug:
            print(f'\nPoint cloud transformation took: {1000*(time() - t0)} msec')
            print(f'Point cloud size {self.points.size()}')

        # get masks of points that are inside of the camera FOV
        t1 = time()
        self.observations = visibility_estimation(points,
                                                  self.img_height, self.img_width,
                                                  self.K, self.pc_clip_limits)

        if debug:
            print(f'Visibility estimation took: {1000 * (time() - t1)} msec')

        t2 = time()
        loss = self.criterion(self.observations)
        if debug:
            print(f'Loss calculation took: {1000 * (time() - t2)} msec \n')
        return loss

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
        p1, p2, p3 = torch.as_tensor(traj_wps[i - 1]).squeeze(), \
                     torch.as_tensor(traj_wps[i]).squeeze(), \
                     torch.as_tensor(traj_wps[i + 1]).squeeze()

        AB = p1 - p2
        AC = p3 - p2

        phi = torch.arccos(torch.dot(AB, AC) / (torch.linalg.norm(AB) * torch.linalg.norm(AC) + eps))
        phi_mean += phi
    return phi_mean / (N_wps - 2)


class ModelTraj(nn.Module):
    def __init__(self,
                 points: torch.tensor,
                 wps_poses: torch.tensor,  # (N, 1, 3): [[[x0, y0, z0]], [[x1, y1, z1]], ...]
                 wps_quats: torch.tensor,  # (N,, 1, 4): torch.tensor: [w, x, y, z]-format
                 intrins: torch.tensor,  # torch.tensor, size=(3, 3)
                 img_width, img_height,
                 min_dist=1.0, max_dist=5.0,
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

        self.K = torch.as_tensor(intrins, dtype=torch.float32).to(self.device)
        self.img_width, self.img_height = float(img_width), float(img_height)
        self.eps = 1e-6
        self.pc_clip_limits = [min_dist, max_dist]  # [m]

        self.loss = {'vis': float('inf'),
                     'length': float('inf'),
                     'l2': float('inf'),
                     'smooth': float('inf')}
        self.smoothness_weight = smoothness_weight
        self.traj_length_weight = traj_length_weight

        self.to(self.device)

    def forward(self, debug=False):
        """
        Trajectory evaluation based on visibility estimation from its waypoints.
        traj_score = log_odds_sum([visibility_estimation(wp) for wp in traj_waypoints])
        """
        t0 = time()
        N_wps = len(self.poses)
        lo_sum = 0.0
        # TODO: replace for loop with tensors operations in order not to calculate rewards consequently for waypoints
        for i in range(N_wps):
            # transform points to camera frame
            points = to_camera_frame(self.points, self.quats[i].unsqueeze(0), self.poses[i].unsqueeze(0))

            dist_mask = get_dist_mask(points, self.pc_clip_limits[0], self.pc_clip_limits[1])
            # fov_mask = get_fov_mask(points, self.img_height, self.img_width, self.K)
            p = dist_mask #* fov_mask
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
        self.loss['vis'] = 1.0 / (torch.mean(rewards) + self.eps)

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
