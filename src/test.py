import torch
import numpy as np
from pytorch3d.transforms import quaternion_apply, quaternion_invert
from pyquaternion import Quaternion


def ego_to_cam(points, trans, pyquat):
    """Transform points (3 x N) from ego frame into a pinhole camera
    """
    points = points - trans
    rot = pyquat.rotation_matrix
    points = rot.T @ points
    return points.T


def ego_to_cam_torch(points, trans, quat):
    """Transform points (N x 3) from ego frame into a pinhole camera
    """
    points = points - trans
    quat_inv = quaternion_invert(quat)
    quaternion_apply(quat_inv, points)
    return points

points = torch.rand(size=(2, 3))
trans = torch.rand(size=(1, 3))
quat = torch.tensor([1., 0., 0., 0.])

print(ego_to_cam_torch(points, trans, quat))

print(ego_to_cam(points.cpu().numpy().T,
                 trans.cpu().numpy().T,
                 Quaternion(quat.cpu().numpy())))