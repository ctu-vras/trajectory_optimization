#!/usr/bin/env python

import os
import torch
import numpy as np
from scipy.spatial import ConvexHull
from tools import render_pc_image
from tools import hidden_pts_removal
from tools import get_cam_frustum_pts
import cv2
import time

# Data structures and functions for rendering
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PulsarPointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor,
    PerspectiveCameras
)


# Render the point cloud from different views
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

R = torch.eye(3, dtype=torch.float32, device=device)[None, ...]
T = torch.zeros((1, 3), dtype=torch.float32, device=device)

cam_info_K = (758.03967, 0.0, 621.46572, 0.0, 761.62359, 756.86402, 0.0, 0.0, 1.0)

K = torch.zeros([4, 4]).to(device)
K[0][0] = cam_info_K[0]
K[0][2] = cam_info_K[2]
K[1][1] = cam_info_K[4]
K[1][2] = cam_info_K[5]
K[2][2] = 1
K[3][3] = 1
K = K.unsqueeze(0)

width, height = 1232, 1616

# Load point cloud
obj_filename = "../../../../catkin_ws/src/frontier_exploration/pts/cam_pts_camera_0_1607456676.1540315.npz"
# path = "../../../../catkin_ws/src/frontier_exploration/pts/"
# np.random.seed(0)
# obj_filename = os.path.join(path, np.random.choice(os.listdir(path)))
pts_np = np.load(obj_filename)['pts'].transpose()

verts = torch.Tensor(pts_np).to(device)        
rgb = (verts - torch.min(verts)) / torch.max(verts - torch.min(verts)).to(device)

point_cloud = Pointclouds(points=[verts], features=[rgb])


# for elevation, azimuth in zip(np.linspace(-30, 30, 50), np.linspace(-30, 30, 50)):
for elevation in np.linspace(-60, 60, 50):
# for distance in [-10, -8, -6, -4, -2, -1]:
# for delta in np.linspace(-2, 2, 50):

    # Select the viewpoint using spherical angles
    # elevation = 0.0  # angle of elevation in degrees
    distance = -0.1  # distance from camera to the object
    azimuth = 0.0  # No rotation so the camera is positioned on the +Z axis.

    # Get the position of the camera based on the spherical angles
    R, T = look_at_view_transform(distance, elevation, azimuth, device=device)
    # T[0][0] += delta

    verts = torch.from_numpy(pts_np).to(device)

    # transform points to camera frame
    R_inv = R.squeeze().T
    verts_cam = R_inv @ (verts - torch.repeat_interleave(T, len(verts), dim=0).to(device)).T
    verts_cam = verts_cam.T

    verts_cam = get_cam_frustum_pts(verts_cam.T,
                                    height, width,
                                    K.squeeze(0),
                                    min_dist=1.0, max_dist=10.0).T

    verts_cam_visible = hidden_pts_removal(verts_cam, device=device)
    print(f'Number of visible points: {verts_cam_visible.size()[0]}/{verts.size()[0]}')

    R = torch.eye(3, dtype=torch.float32, device=device)[None, ...]
    T = torch.zeros((1, 3), dtype=torch.float32, device=device)
    pc_image = render_pc_image(verts_cam, R, T, K, width=width, height=height, device=device)
    pc_visible_image = render_pc_image(verts_cam_visible, R, T, K, width=width, height=height, device=device)

    image = cv2.resize(pc_image.cpu().numpy(), (width // 2, height // 2))
    image = cv2.flip(image, -1)

    image_vis = cv2.resize(pc_visible_image.cpu().numpy(), (width // 2, height // 2))
    image_vis = cv2.flip(image_vis, -1)

    # cv2.imshow('Initial point cloud', image_vis)
    # cv2.imshow('After HPR', image_vis)
    cv2.imshow('Result', np.concatenate([image, image_vis], axis=1))
    cv2.waitKey(3)

    time.sleep(0.1)
