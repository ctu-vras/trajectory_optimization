#!/usr/bin/env python

import sys
import os
FE_PATH = '/home/ruslan/Documents/CTU/catkin_ws/src/frontier_exploration/'
sys.path.append(os.path.join(FE_PATH, 'src/'))
import torch
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import cv2
from pytorch3d.renderer import look_at_view_transform, look_at_rotation
from pytorch3d.transforms import matrix_to_quaternion, random_rotation
from tools import render_pc_image
from tools import hidden_pts_removal

import rospy
from tools import publish_odom
from tools import publish_pointcloud
from tools import publish_tf_pose
from tools import publish_camera_info
from tools import publish_image
from tools import publish_path
from tools import to_pose_stamped
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped


def rewards_from_pose(camera_dist_elev_azim, verts,
                      min_dist=1.0, max_dist=10.0,
                      img_width=1232, img_height=1616,
                      intrins=torch.tensor([[758.03967, 0., 621.46572, 0.],
                                            [0., 761.62359, 756.86402, 0.],
                                            [0., 0., 1., 0.],
                                            [0., 0., 0., 1.]]),
                      device=torch.device('cuda')):
    dist, elev, azim = camera_dist_elev_azim
    R, T = look_at_view_transform(dist, elev, azim, device=device)

    # transform points to camera frame
    R_inv = torch.transpose(torch.squeeze(R, 0), 0, 1)
    verts = torch.transpose(verts - torch.repeat_interleave(T, len(verts), dim=0).to(device), 0, 1)
    verts = torch.matmul(R_inv, verts)

    # get masks of points that are inside of the camera FOV
    dist_mask = (verts[2] > min_dist) & (verts[2] < max_dist)

    pts_homo = intrins[:3, :3].to(device) @ verts
    pts_homo[:2] /= pts_homo[2:3]
    fov_mask = (pts_homo[2] > 0) & \
               (pts_homo[0] > 1) & (pts_homo[0] < img_width - 1) & \
               (pts_homo[1] > 1) & (pts_homo[1] < img_height - 1)

    # HPR: remove occluded points
    # verts, occl_mask = hidden_pts_removal(verts.detach(), device=self.device)

    mask = torch.logical_and(dist_mask, fov_mask).to(device)
    reward = torch.sum(mask, dtype=torch.float)
    return reward


class FrustumVisibilityEst(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx,
                camera_dist_elev_azim, verts,
                delta=0.05,  # small position and angular change for gradient estimation
                device=torch.device('cuda')):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        rewards = rewards_from_pose(camera_dist_elev_azim, verts)

        # calculate how the small rotation ddist=delta affects the amount of rewards, i.e. dr/ddist = ?
        camera_dist_elev_azim_ddist = camera_dist_elev_azim + torch.tensor([delta, 0.0, 0.0]).to(device)
        rewards_ddist = rewards_from_pose(camera_dist_elev_azim_ddist, verts) - rewards

        # calculate how the small rotation delev=delta affects the amount of rewards, i.e. dr/delev = ?
        camera_dist_elev_azim_delev = camera_dist_elev_azim + torch.tensor([0.0, delta, 0.0]).to(device)
        rewards_delev = rewards_from_pose(camera_dist_elev_azim_delev, verts) - rewards

        # calculate how the small rotation dazim=delta affects the amount of rewards, i.e. dr/ddist = ?
        camera_dist_elev_azim_dazim = camera_dist_elev_azim + torch.tensor([0.0, 0.0, delta]).to(device)
        rewards_dazim = rewards_from_pose(camera_dist_elev_azim_dazim, verts) - rewards

        ctx.save_for_backward(rewards_ddist, rewards_delev, rewards_dazim)
        return rewards

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        rewards_ddist, rewards_delev, rewards_dazim, = ctx.saved_tensors

        ddist = grad_output.clone() * rewards_ddist
        delev = grad_output.clone() * rewards_delev
        dazim = grad_output.clone() * rewards_dazim

        return torch.tensor([ddist, delev, dazim]).to(rewards_dazim.device), None


class Model(nn.Module):
    def __init__(self, points, dist, elev, azim, min_dist=1.0, max_dist=5.0):
        super().__init__()
        self.points = points
        self.device = points.device

        # Create optimizable parameters for pose of the camera.
        self.camera_dist_elev_azim = nn.Parameter(
            torch.as_tensor([dist, elev, azim], dtype=torch.float32).to(points.device))
        self.R, self.T = look_at_view_transform(dist, elev, azim, device=self.device)

        self.K, self.width, self.height = self.load_intrinsics()
        self.eps = 1e-6
        self.pc_clip_limits = torch.tensor([min_dist, max_dist])

        self.frustum_visibility = FrustumVisibilityEst.apply

    @staticmethod
    def load_intrinsics():
        width, height = 1232., 1616.
        K = torch.tensor([[758.03967, 0., 621.46572, 0.],
                          [0., 761.62359, 756.86402, 0.],
                          [0., 0., 1., 0.],
                          [0., 0., 0., 1.]]).to(device)
        K = K.unsqueeze(0)
        return K, width, height

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

    def forward(self):
        rewards = self.frustum_visibility(self.camera_dist_elev_azim, self.points)
        loss = 1. / (rewards + self.eps)

        dist, elev, azim = self.camera_dist_elev_azim
        self.R, self.T = look_at_view_transform(dist, elev, azim, device=self.device)
        verts = self.to_camera_frame(self.points, self.R, self.T)

        # get masks of points that are inside of the camera FOV
        dist_mask = self.get_dist_mask(verts.T, self.pc_clip_limits[0], self.pc_clip_limits[1])
        fov_mask = self.get_fov_mask(verts.T, self.height, self.width, self.K.squeeze(0))

        mask = torch.logical_and(dist_mask, fov_mask)

        # remove points that are outside of camera FOV
        verts = verts[mask, :]
        return verts, loss


if __name__ == "__main__":
    rospy.init_node('camera_pose_optimization')
    # Load point cloud
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    # Set paths
    # obj_filename = os.path.join(FE_PATH, "pts/cam_pts_camera_0_1607456676.1540315.npz")  # 2 separate parts
    obj_filename = os.path.join(FE_PATH, "pts/cam_pts_camera_0_1607456663.5413494.npz")  # V-shaped
    # obj_filename = os.path.join(FE_PATH, "pts/", np.random.choice(os.listdir(os.path.join(FE_PATH, "pts/"))))
    pts_np = np.load(obj_filename)['pts'].transpose()
    points = torch.tensor(pts_np).to(device)

    # Initialize camera parameters
    width, height = 1232, 1616
    K = torch.tensor([[758.03967, 0.,        621.46572, 0.],
                      [0.,        761.62359, 756.86402, 0.],
                      [0.,        0.,        1.,        0.],
                      [0.,        0.,        0.,        1.]]).to(device)
    K = K.unsqueeze(0)
    R = torch.eye(3).unsqueeze(0).to(device)
    T = torch.Tensor([[0., 0., 0.]]).to(device)

    # Initialize a model
    model = Model(points=points,
                  # x0=1.0, y0=1.0, z0=-1.0,
                  dist=-4.0, elev=-30.0, azim=20.0,
                  min_dist=1.0, max_dist=5.0).to(device)
    # Create an optimizer. Here we are using Adam and we pass in the parameters of the model
    optimizer = torch.optim.Adam(model.parameters(), lr=0.3)

    # Run optimization loop
    for i in tqdm(range(1000)):
        if rospy.is_shutdown():
            break
        optimizer.zero_grad()
        points_visible, loss = model()
        loss.backward()
        optimizer.step()

        # Visualization
        if i % 4 == 0:
            if points_visible.size()[0] > 0:
                image = render_pc_image(points_visible, R, T, K, height, width, device)

                image_vis = cv2.resize(image.detach().cpu().numpy(), (600, 800))
                publish_image(image_vis, topic='/pc_image')
                # cv2.imshow('Point cloud in camera FOV', image_vis)
                # cv2.waitKey(3)

            # print(f'Loss: {loss.item()}')
            # print(f'Number of visible points: {points_visible.size()[0]}')

            # publish ROS msgs
            # rewards_np = model.rewards.detach().unsqueeze(1).cpu().numpy()
            # points = np.concatenate([pts_np, rewards_np], axis=1)  # add rewards for pts intensity visualization
            points_visible_np = points_visible.detach().cpu().numpy()
            publish_pointcloud(points_visible_np, '/pts_visible', rospy.Time.now(), 'camera_frame')
            publish_pointcloud(pts_np, '/pts', rospy.Time.now(), 'world')
            quat = matrix_to_quaternion(model.R).squeeze()
            quat = (quat[1], quat[2], quat[3], quat[0])
            trans = model.T.squeeze()
            publish_odom(trans, quat, frame='world', topic='/odom')
            publish_tf_pose(trans, quat, "camera_frame", frame_id="world")
            publish_camera_info(topic_name="/camera/camera_info", frame_id="camera_frame")
