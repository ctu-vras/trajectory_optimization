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
from pytorch3d.transforms import matrix_to_quaternion, random_rotation, euler_angles_to_matrix
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


def load_intrinsics():
    width, height = 1232., 1616.
    K = torch.tensor([[758.03967, 0., 621.46572, 0.],
                      [0., 761.62359, 756.86402, 0.],
                      [0., 0., 1., 0.],
                      [0., 0., 0., 1.]]).to(device)
    K = K.unsqueeze(0)
    return K, width, height


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

    # HPR: remove occluded points (currently works only on CPU)
    if hpr:
        verts, occl_mask = hidden_pts_removal(verts.detach(), device=device)

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
        rewards = rewards_from_pose(x, y, z,
                                    roll, pitch, yaw,
                                    verts)

        # calculate how the small rotation ddist=delta affects the amount of rewards, i.e. dr/ddist = ?
        rewards_dx = rewards_from_pose(x + delta, y, z,
                                       roll, pitch, yaw,
                                       verts) - rewards

        # calculate how the small rotation delev=delta affects the amount of rewards, i.e. dr/delev = ?
        rewards_dy = rewards_from_pose(x, y + delta, z,
                                       roll, pitch, yaw,
                                       verts) - rewards

        # calculate how the small rotation dazim=delta affects the amount of rewards, i.e. dr/ddist = ?
        rewards_dz = rewards_from_pose(x, y, z + delta,
                                       roll, pitch, yaw,
                                       verts) - rewards

        # calculate how the small rotation delev=delta affects the amount of rewards, i.e. dr/delev = ?
        rewards_droll = rewards_from_pose(x, y, z,
                                       roll+delta, pitch, yaw,
                                       verts) - rewards

        # calculate how the small rotation dazim=delta affects the amount of rewards, i.e. dr/ddist = ?
        rewards_dpitch = rewards_from_pose(x, y, z,
                                       roll, pitch+delta, yaw,
                                       verts) - rewards

        # calculate how the small rotation dazim=delta affects the amount of rewards, i.e. dr/ddist = ?
        rewards_dyaw = rewards_from_pose(x, y, z,
                                           roll, pitch, yaw+delta,
                                           verts) - rewards

        ctx.save_for_backward(rewards_dx, rewards_dy, rewards_dz,
                              rewards_droll, rewards_dpitch, rewards_dyaw)
        return rewards

    @staticmethod
    def backward(ctx, grad_output):
        rewards_dx, rewards_dy, rewards_dz,\
            rewards_droll, rewards_dpitch, rewards_dyaw = ctx.saved_tensors

        device = rewards_dx.device

        dx = (grad_output.clone() * rewards_dx).to(device)
        dy = (grad_output.clone() * rewards_dy).to(device)
        dz = (grad_output.clone() * rewards_dz).to(device)

        droll = (grad_output.clone() * rewards_droll).to(device)
        dpitch = (grad_output.clone() * rewards_dpitch).to(device)
        dyaw = (grad_output.clone() * rewards_dyaw).to(device)

        return dx, dy, dz, droll, dpitch, dyaw, None


class Model(nn.Module):
    def __init__(self,
                 points,
                 x, y, z,
                 roll, pitch, yaw,
                 min_dist=1.0, max_dist=5.0):
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

    def forward(self):
        self.rewards = self.frustum_visibility(self.x, self.y, self.z,
                                               self.roll, self.pitch, self.yaw,
                                               self.points)
        loss = 1. / (self.rewards + self.eps)

        self.R = euler_angles_to_matrix(torch.tensor([self.roll,
                                                      self.pitch,
                                                      self.yaw]), "XYZ").unsqueeze(0).to(self.device)
        self.T = torch.tensor([self.x, self.y, self.z], device=self.device).unsqueeze(0)
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
    # Set paths to point cloud data
    # obj_filename = os.path.join(FE_PATH, "pts/cam_pts_camera_0_1607456676.1540315.npz")  # 2 separate parts
    # obj_filename = os.path.join(FE_PATH, "pts/cam_pts_camera_0_1607456663.5413494.npz")  # V-shaped
    # obj_filename = os.path.join(FE_PATH, "pts/", np.random.choice(os.listdir(os.path.join(FE_PATH, "pts/"))))
    obj_filename = os.path.join(FE_PATH, "src/traj_data/points/",
                                np.random.choice(os.listdir(os.path.join(FE_PATH, "src/traj_data/points/"))))
    pts_np = np.load(obj_filename)['pts']
    # make sure the point cloud is of (N x 3) shape:
    if pts_np.shape[1] > pts_np.shape[0]:
        pts_np = pts_np.transpose()
    points = torch.tensor(pts_np, dtype=torch.float32).to(device)

    # Initialize camera parameters
    K, img_width, img_height = load_intrinsics()

    # Initialize a model
    model = Model(points=points,
                  x=15.0, y=15.0, z=1.0,
                  roll=np.pi/2, pitch=np.pi/2, yaw=0.0,
                  min_dist=1.0, max_dist=5.0).to(device)
    # Create an optimizer. Here we are using Adam and we pass in the parameters of the model
    optimizer = torch.optim.Adam([
                {'params': list([model.x, model.y]), 'lr': 0.05},
                {'params': list([model.pitch]), 'lr': 0.02},
    ])

    # Run optimization loop
    for i in tqdm(range(800)):
        if rospy.is_shutdown():
            break
        optimizer.zero_grad()
        points_visible, loss = model()
        loss.backward()
        optimizer.step()

        # Visualization
        if i % 4 == 0:
            if points_visible.size()[0] > 0:
                image = render_pc_image(points_visible, K, img_height, img_width, device=device)

                image_vis = cv2.resize(image.detach().cpu().numpy(), (600, 800))
                publish_image(image_vis, topic='/pc_image')
                # cv2.imshow('Point cloud in camera FOV', image_vis)
                # cv2.waitKey(3)

            # print(f'Loss: {loss.item()}')
            # print(f'Number of visible points: {points_visible.size()[0]}')

            # publish ROS msgs
            # rewards_np = model.rewards.detach().cpu().numpy()
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
