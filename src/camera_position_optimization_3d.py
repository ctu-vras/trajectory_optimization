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


def load_intrinsics():
    width, height = 1232., 1616.
    K = torch.tensor([[758.03967, 0., 621.46572, 0.],
                      [0., 761.62359, 756.86402, 0.],
                      [0., 0., 1., 0.],
                      [0., 0., 0., 1.]]).to(device)
    K = K.unsqueeze(0)
    return K, width, height


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
                 min_dist=1.0, max_dist=10.0):
        super().__init__()
        self.points = points
        self.rewards = None
        self.device = points.device

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
    def gaussian(x, mu=3.0, sigma=100.0):
        # https://en.wikipedia.org/wiki/Normal_distribution
        g = torch.exp(-0.5 * ((x - mu) / sigma) ** 2)
        return g / (sigma * torch.sqrt(torch.tensor(2 * np.pi)))

    def distance_rewards(self, verts):
        # compute rewards based on distance of the surrounding points
        dists = torch.linalg.norm(self.T - verts, dim=1)
        rewards = self.gaussian(dists)
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

        self.rewards = self.distance_rewards(self.points)
        loss = self.criterion(self.rewards, mask.to(self.device))
        return verts, loss
        
    def criterion(self, rewards, mask):
        # transform rewards to loss function
        # loss = 1. / (torch.sum(rewards) + self.eps)
        loss = 1. / (self.frustum_visibility(rewards, mask) + self.eps)
        return loss


if __name__ == "__main__":
    rospy.init_node('camera_pose_optimization')
    # Load point cloud
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # Initialize camera parameters
    K, width, height = load_intrinsics()
    R = torch.eye(3).unsqueeze(0).to(device)
    T = torch.tensor([[0., 0., 0.]]).to(device)

    # Set paths
    # points_filename = os.path.join(FE_PATH, "pts/cam_pts_camera_0_1607456676.1540315.npz")  # 2 separate parts
    points_filename = os.path.join(FE_PATH, "pts/cam_pts_camera_0_1607456663.5413494.npz")  # V-shaped
    # points_filename = os.path.join(FE_PATH, "pts/", np.random.choice(os.listdir(os.path.join(FE_PATH, "pts/"))))
    pts_np = np.load(points_filename)['pts']
    # make sure the point cloud is of (N x 3) shape:
    if pts_np.shape[1] > pts_np.shape[0]:
        pts_np = pts_np.transpose()
    points = torch.tensor(pts_np, dtype=torch.float32).to(device)

    # Initial position to optimize
    x0, y0, z0 = torch.zeros(3, dtype=torch.float)

    # Initialize a model
    model = Model(points=points,
                  x0=x0, y0=y0, z0=z0,
                  min_dist=1.0, max_dist=5.0).to(device)
    # Create an optimizer. Here we are using Adam and we pass in the parameters of the model
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    # Run optimization loop
    for i in tqdm(range(300)):
        if rospy.is_shutdown():
            break
        optimizer.zero_grad()
        points_visible, loss = model()
        loss.backward()
        optimizer.step()

        # Visualization
        if i % 2 == 0:
            if points_visible.size()[0] > 0:
                image = render_pc_image(points_visible, K, height, width, R, T, device)

                image_vis = cv2.resize(image.detach().cpu().numpy(), (600, 800))
                publish_image(image_vis, topic='/pc_image')
                # cv2.imshow('Point cloud in camera FOV', image_vis)
                # cv2.waitKey(3)

            # print(f'Loss: {loss.item()}')
            # print(f'Number of visible points: {points_visible.size()[0]}')

            # publish ROS msgs
            rewards_np = model.rewards.detach().unsqueeze(1).cpu().numpy()
            points = np.concatenate([pts_np, rewards_np], axis=1)  # add rewards for pts intensity visualization
            points_visible_np = points_visible.detach().cpu().numpy()
            publish_pointcloud(points_visible_np, '/pts_visible', rospy.Time.now(), 'camera_frame')
            publish_pointcloud(points, '/pts', rospy.Time.now(), 'world')
            quat = matrix_to_quaternion(model.R).squeeze()
            quat = (quat[1], quat[2], quat[3], quat[0])
            trans = model.T.squeeze()
            publish_odom(trans, quat, frame='world', topic='/odom')
            publish_tf_pose(trans, quat, "camera_frame", frame_id="world")
            publish_camera_info(topic_name="/camera/camera_info", frame_id="camera_frame")
