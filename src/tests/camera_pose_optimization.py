import sys
sys.path.append('../')
import torch
from tqdm.notebook import tqdm
import torch.nn as nn
import numpy as np
import cv2

# Data structures and functions for rendering
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    look_at_view_transform,
    PointsRasterizationSettings,
    PointsRenderer,
    PulsarPointsRenderer,
    PointsRasterizer,
    PerspectiveCameras
)
from pytorch3d.loss import chamfer_distance
from pytorch3d.transforms import matrix_to_quaternion

from tools import get_cam_frustum_pts
from tools import hidden_pts_removal
from tools import render_pc_image
from tools import publish_odom
from tools import denormalize
import rospy


class Model(nn.Module):
    def __init__(self,
                 points,
                 dist_init,
                 elev_init,
                 azim_init,
                 K,
                 height, width):
        super().__init__()
        self.device = torch.device('cuda')
        self.points = torch.tensor(points, requires_grad=True).to(self.device)

        # Create an optimizable parameter for distance, elevation, azimuth of the camera.
        self.camera_dist_elev_azim = nn.Parameter(
            torch.as_tensor([dist_init, elev_init, azim_init], dtype=torch.float32).to(self.device))

        self.K = K.to(self.device)
        self.width = torch.tensor([width]).to(self.device)
        self.height = torch.tensor([height]).to(self.device)
        self.eps = torch.tensor([1.0e-6]).to(self.device)
        self.min_dist = torch.tensor([1.0]).to(self.device)
        self.max_dist = torch.tensor([10.0]).to(self.device)

    def to_camera_frame(self, verts, R, T):
        R_inv = R.squeeze().T
        verts_cam = R_inv @ (verts - torch.repeat_interleave(T, len(verts), dim=0).to(self.device)).T
        verts_cam = verts_cam.T
        return verts_cam

    def pc_visibility_estimation(self, verts):
        # remove pts that are outside of the camera FOV
        verts = get_cam_frustum_pts(verts.T,
                                        self.height, self.width,
                                        self.K.squeeze(0),
                                        min_dist=self.min_dist, max_dist=self.max_dist).T
        # verts = hidden_pts_removal(verts.detach(), device=self.device)
        return verts

    def forward(self):
        # Based on the new position of the camera we calculate the rotation and translation matrices
        distance, elevation, azimuth = self.camera_dist_elev_azim
        R, T = look_at_view_transform(distance, elevation, azimuth, device=device)

        # transform points to camera frame
        verts = self.to_camera_frame(self.points, R, T)

        # point cloud visibility estimation
        verts = self.pc_visibility_estimation(verts)
        loss = self.criterion(torch.ones_like(verts, requires_grad=True))
        return verts, loss

    def criterion(self, rewards):
        # Calculate the loss based on the number of visible points in cloud
        loss = 1. / (torch.sum(rewards) + self.eps)
        return loss


if __name__ == "__main__":
    rospy.init_node('cam_pose_opt')
    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # Load reference point cloud
    obj_filename = "../../../../../catkin_ws/src/frontier_exploration/pts/cam_pts_camera_0_1607456676.1540315.npz"
    # obj_filename = "../../../../../catkin_ws/src/frontier_exploration/pts/cam_pts_camera_0_1607456663.5413494.npz"
    pts_np = np.load(obj_filename)['pts'].transpose()
    verts = torch.tensor(pts_np).to(device)
    rgb = torch.zeros_like(verts)
    point_cloud_ref = Pointclouds(points=[verts], features=[rgb])

    # Initialize reference camera extrinsics
    R = torch.eye(3).unsqueeze(0).to(device)
    T = torch.Tensor([[0., 0., 0.]]).to(device)

    # Camera intrinsics
    width, height = 1232., 1616.
    K = torch.tensor([[758.03967, 0.,        621.46572, 0.],
                      [0.,        761.62359, 756.86402, 0.],
                      [0.,        0.,        1.,        0.],
                      [0.,        0.,        0.,        1.]]).to(device)
    K = K.unsqueeze(0)
    # Render reference point cloud on an image plane
    image_ref = render_pc_image(verts, R, T, K, height, width, device).cpu().numpy()

    # Initialize a model: and define starting position of the camera (not at the origin)
    # with its distance in meters, elevation and azimuth angles in degrees
    model = Model(points=verts,
                  dist_init=-5.0,  # [m]
                  elev_init=40,  # [deg]
                  azim_init=20,  # [deg]
                  K=K, height=height, width=width).to(device)
    # Create an optimizer. Here we are using Adam and we pass in the parameters of the model
    optimizer = torch.optim.Adam(model.parameters(), lr=0.04)

    # Run optimization loop
    video_writer = None
    for i in tqdm(range(1000)):
        optimizer.zero_grad()
        verts, loss = model()
        point_cloud = Pointclouds(points=[verts], features=[torch.zeros_like(verts)])
        # loss_icp, _ = chamfer_distance(point_cloud, point_cloud_ref)
        # loss_icp.backward()
        loss.backward()
        optimizer.step()

        if rospy.is_shutdown():
            break
        # ROS msgs publishers
        # distance, elevation, azimuth = model.camera_dist_elev_azim
        # rot, tran = look_at_view_transform(distance, elevation, azimuth, device=device)
        # quat = matrix_to_quaternion(rot).squeeze(0)  # [w, x, y, z]
        # publish_odom(tran.squeeze(0), [quat[1], quat[2], quat[3], quat[0]])

        if i % 10 == 0:
            image = render_pc_image(verts, R, T, K, height, width, device)
            # image_visible = render_pc_image(verts_visible, R, T, K, height, width, device)
            # print(f'Loss: {loss.item()}')

            image = cv2.resize(image.detach().cpu().numpy()[..., :3], (512, 512))
            # image_visible = cv2.resize(image_visible.detach().cpu().numpy()[..., :3], (512, 512))
            image_ref = cv2.resize(image_ref[..., :3], (512, 512))
            frame = np.concatenate([image, image_ref], axis=1)
            # frame = np.concatenate([image, image_visible, image_ref], axis=1)
            # cv2.putText(frame, f'Number of visible points: {verts_visible.size()[0]}/{verts.size()[0]}',
            #             (256, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_4)
            cv2.imshow('Current view / Visible points / Target view', frame)
            cv2.waitKey(3)

            # # write video
            # if video_writer is None:
            #     fourcc = cv2.VideoWriter_fourcc(*'XVID')
            #     video_writer = cv2.VideoWriter('./output.mp4',
            #                                    fourcc, 10,
            #                                    (frame.shape[1], frame.shape[0]))
            # video_writer.write(np.asarray(255 * denormalize(frame), dtype=np.uint8))
