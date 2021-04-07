#!/usr/bin/env python

import sys
import os
import rospkg
FE_PATH = rospkg.RosPack().get_path('trajectory_optimization')
sys.path.append(os.path.join(FE_PATH, 'src/'))
import torch
from tqdm import tqdm
import numpy as np
import cv2
from pytorch3d.transforms import matrix_to_quaternion, random_rotation
from tools import render_pc_image
from tools import hidden_pts_removal
from tools import load_intrinsics
from model import Model

import rospy
from tools import publish_odom
from tools import publish_pointcloud
from tools import publish_tf_pose
from tools import publish_camera_info
from tools import publish_image


if __name__ == "__main__":
    rospy.init_node('camera_pose_optimization')
    # Load point cloud
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # Initialize camera parameters
    K, img_width, img_height = load_intrinsics(device=device)

    # Set paths to data
    index = 2
    # index = np.random.choice(range(0, 98))
    points_filename = os.path.join(FE_PATH, f"data/points/point_cloud_{index}.npz")
    pts_np = np.load(points_filename)['pts']
    # make sure the point cloud is of (N x 3) shape:
    if pts_np.shape[1] > pts_np.shape[0]:
        pts_np = pts_np.transpose()
    points = torch.tensor(pts_np, dtype=torch.float32).to(device)

    # Initial position to optimize
    x0, y0, z0 = torch.tensor([9.0, 2.0, 0.0], dtype=torch.float)
    roll0, pitch0, yaw0 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float)

    # Initialize a model
    model = Model(points=points,
                  x0=x0, y0=y0, z0=z0,
                  roll0=roll0, pitch0=pitch0, yaw0=yaw0,
                  min_dist=1.0, max_dist=5.0).to(device)
    # Create an optimizer. Here we are using Adam and we pass in the parameters of the model
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    # Run optimization loop
    for i in tqdm(range(200)):
        if rospy.is_shutdown():
            break
        optimizer.zero_grad()
        points_visible, loss = model()
        loss.backward(retain_graph=True)
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
            intensity = model.observations.unsqueeze(1).detach().cpu().numpy()
            # intensity = model.reward.unsqueeze(1).detach().cpu().numpy()
            pts_rewards = np.concatenate([pts_np, intensity],
                                         axis=1)  # add observations for pts intensity visualization
            points_visible_np = points_visible.detach().cpu().numpy()
            publish_pointcloud(points_visible_np, '/pts_visible', rospy.Time.now(), 'camera_frame')
            publish_pointcloud(pts_rewards, '/pts', rospy.Time.now(), 'world')
            quat = matrix_to_quaternion(model.R).squeeze()
            quat = (quat[1], quat[2], quat[3], quat[0])
            trans = model.T.squeeze()
            publish_odom(trans, quat, frame='world', topic='/odom')
            publish_tf_pose(trans, quat, "camera_frame", frame_id="world")
            publish_camera_info(topic_name="/camera/camera_info", frame_id="camera_frame")