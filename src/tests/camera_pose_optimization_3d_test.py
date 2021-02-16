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
from tools import load_intrinsics
from model import Model


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


if __name__ == "__main__":
    rospy.init_node('camera_pose_optimization')
    # Load point cloud
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    # Set paths to point cloud data
    index = 1612893730.3432848
    points_filename = os.path.join(FE_PATH, f"src/traj_data/points/point_cloud_{index}.npz")
    # points_filename = os.path.join(FE_PATH, "src/traj_data/points/",
    #                                np.random.choice(os.listdir(os.path.join(FE_PATH, "src/traj_data/points/"))))
    pts_np = np.load(points_filename)['pts']
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
                  min_dist=1.0, max_dist=5.0,
                  dist_rewards_mean=3.0, dist_rewards_sigma=2.0).to(device)
    # Create an optimizer. Here we are using Adam and we pass in the parameters of the model
    optimizer = torch.optim.Adam([
                {'params': list([model.x, model.y]), 'lr': 0.05},
                {'params': list([model.pitch]), 'lr': 0.04},
    ])

    # Run optimization loop
    for i in tqdm(range(500)):
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
            rewards_np = model.rewards.unsqueeze(1).detach().cpu().numpy()
            pts_rewards = np.concatenate([pts_np, rewards_np], axis=1)  # add rewards for pts intensity visualization
            points_visible_np = points_visible.detach().cpu().numpy()
            publish_pointcloud(points_visible_np, '/pts_visible', rospy.Time.now(), 'camera_frame')
            publish_pointcloud(pts_rewards, '/pts', rospy.Time.now(), 'world')
            quat = matrix_to_quaternion(model.R).squeeze()
            quat = (quat[1], quat[2], quat[3], quat[0])
            trans = model.T.squeeze()
            publish_odom(trans, quat, frame='world', topic='/odom')
            publish_tf_pose(trans, quat, "camera_frame", frame_id="world")
            publish_camera_info(topic_name="/camera/camera_info", frame_id="camera_frame")
