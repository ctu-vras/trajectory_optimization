#!/usr/bin/env python

import ctypes
libgcc_s = ctypes.CDLL('libgcc_s.so.1')
import sys
import os
import rospkg
FE_PATH = rospkg.RosPack().get_path('trajectory_optimization')
sys.path.append(os.path.join(FE_PATH, 'src/'))
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.transforms import quaternion_invert, quaternion_apply, random_quaternions
import numpy as np
from model import ModelTraj

# ROS libraries
import rospy
import tf
from tools import load_intrinsics
from tools import publish_pointcloud
from tools import publish_path


if __name__ == "__main__":
    rospy.init_node('camera_traj_optimization')
    # Load the point cloud and initial trajectory to optimize
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    # Set paths
    index = 10
    # index = np.random.choice(range(0, 98))
    points_filename = os.path.join(FE_PATH, f"data/points/point_cloud_{index}.npz")
    pts_np = np.load(points_filename)['pts'].transpose()

    # make sure the point cloud is of (N x 3) shape:
    if pts_np.shape[1] > pts_np.shape[0]:
        pts_np = pts_np.transpose()
    points = torch.tensor(pts_np, dtype=torch.float32).to(device)
    # positions: (x, y, z) for each waypoint
    poses_filename = os.path.join(FE_PATH, f"data/paths/path_poses_{index}.npz")
    poses_0 = np.load(poses_filename)['poses'].tolist()
    # orientations: quaternion for each waypoint
    # quats_0 = torch.tensor([[1., 0., 0., 0.]], dtype=torch.float32)
    # xyzw = tf.transformations.quaternion_from_euler(np.pi / 2, np.pi / 4, 0.0)
    # quats_0 = torch.tensor([[xyzw[1], xyzw[2], xyzw[3], xyzw[0]]], dtype=torch.float32)
    # for i in range(len(poses_0) - 1):
    #     quats_0 = torch.cat([quats_0, torch.tensor([[xyzw[1], xyzw[2], xyzw[3], xyzw[0]]], dtype=torch.float32)])
    quats_0 = random_quaternions(len(poses_0))  # (N, 4)

    # Initialize a model
    model = ModelTraj(points=points,
                      wps_poses=poses_0,
                      wps_quats=quats_0).to(device)
    # Create an optimizer. Here we are using Adam and we pass in the parameters of the model
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    optimizer = torch.optim.Adam([
        {'params': list([model.poses]), 'lr': 0.1},
        {'params': list([model.quats]), 'lr': 0.5},
    ])

    # Run optimization loop
    FIRST_RECORD = True
    reward0 = None
    loop = tqdm(range(400))
    for i in loop:
        if rospy.is_shutdown():
            break
        optimizer.zero_grad()
        loss = model()
        loss.backward()
        optimizer.step()

        if i % 4 == 0:
            for key in model.loss:
                print(f"{key} loss: {model.loss[key]}")
            if FIRST_RECORD:
                reward0 = torch.sum(model.rewards)
                FIRST_RECORD = False
            print(f"Trajectory visibility score: {torch.sum(model.rewards)} / {reward0}")

            # publish ROS msgs
            intensity = model.rewards.detach().unsqueeze(1).cpu().numpy()
            # print(np.min(intensity), np.mean(intensity), np.max(intensity))
            points = np.concatenate([pts_np, intensity], axis=1)  # add observations for pts intensity visualization
            publish_pointcloud(points, '/pts', rospy.Time.now(), 'world')
            publish_path(poses_0, topic_name='/path/initial', frame_id='world')

            # publish path with positions and orientations
            poses_to_pub = model.poses.detach()
            quats_to_pub = [quat / torch.linalg.norm(quat) for quat in model.quats.detach()]
            publish_path(poses_to_pub, quats_to_pub, topic_name='/path/optimized', frame_id='world')
