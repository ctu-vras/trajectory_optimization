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
from tools import render_pc_image
from tools import hidden_pts_removal
from tools import load_intrinsics
from model import ModelPose
import torch.nn.functional as F
from pytorch3d.transforms import random_quaternions
from time import time
import rospy
import tf
from tools import publish_odom
from tools import publish_pointcloud
from tools import publish_tf_pose
from tools import publish_camera_info
from tools import publish_image


## Get parameters values
pub_sample = rospy.get_param('traj_opt/pub_sample', 10)
N_steps = rospy.get_param('traj_opt/opt_steps', 400)
lr_pose = rospy.get_param('traj_opt/lr_pose', 0.1)
lr_quat = rospy.get_param('traj_opt/lr_quat', 0.0)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


if __name__ == "__main__":
    rospy.init_node('camera_pose_optimization')
    # Load point cloud

    # Initialize camera parameters
    K, img_width, img_height = load_intrinsics(device=device)

    # Set paths to data
    index = 90
    # index = np.random.choice(range(0, 98))
    points_filename = os.path.join(FE_PATH, f"data/points/point_cloud_{index}.npz")
    pts_np = np.load(points_filename)['pts']
    # make sure the point cloud is of (N x 3) shape:
    if pts_np.shape[1] > pts_np.shape[0]:
        pts_np = pts_np.transpose()
    points = torch.tensor(pts_np, dtype=torch.float32).to(device)

    # Initial position to optimize
    trans0 = torch.tensor([[9.0, 2.0, 0.0]], dtype=torch.float32)

    # xyzw = torch.tensor([0., 0., 0., 1.], dtype=torch.float32)
    xyzw = tf.transformations.quaternion_from_euler(0.0, np.pi/2, np.pi/3)
    q0 = torch.tensor([[xyzw[3], xyzw[0], xyzw[1], xyzw[2]]], dtype=torch.float32)
    # q0 = random_quaternions(1)

    # Initialize a model
    model = ModelPose(points=points,
                      trans0=trans0,
                      q0=q0,
                      min_dist=1.0, max_dist=5.0,
                      device=device).to(device)

    # Create an optimizer. Here we are using Adam and we pass in the parameters of the model
    optimizer = torch.optim.Adam([
        {'params': list([model.trans]), 'lr': lr_pose},
        {'params': list([model.quat]), 'lr': lr_quat},
    ])

    # Run optimization loop
    debug = False
    t_step = 0.0
    t_pub = 0.0
    for i in tqdm(range(N_steps)):
        if rospy.is_shutdown():
            break

        ## Optimization step
        t0 = time()
        optimizer.zero_grad()
        points_visible, loss = model(debug=debug)
        loss.backward()
        optimizer.step()

        t_step += (time() - t0) / N_steps

        ## Data publication
        debug = False
        if i % pub_sample == 0:
            t2 = time()
            debug = True

            # render point cloud image
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
            pts_rewards = np.concatenate([pts_np, intensity],
                                         axis=1)  # add observations for pts intensity visualization
            # pts_rewards = pts_np
            points_visible_np = points_visible.detach().cpu().numpy()
            publish_pointcloud(points_visible_np, '/pts_visible', rospy.Time.now(), 'camera_frame')
            publish_pointcloud(pts_rewards, '/pts', rospy.Time.now(), 'world')
            quat = F.normalize(model.quat).squeeze()
            quat = (quat[1], quat[2], quat[3], quat[0])
            trans = model.trans.squeeze()
            publish_odom(trans, quat, frame='world', topic='/odom')
            publish_tf_pose(trans, quat, "camera_frame", frame_id="world")
            publish_camera_info(topic_name="/camera/camera_info", frame_id="camera_frame")

            t_pub += (time() - t2) / N_steps * pub_sample

    print(f'Mean optimization time: {1000 * t_step} msec')
    print(f'Mean publication time: {1000 * t_pub} msec')
