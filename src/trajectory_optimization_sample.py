#!/usr/bin/env python

import ctypes
# libgcc_s = ctypes.CDLL('libgcc_s.so.1')
import sys
import os
import rospkg
FE_PATH = rospkg.RosPack().get_path('trajectory_optimization')
sys.path.append(os.path.join(FE_PATH, 'src/'))
import torch
from tqdm import tqdm
from pytorch3d.transforms import random_quaternions
import numpy as np
import matplotlib.pyplot as plt
from model import ModelTraj
from time import time
# ROS libraries
import rospy
import tf
from tools import publish_pointcloud
from tools import publish_path
from tools import load_intrinsics


def quat_wxyz_to_xyzw(quat):
    return torch.tensor([quat[1], quat[2], quat[3], quat[0]], device=quat.device)


def load_data(index=None):
    # Set paths
    if index is None:
        index = np.random.choice(range(0, 98))
    print(f"Sequence number: {index}")
    points_filename = os.path.join(FE_PATH, f"data/points/point_cloud_{index}.npz")
    pts_np = np.load(points_filename)['pts'].transpose()

    # make sure the point cloud is of (N x 3) shape:
    if pts_np.shape[1] > pts_np.shape[0]:
        pts_np = pts_np.transpose()

    # positions: (x, y, z) for each waypoint
    poses_filename = os.path.join(FE_PATH, f"data/paths/path_poses_{index}.npz")
    poses_np = np.load(poses_filename)['poses']

    # orientations: quaternion for each waypoint
    xyzw = tf.transformations.quaternion_from_euler(0.0, 0.0, 0.0)
    quats_wxyz_np = np.asarray([[xyzw[3], xyzw[0], xyzw[1], xyzw[2]]], dtype=np.float32)
    for _ in range(len(poses_np) - 1):
        quats_wxyz_np = np.vstack([quats_wxyz_np, np.asarray([[xyzw[3], xyzw[0], xyzw[1], xyzw[2]]], dtype=np.float32)])
    return pts_np, poses_np, quats_wxyz_np


## Get parameters values
pub_sample = rospy.get_param('traj_opt/pub_sample', 10)
N_steps = rospy.get_param('traj_opt/opt_steps', 400)
smooth_weight = rospy.get_param('traj_opt/smooth_weight', 14.0)
length_weight = rospy.get_param('traj_opt/length_weight', 0.02)
lr_pose = rospy.get_param('traj_opt/lr_pose', 0.1)
lr_quat = rospy.get_param('traj_opt/lr_quat', 0.0)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

if __name__ == "__main__":
    rospy.init_node('camera_traj_optimization')

    # Load the point cloud and initial trajectory to optimize
    index = 3  # np.random.choice(range(0, 15))  # 0-98 or None - for random
    pts_np, poses_np, quats_wxyz_np = load_data(index=index)

    points = torch.tensor(pts_np, dtype=torch.float32).to(device)
    poses_0 = torch.from_numpy(poses_np).float().to(device)
    quats_wxyz_0 = torch.from_numpy(quats_wxyz_np).float().to(device)

    K, img_width, img_height = load_intrinsics(device=device)

    ## Initialize a model
    model = ModelTraj(points=points,
                      wps_poses=poses_0,
                      wps_quats=quats_wxyz_0,
                      intrins=K,
                      img_width=img_width, img_height=img_height,
                      smoothness_weight=smooth_weight, traj_length_weight=length_weight).to(device)

    # Create an optimizer. Here we are using Adam and pass in the parameters of the model
    decayRate = 0.9  # decayRate = 0.9, lr_pose = 0.15
    optimizer = torch.optim.Adam([
        {'params': list([model.poses]), 'lr': lr_pose},
        {'params': list([model.quats]), 'lr': lr_quat},
    ])
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)

    ## Run optimization loop
    FIRST_RECORD = True
    reward0 = None
    smooth_loss0 = None
    OPTIMIZATION_COMPLETE = False
    N_optimal = N_steps
    REWARDS_TH = 1.01
    SMOOTHNESS_TH = 0.9

    t_step = 0.0
    t_pub = 0.0
    debug = True
    fig = plt.figure(figsize=(16, 8))
    plt.grid()
    log = {'visibility': [],
           'smoothness': []}
    for i in tqdm(range(N_steps)):
        if rospy.is_shutdown():
            plt.close('all')
            break
        # Optimization step
        t0 = time()
        optimizer.zero_grad()
        loss = model(debug=debug)
        t1 = time()
        loss.backward()
        optimizer.step()
        if i % int(N_steps // 10) == 0:
            lr_scheduler.step()

        debug = False
        t_step += (time() - t0) / N_steps

        ## Data publishing, debugging and visualization
        if i % pub_sample == 0:
            t2 = time()
            # debug = True
            # if debug:
            #     print(f'Gradient backprop took: {1000 * (time() - t1)} msec')
            # for key in model.loss:
            #     print(f"{key} loss: {model.loss[key]}")
            if FIRST_RECORD:
                reward0 = torch.mean(model.rewards)
                smooth_loss0 = model.loss['smooth']
                FIRST_RECORD = False
            # print(f"Trajectory visibility score: {torch.mean(model.rewards) / reward0}")

            log['visibility'].append(torch.mean(model.rewards) / reward0)
            log['smoothness'].append(smooth_loss0 / model.loss['smooth'])
            # plt.cla()
            # plt.subplot(1,2,1)
            # # plt.grid()
            # plt.title('Visibility reward gain: R / R0')
            # plt.ylabel('R / R0')
            # plt.xlabel('opt steps')
            # plt.plot(log['visibility'], color='b')
            # if OPTIMIZATION_COMPLETE:
            #     plt.axvline(N_optimal, 0, 1)
            #
            # plt.subplot(1,2,2)
            # # plt.grid()
            # plt.title('Trajectory smoothness')
            # plt.ylabel('Loss_{smooth}0 / Loss_{smooth}')
            # plt.xlabel('opt steps')
            # plt.plot(log['smoothness'], color='b')
            # if OPTIMIZATION_COMPLETE:
            #     plt.axvline(N_optimal, 0, 1)
            # plt.pause(0.01)
            # plt.draw()

            if not OPTIMIZATION_COMPLETE and \
                   log['visibility'][-1] > REWARDS_TH and \
                   log['smoothness'][-1] > SMOOTHNESS_TH:
                OPTIMIZATION_COMPLETE = True
                N_optimal = i
                print(f'Found optimal trajectory after {N_optimal} steps')

            # publish ROS msgs
            intensity = model.rewards.detach().unsqueeze(1).cpu().numpy()
            # print(np.min(intensity), np.mean(intensity), np.max(intensity))
            points = np.concatenate([pts_np, intensity], axis=1)  # add observations for pts intensity visualization
            publish_pointcloud(points, '/pts', rospy.Time.now(), 'world')
            quats_to_pub_0 = [quat_wxyz_to_xyzw(quat / torch.linalg.norm(quat)) for quat in quats_wxyz_0]
            publish_path(poses_np.tolist(), quats_to_pub_0, topic_name='/path/initial', frame_id='world')

            # publish path with positions and orientations
            poses_to_pub = model.poses.detach()
            quats_to_pub = [quat_wxyz_to_xyzw(quat / torch.linalg.norm(quat)) for quat in model.quats.detach()]
            publish_path(poses_to_pub, quats_to_pub, topic_name='/path/optimized', frame_id='world')

            t_pub += (time() - t2) / N_steps * pub_sample

    print(f'Mean optimization step time: {1000 * t_step} msec')
    print(f'Mean publication time: {1000 * t_pub} msec')
