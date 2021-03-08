#!/usr/bin/env python

import sys
import os
import rospkg
FE_PATH = rospkg.RosPack().get_path('frontier_exploration')
sys.path.append(os.path.join(FE_PATH, 'src/'))
import torch
from tqdm import tqdm
import numpy as np
import cv2
from pytorch3d.transforms import euler_angles_to_matrix
from tools import render_pc_image
from tools import hidden_pts_removal
from tools import load_intrinsics
import math

import rospy
import tf
from tools import publish_odom
from tools import publish_pointcloud
from tools import publish_tf_pose
from tools import publish_camera_info
from tools import publish_image
from tools import publish_path
from tools import to_pose_stamped
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped


def dwa_control(x, config, goal, points):
    """
    Dynamic Window Approach control
    """
    dw = calc_dynamic_window(x, config)

    u, trajectory = calc_control_and_trajectory(x, dw, config, goal, points)

    return u, trajectory


class Config:
    """
    simulation parameter class
    """

    def __init__(self):
        # robot parameter
        self.max_speed = 1.0  # [m/s]
        self.min_speed = -0.5  # [m/s]
        self.max_yaw_rate = 40.0 * math.pi / 180.0  # [rad/s]
        self.max_accel = 0.2  # [m/ss]
        self.max_delta_yaw_rate = 40.0 * math.pi / 180.0  # [rad/ss]
        self.v_resolution = 0.02  # [m/s]
        self.yaw_rate_resolution = 0.2 * math.pi / 180.0  # [rad/s]
        self.dt = 0.1  # [s] Time tick for motion prediction
        self.predict_time = 1.0  # [s]
        self.to_goal_cost_gain = 0.15
        self.speed_cost_gain = 1.0
        self.obstacle_cost_gain = 1.0
        self.visibility_reward_gain = 1e-4
        self.robot_stuck_flag_cons = 0.001  # constant to prevent robot stucked

        # Also used to check if goal is reached in both types
        self.robot_radius = 1.0  # [m] for collision check


def motion(x, u, dt):
    """
    motion model
    """

    x[2] += u[1] * dt
    x[0] += u[0] * math.cos(x[2]) * dt
    x[1] += u[0] * math.sin(x[2]) * dt
    x[3] = u[0]
    x[4] = u[1]

    return x


def calc_dynamic_window(x, config):
    """
    calculation dynamic window based on current state x
    """

    # Dynamic window from robot specification
    Vs = [config.min_speed, config.max_speed,
          -config.max_yaw_rate, config.max_yaw_rate]

    # Dynamic window from motion model
    Vd = [x[3] - config.max_accel * config.dt,
          x[3] + config.max_accel * config.dt,
          x[4] - config.max_delta_yaw_rate * config.dt,
          x[4] + config.max_delta_yaw_rate * config.dt]

    #  [v_min, v_max, yaw_rate_min, yaw_rate_max]
    dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
          max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]

    return dw


def predict_trajectory(x_init, v, y, config):
    """
    predict trajectory with an input
    """

    x = np.array(x_init)
    trajectory = np.array(x)
    time = 0
    while time <= config.predict_time:
        x = motion(x, [v, y], config.dt)
        trajectory = np.vstack((trajectory, x))
        time += config.dt

    return trajectory


def calc_control_and_trajectory(x, dw, config, goal, points):
    """
    calculation final input with dynamic window
    """

    x_init = x[:]
    min_cost = float("inf")
    best_u = [0.0, 0.0]
    best_trajectory = np.array([x])

    # evaluate all trajectory with sampled input in dynamic window
    for v in np.arange(dw[0], dw[1], config.v_resolution):
        for y in np.arange(dw[2], dw[3], config.yaw_rate_resolution):
            trajectory = predict_trajectory(x_init, v, y, config)

            # calc costs and rewards
            to_goal_cost = config.to_goal_cost_gain * calc_to_goal_cost(trajectory, goal)
            speed_cost = config.speed_cost_gain * (config.max_speed - trajectory[-1, 3])
            visibility_reward = config.visibility_reward_gain * calc_visibility_reward(trajectory, points)
            # print(to_goal_cost, speed_cost, visibility_reward)

            final_cost = to_goal_cost + speed_cost - visibility_reward

            # search minimum trajectory
            if min_cost >= final_cost:
                min_cost = final_cost
                best_u = [v, y]
                best_trajectory = trajectory
                if abs(best_u[0]) < config.robot_stuck_flag_cons \
                        and abs(x[3]) < config.robot_stuck_flag_cons:
                    # to ensure the robot do not get stuck in
                    # best v=0 m/s (in front of an obstacle) and
                    # best omega=0 rad/s (heading to the goal with
                    # angle difference of 0)
                    best_u[1] = -config.max_delta_yaw_rate
    return best_u, best_trajectory


def calc_to_goal_cost(trajectory, goal):
    """
        calc to goal cost with angle difference
    """

    dx = goal[0] - trajectory[-1, 0]
    dy = goal[1] - trajectory[-1, 1]
    error_angle = math.atan2(dy, dx)
    cost_angle = error_angle - trajectory[-1, 2]
    cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))

    return cost


def calc_visibility_from_pose(x, y, z,
                              roll, pitch, yaw,
                              verts,
                              min_dist=1.0, max_dist=5.0,
                              mu=3.0, sigma=2.0,
                              device=torch.device('cuda'),
                              hpr=False,  # whether to use hidden points removal algorithm
                              ):
    intrins = torch.as_tensor(K.squeeze(0), dtype=torch.float32).to(device)
    R = euler_angles_to_matrix(torch.tensor([roll, pitch, yaw], dtype=torch.float32), "XYZ").unsqueeze(0).to(device)
    T = torch.tensor([x, y, z], dtype=torch.float32, device=device).unsqueeze(0)

    # transform points to camera frame
    R_inv = torch.transpose(torch.squeeze(R, 0), 0, 1)
    verts = torch.transpose(verts - torch.repeat_interleave(T, len(verts), dim=0).to(device), 0, 1)
    verts = torch.matmul(R_inv, verts)

    # HPR: remove occluded points (currently works only on CPU)
    if hpr:
        verts, occl_mask = hidden_pts_removal(torch.transpose(verts, 0, 1).detach(), device=device)
        verts = torch.transpose(verts, 0, 1)

    # get masks of points that are inside of the camera FOV
    dist_mask = (verts[2] > min_dist) & (verts[2] < max_dist)

    pts_homo = torch.matmul(intrins[:3, :3], verts)
    pts_homo[:2] /= pts_homo[2:3]
    fov_mask = (pts_homo[2] > 0) & \
               (pts_homo[0] > 1) & (pts_homo[0] < img_width - 1) & \
               (pts_homo[1] > 1) & (pts_homo[1] < img_height - 1)

    mask = torch.logical_and(dist_mask, fov_mask).to(device)

    # compute visibility based on distance of the surrounding points
    dists = torch.linalg.norm(verts, dim=0)
    # https://en.wikipedia.org/wiki/Normal_distribution
    observations = torch.exp(-0.5 * ((dists - mu) / sigma) ** 2) * mask
    return observations


def calc_visibility_reward(trajectory, points, eps=1e-6):
    lo_sum = 0.0
    for x in trajectory:
        p = calc_visibility_from_pose(x=x[0], y=x[1], z=0.0,
                                      roll=0.0, pitch=0.0, yaw=x[2],
                                      verts=points)
        # apply log odds conversion for global voxel map observations update
        p = torch.clip(p, 0.5, 1 - eps)
        lo = torch.log(p / (1 - p))
        lo_sum += lo
    rewards = 1 / (1 + torch.exp(-lo_sum))
    return torch.sum(rewards, dtype=torch.float)


if __name__ == "__main__":
    rospy.init_node('camera_pose_optimization')
    config = Config()

    # Load point cloud
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    # Set paths to point cloud data
    index = 1612893730.3432848
    points_filename = os.path.join(FE_PATH, f"src/traj_data/points/point_cloud_{index}.npz")
    pts_np = np.load(points_filename)['pts']
    # make sure the point cloud is of (N x 3) shape:
    if pts_np.shape[1] > pts_np.shape[0]:
        pts_np = pts_np.transpose()
    points = torch.tensor(pts_np, dtype=torch.float32).to(device)

    # Initialize camera parameters
    K, img_width, img_height = load_intrinsics()

    # goal position
    gx, gy = 10.0, 10.0
    # initial state [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
    x = np.array([0.0, 0.0, math.pi / 8.0, 0.0, 0.0])
    # goal position [x(m), y(m)]
    goal = np.array([gx, gy])

    trajectory = np.array(x)
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        u, predicted_trajectory = dwa_control(x, config, goal, points)
        x = motion(x, u, config.dt)  # simulate robot
        trajectory = np.vstack((trajectory, x))  # store state history

        # check reaching goal
        dist_to_goal = math.hypot(x[0] - goal[0], x[1] - goal[1])
        if dist_to_goal <= config.robot_radius:
            print("Goal!!")
            break

        # Visualization: publish ROS msgs
        publish_pointcloud(pts_np, '/pts', rospy.Time.now(), 'world')
        trans = torch.tensor([x[0], x[1], 0.0])  # z = 0
        quat = tf.transformations.quaternion_from_euler(0, 0, x[2])
        publish_odom(trans, quat, frame='world', topic='/odom')
        publish_tf_pose(trans, quat, "camera_frame", frame_id="world")
        publish_camera_info(topic_name="/camera/camera_info", frame_id="camera_frame")
        rate.sleep()

    print("Done")
