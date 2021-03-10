#!/usr/bin/env python

import os
import rospkg
FE_PATH = rospkg.RosPack().get_path('frontier_exploration')
import torch
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

        # Visibility parameters
        self.frustum_min_dist = 1.0
        self.frustum_max_dist = 5.0
        self.dist_rewards_mean = 3.0
        self.dist_rewards_sigma = 2.0
        self.eps = 1e-6


class Planner:
    def __init__(self, cfg, pts_np):
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")
        self.pts_np = pts_np
        self.points = torch.tensor(pts_np, dtype=torch.float32).to(self.device)
        self.cfg = cfg
        # Initialize camera parameters
        self.K, self.img_width, self.img_height = load_intrinsics()
        self.rewards = None
        self.observations = None
        self.points_visible = None
        self.lo_sum = 0.0

    def dwa_control(self, x, goal, points):
        """
        Dynamic Window Approach control
        """
        dw = self.calc_dynamic_window(x)

        u, trajectory = self.calc_control_and_trajectory(x, dw, goal, points)

        return u, trajectory

    def motion(self, x, u, dt):
        """
        motion model
        """
        x[2] += u[1] * dt
        x[0] += u[0] * math.cos(x[2]) * dt
        x[1] += u[0] * math.sin(x[2]) * dt
        x[3] = u[0]
        x[4] = u[1]

        return x

    def calc_dynamic_window(self, x):
        """
        calculation dynamic window based on current state x
        """
        # Dynamic window from robot specification
        Vs = [self.cfg.min_speed, self.cfg.max_speed,
              -self.cfg.max_yaw_rate, self.cfg.max_yaw_rate]

        # Dynamic window from motion model
        Vd = [x[3] - self.cfg.max_accel * self.cfg.dt,
              x[3] + self.cfg.max_accel * self.cfg.dt,
              x[4] - self.cfg.max_delta_yaw_rate * self.cfg.dt,
              x[4] + self.cfg.max_delta_yaw_rate * self.cfg.dt]

        #  [v_min, v_max, yaw_rate_min, yaw_rate_max]
        dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
              max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]

        return dw

    def predict_trajectory(self, x_init, v, y):
        """
        predict trajectory with an input
        """
        x = np.array(x_init)
        trajectory = np.array(x)
        time = 0
        while time <= self.cfg.predict_time:
            x = self.motion(x, [v, y], self.cfg.dt)
            trajectory = np.vstack((trajectory, x))
            time += self.cfg.dt

        return trajectory

    def calc_control_and_trajectory(self, x, dw, goal, points):
        """
        calculation final input with dynamic window
        """

        x_init = x[:]
        max_reward = -float("inf")
        best_u = [0.0, 0.0]
        best_trajectory = np.array([x])

        # evaluate all trajectory with sampled input in dynamic window
        for v in np.arange(dw[0], dw[1], self.cfg.v_resolution):
            for y in np.arange(dw[2], dw[3], self.cfg.yaw_rate_resolution):
                trajectory = self.predict_trajectory(x_init, v, y)

                # calc costs and rewards
                to_goal_cost = self.cfg.to_goal_cost_gain * self.calc_to_goal_cost(trajectory, goal)
                speed_cost = self.cfg.speed_cost_gain * (self.cfg.max_speed - trajectory[-1, 3])
                visibility_reward = self.cfg.visibility_reward_gain * self.calc_visibility_reward(trajectory, points)
                # print(to_goal_cost, speed_cost, visibility_reward)

                final_reward = visibility_reward - to_goal_cost - speed_cost

                # search maximum reward trajectory
                if max_reward <= final_reward:
                    max_reward = final_reward
                    best_u = [v, y]
                    best_trajectory = trajectory
                    if abs(best_u[0]) < self.cfg.robot_stuck_flag_cons \
                            and abs(x[3]) < self.cfg.robot_stuck_flag_cons:
                        # to ensure the robot do not get stuck in
                        # best v=0 m/s (in front of an obstacle) and
                        # best omega=0 rad/s (heading to the goal with
                        # angle difference of 0)
                        best_u[1] = -self.cfg.max_delta_yaw_rate
        return best_u, best_trajectory

    def calc_to_goal_cost(self, trajectory, goal):
        """
            calc to goal cost with angle difference
        """

        dx = goal[0] - trajectory[-1, 0]
        dy = goal[1] - trajectory[-1, 1]
        error_angle = math.atan2(dy, dx)
        cost_angle = error_angle - trajectory[-1, 2]
        cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))

        return cost

    def calc_visibility_from_pose(self, x, y, z,
                                  roll, pitch, yaw,
                                  points,
                                  hpr=False,  # whether to use hidden points removal algorithm
                                  ):
        intrins = torch.as_tensor(self.K.squeeze(0), dtype=torch.float32).to(self.device)
        R = euler_angles_to_matrix(torch.tensor([roll, pitch, yaw], dtype=torch.float32), "XYZ").unsqueeze(0).to(self.device)
        T = torch.tensor([x, y, z], dtype=torch.float32, device=self.device).unsqueeze(0)

        # transform points to camera frame
        R_inv = torch.transpose(torch.squeeze(R, 0), 0, 1)
        points = torch.transpose(points - torch.repeat_interleave(T, len(points), dim=0).to(self.device), 0, 1)
        points = torch.matmul(R_inv, points)

        # HPR: remove occluded points (currently works only on CPU)
        if hpr:
            points, occl_mask = hidden_pts_removal(torch.transpose(points, 0, 1).detach(), device=self.device)
            points = torch.transpose(points, 0, 1)

        # get masks of points that are inside of the camera FOV
        dist_mask = (points[2] > self.cfg.frustum_min_dist) & (points[2] < self.cfg.frustum_max_dist)

        pts_homo = torch.matmul(intrins[:3, :3], points)
        pts_homo[:2] /= pts_homo[2:3]
        fov_mask = (pts_homo[2] > 0) & \
                   (pts_homo[0] > 1) & (pts_homo[0] < self.img_width - 1) & \
                   (pts_homo[1] > 1) & (pts_homo[1] < self.img_height - 1)

        mask = torch.logical_and(dist_mask, fov_mask).to(self.device)
        self.points_visible = torch.transpose(points, 0, 1)[mask, :]

        # compute visibility based on distance of the surrounding points
        dists = torch.linalg.norm(points, dim=0)
        # https://en.wikipedia.org/wiki/Normal_distribution
        self.observations = torch.exp(-0.5 * ((dists - self.cfg.dist_rewards_mean) / self.cfg.dist_rewards_sigma) ** 2)
        self.observations = self.observations * mask
        return self.observations

    def calc_visibility_reward(self, trajectory, points):
        lo_sum = 0.0
        for x in trajectory:
            p = self.calc_visibility_from_pose(x=x[0], y=x[1], z=0.0,
                                               roll=0.0, pitch=0.0, yaw=x[2],
                                               points=points)
            # apply log odds conversion for global voxel map observations update
            p = torch.clip(p, 0.5, 1 - self.cfg.eps)
            lo = torch.log(p / (1 - p))
            lo_sum += lo
            self.lo_sum += lo
        local_traj_rewards = 1 / (1 + torch.exp(-lo_sum))
        self.rewards = 1 / (1 + torch.exp(-self.lo_sum))
        return torch.sum(local_traj_rewards, dtype=torch.float)

    def data_publisher(self, x):
        if self.points_visible.size()[0] > 0:
            image = render_pc_image(self.points_visible, self.K, self.img_height, self.img_width, device=self.device)

            image_vis = cv2.resize(image.detach().cpu().numpy(), (600, 800))
            publish_image(image_vis, topic='/pc_image')
            # cv2.imshow('Point cloud in camera FOV', image_vis)
            # cv2.waitKey(3)
        rewards_np = self.rewards.unsqueeze(1).detach().cpu().numpy()
        pts_rewards = np.concatenate([self.pts_np, rewards_np],
                                     axis=1)  # add observations for pts intensity visualization
        publish_pointcloud(pts_rewards, '/pts', rospy.Time.now(), 'world')
        trans = torch.tensor([x[0], x[1], 0.0])  # z = 0
        quat = tf.transformations.quaternion_from_euler(0, 0, x[2])
        publish_odom(trans, quat, frame='world', topic='/odom')
        publish_tf_pose(trans, quat, "camera_frame", frame_id="world")
        publish_camera_info(topic_name="/camera/camera_info", frame_id="camera_frame")

    def plan(self, x, goal):
        trajectory = np.array(x)
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            u, predicted_trajectory = self.dwa_control(x, goal, self.points)
            x = self.motion(x, u, self.cfg.dt)  # simulate robot
            trajectory = np.vstack((trajectory, x))  # store state history

            # check reaching goal
            dist_to_goal = math.hypot(x[0] - goal[0], x[1] - goal[1])
            if dist_to_goal <= self.cfg.robot_radius:
                print("Goal!!")
                break

            # Visualization: publish ROS msgs
            self.data_publisher(x)
            rate.sleep()
        print("Done")


if __name__ == "__main__":
    rospy.init_node('camera_pose_optimization')
    config = Config()

    # Load point cloud
    index = 1612893730.3432848
    points_filename = os.path.join(FE_PATH, f"src/traj_data/points/point_cloud_{index}.npz")
    pts_np = np.load(points_filename)['pts']
    # make sure the point cloud is of (N x 3) shape:
    if pts_np.shape[1] > pts_np.shape[0]:
        pts_np = pts_np.transpose()

    planner = Planner(config, pts_np)

    # initial state [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
    x_initial = np.array([13.0, 10.0, math.pi / 8.0, 0.0, 0.0])
    # goal position [x(m), y(m)]
    goal = np.array([15.0, 15.0])

    planner.plan(x_initial, goal)
