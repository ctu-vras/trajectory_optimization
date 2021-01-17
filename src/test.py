#!/usr/bin/env python

# import torch
# import pytorch3d as p3d
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from tqdm import tqdm


np.random.seed(0)
N_points = 50
dist_range = [-10, 10]
reward_visible = 1.0
reward_invisible = 0.9

pts = []
for i in range(N_points):
    pose = np.random.random(2) * (dist_range[1] - dist_range[0]) + dist_range[0]
    point = {'pose': pose,
             'reward': reward_invisible}
    pts.append(point)

robot_pose = [-1, -2]
robot_yaw = -np.pi / 4
visibility_polygon = Polygon(np.array([(0, 0), (2, 7), (-2, 7)])+robot_pose)
traj = {'coords': np.array(robot_pose),
        'reward': 0.0}

plt.figure(figsize=(8, 8))

for _ in tqdm(range(50)):
    plt.cla()

    robot_pose[0] += -0.05  # np.random.random() * 0.1 - 0.05
    robot_pose[1] += 0.05  # np.random.random() * 0.1 - 0.05
    robot_yaw += 0.02  # np.random.random() * 0.04 - 0.02

    rot = np.array([[np.cos(robot_yaw), -np.sin(robot_yaw)],
                    [np.sin(robot_yaw), np.cos(robot_yaw)]])
    visibility_polygon = Polygon((rot @ np.array([(0, 0), (2, 7), (-2, 7)]).T).T + robot_pose)

    # Visualization
    plt.plot(*visibility_polygon.exterior.xy)
    plt.plot(robot_pose[0], robot_pose[1], 'x', color='k', markersize=5)

    pts_visible = []
    rewards = 0
    for point in pts:
        p = point['pose']
        r = point['reward']
        if visibility_polygon.contains(Point(p)):
            point['reward'] = reward_visible
            pts_visible.append(point)
            rewards += reward_visible
            plt.plot(p[0], p[1], 'ro', color='b')
            plt.text(p[0], p[1], reward_visible)
        else:
            plt.plot(p[0], p[1], 'ro', color='r')
            plt.text(p[0], p[1], reward_invisible)

    traj['coords'] = np.vstack([traj['coords'], robot_pose])
    traj['reward'] += rewards

    plt.plot(traj['coords'][:, 0], traj['coords'][:, 1],
             linewidth=2, color='green', label='trajectory')
    print(f'Number of visible points: {len(pts_visible)}, \nRewards: {rewards}')

    plt.grid()
    plt.legend()
    plt.xlim(dist_range)
    plt.ylim(dist_range)
    plt.draw()
    plt.pause(0.01)

print(f"Trajectory reward: {traj['reward']}")

plt.show()
