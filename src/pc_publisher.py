#!/usr/bin/env python

import sys
import os
import rospkg
FE_PATH = rospkg.RosPack().get_path('trajectory_optimization')
sys.path.append(os.path.join(FE_PATH, 'src/'))
import numpy as np
from tools import publish_pointcloud
import rospy


if __name__ == "__main__":
    rospy.init_node('pc_publisher')

    # Run loop
    rate = rospy.Rate(rospy.get_param('pc_publisher/rate', 1))
    while True:
        if rospy.is_shutdown():
            break

        # Load point cloud
        index = rospy.get_param('pc_publisher/pc_index', 10)
        if index == -1:
            index = np.random.choice(range(0, 30))
        points_filename = os.path.join(FE_PATH, f"data/points/point_cloud_{index}.npz")
        pts_np = np.load(points_filename)['pts']
        # make sure the point cloud is of (N x 3) shape:
        if pts_np.shape[1] > pts_np.shape[0]:
            pts_np = pts_np.transpose()

        publish_pointcloud(pts_np, rospy.get_param('pc_publisher/output_topic', '/pts'), rospy.Time.now(), 'world')
        rate.sleep()
