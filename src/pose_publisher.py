#!/usr/bin/env python

import sys
import os
import rospkg
FE_PATH = rospkg.RosPack().get_path('trajectory_optimization')
sys.path.append(os.path.join(FE_PATH, 'src/'))
import numpy as np
from tools import publish_pose
import rospy
import tf


if __name__ == "__main__":
    rospy.init_node('pose_publisher')
    rate = rospy.Rate(rospy.get_param('pose_publisher/rate', 1))
    while True:
        if rospy.is_shutdown():
            break
        # Define initial position to optimize
        trans = np.array([rospy.get_param('pose_publisher/x', np.random.random() * 5 + 15),
                          rospy.get_param('pose_publisher/y', np.random.random() * 5 + 15),
                          rospy.get_param('pose_publisher/z', np.random.random() * 2)])

        quat_xyzw = tf.transformations.quaternion_from_euler(
            rospy.get_param('pose_publisher/roll', np.random.random() * np.pi),
            rospy.get_param('pose_publisher/pitch', np.random.random() * np.pi),
            rospy.get_param('pose_publisher/yaw', np.random.random() * np.pi))

        publish_pose(trans, quat_xyzw, rospy.get_param('pose_publisher/output_topic', '/pose'), rospy.Time.now(), frame_id='world')
        rate.sleep()
