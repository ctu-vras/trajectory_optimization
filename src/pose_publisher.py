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
    # Define initial position to optimize
    trans = np.array([rospy.get_param('pose_publisher/x', 6.0),
                      rospy.get_param('pose_publisher/y', 2.0),
                      rospy.get_param('pose_publisher/z', 0.0)])

    quat_xyzw = tf.transformations.quaternion_from_euler(
                      rospy.get_param('pose_publisher/roll', 0.0),
                      rospy.get_param('pose_publisher/pitch', 0.0),
                      rospy.get_param('pose_publisher/yaw', 0.0))

    # Run optimization loop
    rate = rospy.Rate(rospy.get_param('pose_publisher/rate', 1))
    while True:
        if rospy.is_shutdown():
            break

        publish_pose(trans, quat_xyzw, rospy.get_param('pose_publisher/output_topic', '/pose'), rospy.Time.now(), frame_id='world')
        rate.sleep()
