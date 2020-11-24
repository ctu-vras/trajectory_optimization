#!/usr/bin/env python

import rospy
from sensor_msgs.msg import PointCloud2

from pointcloud_utils import pointcloud2_to_xyz_array, pointcloud2_to_xyzrgb_array
from pointcloud_utils import xyz_array_to_pointcloud2
import numpy as np


class PointsProcessor:
    def __init__(self,
                 pc_topic='/dynamic_point_cloud',
                 # pc_topic='/final_cost_cloud',
                 cam_info_topic='/viz/camera_0/camera_info',
                 path_topic='/path'):
        self.points = None
        self.pc_clip_limits = [1.0, 15.0]

        self.pc_topic = rospy.get_param('~pointcloud_topic', pc_topic)
        print("Subscribed to " + self.pc_topic)
        pc_sub = rospy.Subscriber(pc_topic, PointCloud2, self.pc_callback)


    @staticmethod
    def publish_pointcloud(points, topic_name, stamp, frame_id):
        # create PointCloud2 msg
        pc_msg = xyz_array_to_pointcloud2(points, stamp=stamp, frame_id=frame_id)
        pub = rospy.Publisher(topic_name, PointCloud2, queue_size=1)
        pub.publish(pc_msg)

    def pc_callback(self, pc_msg):
        points = pointcloud2_to_xyzrgb_array(pc_msg)
        print(points.shape)
        # np.savez('final_cost_cloud.npz', verts=points)
        # print(points.shape)


if __name__ == '__main__':
    rospy.init_node('pc_processor_node')
    proc = PointsProcessor()
    rospy.spin()
