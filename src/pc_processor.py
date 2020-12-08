#!/usr/bin/env python

import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import CameraInfo
import tf, tf2_ros

import torch
from pytorch3d.transforms import quaternion_apply, quaternion_invert
from pointcloud_utils import pointcloud2_to_xyz_array, xyz_array_to_pointcloud2
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PulsarPointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor,
    PerspectiveCameras
)
import time
import cv2


class PointsProcessor:
    def __init__(self,
                 pc_topic='/final_cost_cloud',
                 cam_info_topics=['/viz/camera_0/camera_info',
                                  # '/viz/camera_1/camera_info',
                                  # '/viz/camera_2/camera_info',
                                  # '/viz/camera_3/camera_info',
                                  # '/viz/camera_4/camera_info',
                                  # '/viz/camera_5/camera_info'
                                  ],
                 min_dist=1.0,
                 max_dist=15.0,
                 ):
        self.pc_frame = None
        self.points = None
        self.pc_clip_limits = [rospy.get_param('~min_dist', min_dist),
                               rospy.get_param('~max_dist', max_dist)]
        self.device = torch.device("cuda:0")

        self.pc_topic = rospy.get_param('~pointcloud_topic', pc_topic)
        print("Subscribed to " + self.pc_topic)
        pc_sub = rospy.Subscriber(pc_topic, PointCloud2, self.pc_callback)

        for cam_info_topic in cam_info_topics:
            print("Subscribed to " + cam_info_topic)
            cam_info_sub = rospy.Subscriber(cam_info_topic, CameraInfo, self.cam_info_callback)

        self.tl = tf.TransformListener()

    @staticmethod
    def ego_to_cam_torch(points, trans, quat):
        """Transform points (N x 3) from ego frame into a pinhole camera
        """
        points = points - trans
        quat_inv = quaternion_invert(quat)
        points = quaternion_apply(quat_inv, points)
        return points.T.float()

    @staticmethod
    def get_only_in_img_mask(pts, H, W, intrins):
        """pts should be 3 x N
        """
        pts = intrins @ pts
        pts[:2] /= pts[2:3]
        return (pts[2] > 0) & \
               (pts[0] > 1) & (pts[0] < W - 1) & \
               (pts[1] > 1) & (pts[1] < H - 1)

    def render_pc_image(self,
                        verts: torch.Tensor,
                        intrins: torch.Tensor,
                        fovH, fovW):
        """
        verts.size() = N x 3
        """
        rgb = verts - torch.min(verts)
        rgb = rgb / torch.max(rgb).to(self.device)

        point_cloud = Pointclouds(points=[verts], features=[rgb])

        R = torch.eye(3, dtype=torch.float32, device=self.device)[None, ...]
        T = torch.Tensor([[0., 0., 0.]]).to(self.device)
        cameras = PerspectiveCameras(
            R=R,
            T=T,
            K=intrins.unsqueeze(0),
            device=self.device,
        )

        n_points = verts.size()[0]
        vert_rad = 0.01 * torch.ones(n_points, dtype=torch.float32, device=self.device)

        raster_settings = PointsRasterizationSettings(
            image_size=(fovW, fovH),
            radius=vert_rad,
        )
        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
        renderer = PulsarPointsRenderer(rasterizer=rasterizer).to(self.device)

        # Render an image
        image = renderer(point_cloud,
                         gamma=(1.0e-1,),  # Renderer blending parameter gamma, in [1., 1e-5].
                         znear=(1.0,),
                         zfar=(45.0,),
                         radius_world=True,
                         bg_col=torch.ones((3,), dtype=torch.float32, device=self.device),
                         )[0]
        return image

    @staticmethod
    def publish_pointcloud(points, topic_name, stamp, frame_id):
        # create PointCloud2 msg
        pc_msg = xyz_array_to_pointcloud2(points, stamp=stamp, frame_id=frame_id)
        pub = rospy.Publisher(topic_name, PointCloud2, queue_size=1)
        pub.publish(pc_msg)

    def pc_callback(self, pc_msg):
        points = pointcloud2_to_xyz_array(pc_msg)
        self.pc_frame = pc_msg.header.frame_id
        self.points = points.T

    def cam_info_callback(self, cam_info_msg):
        t0 = time.time()
        fovH = cam_info_msg.height
        fovW = cam_info_msg.width

        cam_frame = cam_info_msg.header.frame_id
        K = torch.zeros((4, 4))
        K[0][0] = cam_info_msg.K[0]
        K[0][2] = cam_info_msg.K[2]
        K[1][1] = cam_info_msg.K[4]
        K[1][2] = cam_info_msg.K[5]
        K[2][2] = 1.
        K[3][3] = 1.
        K = K.float().to(self.device)

        if self.pc_frame is not None:  # and self.tl.frameExists(self.pc_frame):
            self.run(fovH, fovW, K, cam_frame, output_pc_topic=f'/{cam_frame}/pointcloud')
        # print(f'[INFO]: Callback run time {1000 * (time.time() - t0):.1f} ms')

    def run(self, fovH, fovW, intrins, cam_frame, output_pc_topic='/output/pointcloud'):
        t1 = time.time()
        # find transformation between lidar and camera
        t = self.tl.getLatestCommonTime(self.pc_frame, cam_frame)
        trans, quat = self.tl.lookupTransform(self.pc_frame, cam_frame, t)

        quat_torch = torch.tensor([quat[3], quat[0], quat[1], quat[2]]).to(self.device)
        points_torch = torch.from_numpy(self.points).T.to(self.device)
        trans_torch = torch.unsqueeze(torch.tensor(trans), 0).to(self.device)
        ego_pts_torch = self.ego_to_cam_torch(points_torch, trans_torch, quat_torch)

        # find points that are observed by the camera (in its FOV)
        frame_mask = self.get_only_in_img_mask(ego_pts_torch, fovH, fovW, intrins[:3, :3])
        cam_pts = ego_pts_torch[:, frame_mask]

        # clip points between 1.0 and 5.0 meters distance from the camera
        dist_mask = (cam_pts[2] > self.pc_clip_limits[0]) & (cam_pts[2] < self.pc_clip_limits[1])
        cam_pts = cam_pts[:, dist_mask]
        print(f'[INFO]: Number of observed points from {cam_frame} is: {cam_pts.shape[1]}')

        self.publish_pointcloud(cam_pts.cpu().numpy().T, output_pc_topic, rospy.Time.now(), cam_frame)
        # print(f'[INFO]: Processing took {1000*(time.time()-t1):.1f} ms')

        # render and image of observed point cloud
        image = self.render_pc_image(cam_pts.T, intrins, fovH, fovW)

        image_vis = cv2.resize(image.cpu().numpy(), (fovW//2, fovH//2))
        image_vis = cv2.flip(image_vis, -1)
        cv2.imshow('Rendered pc image', image_vis)
        cv2.waitKey(3)


if __name__ == '__main__':
    rospy.init_node('pc_processor_node')
    proc = PointsProcessor()
    rospy.spin()
