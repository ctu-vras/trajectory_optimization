import numpy as np
from scipy.spatial import ConvexHull
import open3d as o3d
import torch
# Data structures and functions for rendering
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
import rospy
from nav_msgs.msg import Odometry


# Torch HPR
def sphericalFlip(points, device, param):
    """
    Function used to Perform Spherical Flip on the Original Point Cloud
    """
    n = len(points)  # total number of points
    normPoints = torch.linalg.norm(points, dim=1)  # Normed points, sqrt(x^2 + y^2 + z^2)

    radius = torch.max(normPoints) * 10.0**param  # Radius of a sphere
    flippedPointsTemp = 2*torch.multiply(
                            torch.repeat_interleave((radius - normPoints).view(n, 1), len(points[0]), dim=1).to(device),
                            points)
    flippedPoints = torch.divide(flippedPointsTemp,
                                 torch.repeat_interleave(normPoints.view(n, 1), len(points[0]), dim=1).to(device),
                                 )  # Apply Equation to get Flipped Points
    flippedPoints += points
    return flippedPoints


def convexHull(points, device):
    """
    Function used to Obtain the Convex hull
    """
    points = torch.cat([points, torch.zeros((1, 3), device=device)], dim=0)  # All points plus origin
    # TODO: rewrite it with torch tensors to be differentiable
    #       and being possible to include in the optimization problem
    hull = ConvexHull(points.cpu().numpy())  # Visible points plus possible origin. Use its vertices property.
    return hull


def hidden_pts_removal(pts: torch.Tensor, device, R_param: int=2) -> torch.Tensor:
    """
    :param pts: input point cloud, Nx3
    :type pts: torch.Tensor
    :param device: CPU, torch.device("cpu"), or GPU, torch.device("cuda")
    :returns: point cloud after HPR algorithm, Nx3
    :rtype: torch.Tensor
    """
    # Initialize the points visible from camera location
    flippedPoints = sphericalFlip(pts, device, R_param)

    # try:
    #     visibleHull = convexHull(flippedPoints, device)
    #     visibleVertex = visibleHull.vertices[:-1]  # indexes of visible points
    #
    #     pts_visible = pts[visibleVertex, :]
    # except:
    #     print("HPR: Not enough pts to construct convex hull")
    #     pts_visible = pts

    visibleHull = convexHull(flippedPoints, device)
    visibleVertex = visibleHull.vertices[:-1]  # indexes of visible points

    pts_visible = pts[visibleVertex, :]
    return pts_visible


def hidden_pts_removal_o3d(pts):
    # transformations from ROS coord system to Open3d
    # angle = np.pi
    # Rx = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
    # Ry = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    # Rz = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    # pts = Ry @ Rz @ pts.T
    R = np.diag([1, -1, -1])
    pts = R @ pts.T
    pts = pts.T
    # define Open3d point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    diameter = np.linalg.norm(
        np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
    if diameter > 0:
        # print("Define parameters used for hidden_point_removal")
        # camera = [0, 0, diameter]
        camera = [0, 0, 0.]
        radius = diameter * 100
        # print("Get all points that are visible from given view point")
        _, pt_map = pcd.hidden_point_removal(camera, radius)
        # print("Visualize result")
        pcd_res = pcd.select_by_index(pt_map)
        pts_visible = np.asarray(pcd_res.points)
    else:
        # print('All the pts are visible here')
        pts_visible = pts
    # back to ROS coord system
    pts_visible = R.T @ pts_visible.T
    pts_visible = pts_visible.T
    return pts_visible


def render_pc_image(
                    verts: torch.Tensor,
                    R: torch.Tensor,
                    T: torch.Tensor,
                    K: torch.Tensor,
                    height: float,
                    width: float,
                    device,
                    gamma=1.0e-1,  # Renderer blending parameter gamma, in [1., 1e-5].
                    znear=1.0,
                    zfar=10.0,
):
    """
    verts.size() = N x 3, point cloud in camera frame
    """
    rgb = verts - torch.min(verts)
    rgb = rgb / torch.max(rgb).to(device)

    point_cloud = Pointclouds(points=[verts], features=[rgb])

    cameras = PerspectiveCameras(
        R=R,
        T=T,
        K=K,
        device=device,
    )

    n_points = verts.size()[0]
    vert_rad = 0.03 * torch.ones(n_points, dtype=torch.float32, device=device)

    raster_settings = PointsRasterizationSettings(
        image_size=(width, height),
        radius=vert_rad,
        points_per_pixel=1,
    )
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    renderer = PulsarPointsRenderer(rasterizer=rasterizer).to(device)

    # Render an image
    image = renderer(point_cloud,
                     gamma=(gamma,),  # Renderer blending parameter gamma, in [1., 1e-5].
                     znear=(znear,),
                     zfar=(zfar,),
                     radius_world=True,
                     bg_col=torch.ones((3,), dtype=torch.float32, device=device),
                     )[0]
    return image


def get_cam_frustum_pts(points, img_height, img_width, intrins, min_dist=1.0, max_dist=10.0):
    # clip points between MIN_DIST and MAX_DIST meters distance from the camera
    dist_mask = (points[2] > min_dist) & (points[2] < max_dist)
    points = points[:, dist_mask]

    # find points that are observed by the camera (in its FOV)
    pts_homo = intrins[:3, :3] @ points
    pts_homo[:2] /= pts_homo[2:3]
    frame_mask = (pts_homo[2] > 0) & \
                 (pts_homo[0] > 1) & (pts_homo[0] < img_width - 1) & \
                 (pts_homo[1] > 1) & (pts_homo[1] < img_height - 1)
    points = points[:, frame_mask]
    return points


def publish_odom(pose, orient, frame='/odom', topic='/odom_0'):
    odom_msg_0 = Odometry()
    odom_msg_0.header.stamp = rospy.Time.now()
    odom_msg_0.header.frame_id = frame
    odom_msg_0.pose.pose.position.x = pose[0]
    odom_msg_0.pose.pose.position.y = pose[1]
    odom_msg_0.pose.pose.position.z = pose[2]
    odom_msg_0.pose.pose.orientation.x = orient[0]
    odom_msg_0.pose.pose.orientation.y = orient[1]
    odom_msg_0.pose.pose.orientation.z = orient[2]
    odom_msg_0.pose.pose.orientation.w = orient[3]
    pub = rospy.Publisher(topic, Odometry, queue_size=1)
    pub.publish(odom_msg_0)


def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x
