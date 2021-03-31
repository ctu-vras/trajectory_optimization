import numpy as np
from scipy.spatial import ConvexHull
import torch
try:
    import open3d as o3d
except:
    print("No Open3D installed")
# Data structures and functions for rendering
try:
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
except:
    print("No pytorch3d installed")
import rospy
from cv_bridge import CvBridge
import tf2_ros
import tf
from pointcloud_utils import xyz_array_to_pointcloud2
from pointcloud_utils import xyzi_array_to_pointcloud2
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, CameraInfo
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import TransformStamped, PoseStamped
from nav_msgs.msg import Path


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


def hidden_pts_removal(pts: torch.Tensor, device, R_param: int=2):
    """
    :param pts: input point cloud, Nx3
    :type pts: torch.Tensor
    :param device: CPU, torch.device("cpu"), or GPU, torch.device("cuda")
    :returns: point cloud after HPR algorithm, Nx3
    :rtype: torch.Tensor
    """
    # Initialize the points visible from camera location
    flippedPoints = sphericalFlip(pts, device, R_param)

    visibleHull = convexHull(flippedPoints, device)
    visibleVertex = visibleHull.vertices[:-1]  # indexes of visible points
    # convert indexes to mask
    visibleMask = torch.zeros(pts.size()[0], device=device)
    visibleMask[visibleVertex] = 1

    pts_visible = pts[visibleVertex, :]
    return pts_visible, visibleMask


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
                    K: torch.Tensor,
                    height: float,
                    width: float,
                    R=None,
                    T=None,
                    device=torch.device('cuda'),
                    gamma=1.0e-1,  # Renderer blending parameter gamma, in [1., 1e-5].
                    znear=1.0,
                    zfar=10.0,
):
    """
    points.size() = N x 3, point cloud in camera frame
    """
    rgb = verts - torch.min(verts)
    rgb = rgb / torch.max(rgb).to(device)

    point_cloud = Pointclouds(points=[verts], features=[rgb])

    if R is None:
        R = torch.eye(3).unsqueeze(0).to(device)
    if T is None:
        T = torch.tensor([[0., 0., 0.]]).to(device)

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

    # find points that are observed by the camera (in its FOV)
    pts_homo = intrins[:3, :3] @ points
    pts_homo[:2] /= pts_homo[2:3]
    fov_mask = (pts_homo[2] > 0) & \
               (pts_homo[0] > 1) & (pts_homo[0] < img_width - 1) & \
               (pts_homo[1] > 1) & (pts_homo[1] < img_height - 1)
    points = points[:, torch.logical_and(dist_mask, fov_mask)].T
    return points, dist_mask, fov_mask


def denormalize(x, eps=1e-6):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)
    x = (x - x_min) / np.max([(x_max - x_min), eps])
    x = x.clip(0, 1)
    return x


def publish_image(img, topic='/image/compressed'):
    img_uint8 = np.uint8(255 * denormalize(img))
    bridge = CvBridge()
    img_msg = bridge.cv2_to_imgmsg(img_uint8, "bgr8")
    pub = rospy.Publisher(topic, Image, queue_size=1)
    pub.publish(img_msg)


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


def publish_pointcloud(points, topic_name, stamp, frame_id):
    # create PointCloud2 msg
    if points.shape[1] == 3:
        pc_msg = xyz_array_to_pointcloud2(points, stamp=stamp, frame_id=frame_id)
    elif points.shape[1] == 4:
        pc_msg = xyzi_array_to_pointcloud2(points, stamp=stamp, frame_id=frame_id)
    pub = rospy.Publisher(topic_name, PointCloud2, queue_size=1)
    pub.publish(pc_msg)


def publish_tf_pose(pose, orient, child_frame_id, frame_id="world"):
    br = tf2_ros.TransformBroadcaster()
    t = TransformStamped()
    t.header.stamp = rospy.Time.now()
    t.header.frame_id = frame_id
    t.child_frame_id = child_frame_id
    t.transform.translation.x = pose[0]
    t.transform.translation.y = pose[1]
    t.transform.translation.z = pose[2]
    t.transform.rotation.x = orient[0]
    t.transform.rotation.y = orient[1]
    t.transform.rotation.z = orient[2]
    t.transform.rotation.w = orient[3]
    br.sendTransform(t)


def publish_camera_info(image_width=1232, image_height=1616,
                        K=[758.03967, 0.0, 621.46572, 0.0, 761.62359, 756.86402, 0.0, 0.0, 1.0],
                        D=[-0.20571, 0.04103, -0.00101, 0.00098, 0.0],
                        R=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                        P=[638.81494, 0.0, 625.98561, 0.0, 0.0, 585.79797, 748.57858, 0.0, 0.0, 0.0, 1.0, 0.0],
                        topic_name="/camera_info",
                        frame_id="camera_frame",
                        distortion_model="plumb_bob"):
    camera_info_msg = CameraInfo()
    camera_info_msg.header.frame_id = frame_id
    camera_info_msg.header.stamp = rospy.Time.now()
    camera_info_msg.width = image_width
    camera_info_msg.height = image_height
    camera_info_msg.K = K  # calib_data["camera_matrix"]["data"]
    camera_info_msg.D = D  # calib_data["distortion_coefficients"]["data"]
    camera_info_msg.R = R  # calib_data["rectification_matrix"]["data"]
    camera_info_msg.P = P  # calib_data["projection_matrix"]["data"]
    camera_info_msg.distortion_model = distortion_model
    pub = rospy.Publisher(topic_name, CameraInfo, queue_size=1)
    pub.publish(camera_info_msg)


def to_pose_stamped(pose, orient, frame_id='world'):
    msg = PoseStamped()
    msg.header.seq = 0
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = frame_id
    msg.pose.position.x = pose[0]
    msg.pose.position.y = pose[1]
    msg.pose.position.z = pose[2]
    quaternion = tf.transformations.quaternion_from_euler(orient[0], orient[1], orient[2])
    msg.pose.orientation.x = quaternion[0]
    msg.pose.orientation.y = quaternion[1]
    msg.pose.orientation.z = quaternion[2]
    msg.pose.orientation.w = quaternion[3]
    msg.header.seq += 1
    msg.header.stamp = rospy.Time.now()
    return msg


def publish_pose(pose, orient, topic_name):
    msg = to_pose_stamped(pose, orient)
    pub = rospy.Publisher(topic_name, PoseStamped, queue_size=1)
    pub.publish(msg)


def publish_path(path_list, orient=[0,0,0,1], topic_name='/path', frame_id='world'):
    path = Path()
    for pose in path_list:
        msg = to_pose_stamped(pose, orient, frame_id=frame_id)
        path.header = msg.header
        path.poses.append(msg)
    pub = rospy.Publisher(topic_name, Path, queue_size=1)
    pub.publish(path)


def load_intrinsics(device=torch.device('cuda')):
    width, height = 1232., 1616.
    K = torch.tensor([[758.03967, 0., 621.46572, 0.],
                      [0., 761.62359, 756.86402, 0.],
                      [0., 0., 1., 0.],
                      [0., 0., 0., 1.]], dtype=torch.float32).to(device)
    K = K.unsqueeze(0)
    return K, width, height
