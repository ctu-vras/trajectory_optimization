import torch
from tqdm.notebook import tqdm
import torch.nn as nn
import numpy as np
import cv2

# Data structures and functions for rendering
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    look_at_view_transform,
    PointsRasterizationSettings,
    PointsRenderer,
    PulsarPointsRenderer,
    PointsRasterizer,
    PerspectiveCameras
)
from tools import get_cam_frustum_pts
from tools import hidden_pts_removal


class Model(nn.Module):
    def __init__(self, points, dist_init, elev_init, azim_init):
        super().__init__()
        self.points = points
        self.device = points.device
        self.renderer = None
        # Renderer blending parameter gamma, in [1., 1e-5].
        self.gamma = torch.tensor([1.0e-1]).to(points.device)
        self.znear = torch.tensor([1.0]).to(points.device)
        self.zfar = torch.tensor([15.0]).to(points.device)
        self.bg_col = torch.ones((3,), dtype=torch.float32, device=points.device)

        # Create an optimizable parameter for distance, elevation, azimuth of the camera.
        self.camera_dist_elev_azim = nn.Parameter(
            torch.as_tensor([dist_init, elev_init, azim_init], dtype=torch.float32).to(points.device))

        K, width, height = self.load_intrinsics()
        self.K = K.to(points.device)
        self.width = torch.tensor([width]).to(points.device)
        self.height = torch.tensor([height]).to(points.device)
        self.raster_settings = self.load_raster_setting(self.width, self.height)
        self.eps = torch.tensor([1.0e-6]).to(points.device)
        self.min_dist = torch.tensor([1.0]).to(points.device)
        self.max_dist = torch.tensor([10.0]).to(points.device)

    def load_raster_setting(self, width, height):
        n_points = self.points.size()[0]
        vert_rad = 0.003 * torch.ones(n_points, dtype=torch.float32, device=device)
        raster_settings = PointsRasterizationSettings(
            image_size=(width, height),
            radius=vert_rad,
            points_per_pixel=1
        )
        return raster_settings

    @staticmethod
    def load_intrinsics():
        width, height = 1232., 1616.
        K = torch.tensor([[758.03967, 0.,        621.46572, 0.],
                          [0.,        761.62359, 756.86402, 0.],
                          [0.,        0.,        1.,        0.],
                          [0.,        0.,        0.,        1.]]).to(device)
        K = K.unsqueeze(0)
        return K, width, height

    def to_camera_frame(self, verts, R, T):
        R_inv = R.squeeze().T
        verts_cam = R_inv @ (verts - torch.repeat_interleave(T, len(verts), dim=0).to(self.device)).T
        verts_cam = verts_cam.T
        return verts_cam

    def render_pc_image(self, verts):
        # verts_norm = verts - torch.min(verts)
        # rgb = verts_norm / torch.max(verts_norm).to(self.device)
        rgb = torch.zeros_like(verts)
        point_cloud = Pointclouds(points=[verts], features=[rgb])

        R = torch.eye(3, dtype=torch.float32, device=self.device)[None, ...]
        T = torch.zeros((1, 3), dtype=torch.float32, device=self.device)
        cameras = PerspectiveCameras(R=R, T=T, K=self.K, device=self.device)
        self.renderer = PulsarPointsRenderer(
            rasterizer=PointsRasterizer(cameras=cameras, raster_settings=self.raster_settings),
            n_channels=3
        ).to(self.device)
        image = self.renderer(point_cloud.clone(),
                              gamma=self.gamma,
                              znear=self.znear,
                              zfar=self.zfar,
                              bg_col=self.bg_col,
                              )[0]
        return image

    def forward(self):
        # Render the image using the updated camera position. Based on the new position of the
        # camera we calculate the rotation and translation matrices
        distance, elevation, azimuth = self.camera_dist_elev_azim
        R, T = look_at_view_transform(distance, elevation, azimuth, device=device)

        # transform points to camera frame
        verts = self.to_camera_frame(self.points, R, T)

        # remove pts that are outside of the camera FOV
        verts_cam = get_cam_frustum_pts(verts.T,
                                        self.height, self.width,
                                        self.K.squeeze(0),
                                        min_dist=self.min_dist, max_dist=self.max_dist).T
        verts_visible = hidden_pts_removal(verts_cam.detach(), device=self.device)
        # verts_visible = verts_cam
        N_visible_pts = verts_visible.size()[0]
        # print(f'Number of visible points: {N_visible_pts}/{verts.size()[0]}')

        # render point cloud on an image plane
        image = self.render_pc_image(verts)
        # image = self.render_pc_image(verts_visible)
        return image, N_visible_pts

    def criterion(self, image, image_ref):
        # Calculate the loss
        loss = torch.mean((image - image_ref) ** 2)
        return loss

    def criterion_pts(self, N_visible_pts):
        # Calculate the loss based on the number of visible points in cloud
        N_visible_pts = torch.tensor([N_visible_pts], dtype=torch.float, requires_grad=True).to(self.device)
        loss = 1. / (N_visible_pts + self.eps)
        return loss


if __name__ == "__main__":
    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    # Load point cloud
    obj_filename = "../../../../catkin_ws/src/frontier_exploration/pts/cam_pts_camera_0_1607456676.1540315.npz"
    # obj_filename = "../../../../catkin_ws/src/frontier_exploration/pts/cam_pts_camera_0_1607456663.5413494.npz"
    pts_np = np.load(obj_filename)['pts'].transpose()
    verts = torch.tensor(pts_np).to(device)
    # rgb = (verts - torch.min(verts)) / torch.max(verts - torch.min(verts)).to(device)
    rgb = torch.zeros_like(verts)

    point_cloud = Pointclouds(points=[verts], features=[rgb])

    # Camera intrinsics
    width, height = 1232., 1616.
    K = torch.tensor([[758.03967, 0., 621.46572, 0.],
                      [0., 761.62359, 756.86402, 0.],
                      [0., 0., 1., 0.],
                      [0., 0., 0., 1.]]).to(device)
    K = K.unsqueeze(0)

    # Initialize a camera.
    R = torch.eye(3).unsqueeze(0).to(device)
    T = torch.Tensor([[0., 0., 0.]]).to(device)
    cameras = PerspectiveCameras(device=device, R=R, T=T, K=K)

    # Define the settings for rasterization.
    n_points = verts.size()[0]
    vert_rad = 0.003 * torch.ones(n_points, dtype=torch.float32, device=device)
    raster_settings = PointsRasterizationSettings(
        image_size=(width, height),
        radius=vert_rad,
        points_per_pixel=1
    )
    # Create a renderer
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    renderer = PulsarPointsRenderer(
        rasterizer=rasterizer,
        n_channels=3
    ).to(device)

    # Reference image
    image_ref = renderer(
        point_cloud,
        gamma=(1.0e-1,),  # Renderer blending parameter gamma, in [1., 1e-5].
        znear=(1.0,),
        zfar=(15.0,),
        bg_col=torch.ones((3,), dtype=torch.float32, device=device),
    )[0]

    # Initialize a model
    model = Model(points=verts,
                  dist_init=-1.0,
                  elev_init=10,
                  azim_init=50).to(device)
    # Create an optimizer. Here we are using Adam and we pass in the parameters of the model
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    for p in model.parameters():
        print(p)

    # Run optimization loop
    loop = tqdm(range(100))
    for i in loop:
        optimizer.zero_grad()
        image, N_visible_pts = model()
        # loss = model.criterion(image, image_ref)
        # loss.backward()
        loss = model.criterion_pts(N_visible_pts)
        loss.backward()
        print(loss)
        optimizer.step()

        loop.set_description('Optimizing (loss %.4f)' % loss.data)
        # if loss.item() < 200:
        #     break

        if i % 10 == 0:
            image = image[..., :3].detach().squeeze().cpu().numpy()
            # print(f'Loss: {loss.item()}')

            image_vis = cv2.resize(image[..., :3], (512, 512))
            image_ref_vis = cv2.resize(image_ref.cpu().numpy()[..., :3], (512, 512))
            cv2.imshow('Camera view', np.concatenate([image_vis, image_ref_vis], axis=1))
            cv2.waitKey(3)
