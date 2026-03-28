"""
Proiezione di punti LiDAR 3D sul piano immagine della camera (LiDAR→Camera).
Utilizza i parametri di calibrazione estrinseci e intrinseci del dataset A2D2.
"""

import numpy as np
from dataclasses import dataclass

from ingestion.a2d2_loader import CalibrationData
from lidar.point_cloud import ProcessedPointCloud


@dataclass
class ProjectedPoints:
    """Risultato della proiezione LiDAR su piano camera."""
    pixels_uv: np.ndarray       # (M, 2) - coordinate pixel (u, v) valide
    depths: np.ndarray          # (M,)   - profondità in metri nel riferimento camera
    points_3d: np.ndarray       # (M, 3) - punti 3D originali proiettati con successo
    reflectance: np.ndarray     # (M,)   - riflettanza dei punti validi
    image_width: int
    image_height: int


class LidarCameraProjector:
    """
    Proietta punti LiDAR 3D sul piano immagine di una camera usando la calibrazione A2D2.

    Il dataset A2D2 fornisce la trasformazione cam2lidar; qui viene invertita
    per ottenere la trasformazione lidar→camera necessaria alla proiezione.
    """

    def __init__(self, calibration: CalibrationData, image_height: int, image_width: int):
        self.calibration = calibration
        self.image_height = image_height
        self.image_width = image_width

        # Inverti cam2lidar per ottenere lidar→camera
        self.lidar_to_camera = np.linalg.inv(calibration.extrinsic)

    def project(self, point_cloud: ProcessedPointCloud) -> ProjectedPoints:
        """
        Proietta la point cloud sul piano immagine della camera.

        Pipeline:
        1. Trasforma punti da frame LiDAR a frame Camera
        2. Filtra punti dietro la camera (Z ≤ 0)
        3. Applica proiezione prospettica con matrice intrinseca
        4. Filtra punti fuori dal campo visivo dell'immagine
        """
        points = point_cloud.points  # (N, 3)
        n_points = points.shape[0]

        # Step 1: trasformazione nel riferimento camera
        points_h = np.hstack([points, np.ones((n_points, 1))])       # (N, 4) omogeneo
        points_cam = (self.lidar_to_camera @ points_h.T).T           # (N, 4)
        points_cam_xyz = points_cam[:, :3]                            # (N, 3)

        # Step 2: filtra punti con Z ≤ 0 (dietro la camera)
        front_mask = points_cam_xyz[:, 2] > 0
        points_cam_xyz      = points_cam_xyz[front_mask]
        reflectance_filtered = point_cloud.reflectance[front_mask]
        points_3d_filtered  = points[front_mask]

        # Step 3: proiezione prospettica
        K = self.calibration.intrinsic  # (3, 3)
        projected = (K @ points_cam_xyz.T).T  # (M, 3)

        u = projected[:, 0] / projected[:, 2]
        v = projected[:, 1] / projected[:, 2]
        depth = points_cam_xyz[:, 2]

        # Step 4: filtra punti fuori dal campo visivo
        in_bounds = (
            (u >= 0) & (u < self.image_width) &
            (v >= 0) & (v < self.image_height)
        )

        return ProjectedPoints(
            pixels_uv=np.stack([u[in_bounds], v[in_bounds]], axis=1).astype(np.float32),
            depths=depth[in_bounds].astype(np.float32),
            points_3d=points_3d_filtered[in_bounds],
            reflectance=reflectance_filtered[in_bounds],
            image_width=self.image_width,
            image_height=self.image_height,
        )

    def create_depth_map(self, projected: ProjectedPoints) -> np.ndarray:
        """
        Genera una depth map densa (H×W) dai punti proiettati.
        I pixel senza punto LiDAR hanno valore 0.0.
        In caso di collisioni, viene mantenuto il punto più vicino.
        """
        depth_map = np.zeros((self.image_height, self.image_width), dtype=np.float32)

        u_int = projected.pixels_uv[:, 0].astype(np.int32)
        v_int = projected.pixels_uv[:, 1].astype(np.int32)

        for i in range(len(u_int)):
            u, v, d = u_int[i], v_int[i], projected.depths[i]
            if depth_map[v, u] == 0.0 or d < depth_map[v, u]:
                depth_map[v, u] = d

        return depth_map
