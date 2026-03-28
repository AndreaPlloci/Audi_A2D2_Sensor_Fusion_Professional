"""
Operazioni di elaborazione sulle point cloud LiDAR del dataset A2D2.
Include filtraggio per distanza, downsampling con voxel grid e normalizzazione.
"""

import numpy as np
import open3d as o3d
from dataclasses import dataclass

from config import LIDAR_VOXEL_SIZE, LIDAR_MAX_RANGE, LIDAR_MIN_RANGE
from ingestion.a2d2_loader import LidarFrame


@dataclass
class ProcessedPointCloud:
    """Point cloud elaborata e pronta per la fase di fusione."""
    points: np.ndarray          # (N, 3) - XYZ filtrati e downsamplati
    reflectance: np.ndarray     # (N,)   - riflettanza normalizzata
    point_count_original: int   # Numero punti prima del filtraggio


class PointCloudProcessor:
    """Elabora frame LiDAR grezzi A2D2 applicando filtraggio e downsampling."""

    def __init__(
        self,
        voxel_size: float = LIDAR_VOXEL_SIZE,
        max_range: float = LIDAR_MAX_RANGE,
        min_range: float = LIDAR_MIN_RANGE,
    ):
        self.voxel_size = voxel_size
        self.max_range = max_range
        self.min_range = min_range

    def process(self, frame: LidarFrame) -> ProcessedPointCloud:
        """
        Pipeline completa di preprocessing:
        1. Filtra i punti non validi
        2. Applica filtro per range min/max
        3. Applica voxel grid downsampling
        """
        n_original = len(frame.points)

        # Step 1: rimuovi punti non validi
        points = frame.points[frame.valid]
        reflectance = frame.reflectance[frame.valid]

        # Step 2: filtra per range
        distances = np.linalg.norm(points, axis=1)
        range_mask = (distances >= self.min_range) & (distances <= self.max_range)
        points = points[range_mask]
        reflectance = reflectance[range_mask]

        # Step 3: voxel grid downsampling tramite Open3D
        points, reflectance = self._voxel_downsample(points, reflectance)

        return ProcessedPointCloud(
            points=points,
            reflectance=reflectance,
            point_count_original=n_original,
        )

    def _voxel_downsample(
        self, points: np.ndarray, reflectance: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Riduce la densità della point cloud tramite voxel grid averaging."""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

        # Codifica la riflettanza come colore per mantenerla durante il downsampling
        refl_rgb = np.stack([reflectance, reflectance, reflectance], axis=1)
        pcd.colors = o3d.utility.Vector3dVector(refl_rgb.astype(np.float64))

        pcd_down = pcd.voxel_down_sample(voxel_size=self.voxel_size)

        points_down = np.asarray(pcd_down.points, dtype=np.float32)
        reflectance_down = np.asarray(pcd_down.colors, dtype=np.float32)[:, 0]

        return points_down, reflectance_down

    def to_open3d(self, processed: ProcessedPointCloud) -> o3d.geometry.PointCloud:
        """Converte una ProcessedPointCloud in oggetto Open3D per visualizzazione."""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(processed.points.astype(np.float64))

        colors = np.stack(
            [processed.reflectance, processed.reflectance, processed.reflectance], axis=1
        )
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

        return pcd
