"""
Costruzione di scene 3D aggregate da più frame fusi.
Combina point cloud da frame multipli in una rappresentazione 3D unificata.
"""

import numpy as np
import open3d as o3d
from dataclasses import dataclass
from pathlib import Path

from fusion.fusion_engine import FusedFrame


@dataclass
class Scene3D:
    """Scena 3D ricostruita da più frame fusi."""
    points: np.ndarray              # (N, 3) - tutti i punti aggregati
    reflectance: np.ndarray         # (N,)   - riflettanza per punto
    frame_ids: np.ndarray           # (N,)   - ID frame di origine
    n_frames: int
    bounding_box: tuple[np.ndarray, np.ndarray]  # (min_xyz, max_xyz)


class SceneBuilder:
    """
    Aggrega frame fusi multipli in un'unica scena 3D coerente.
    Utile per ricostruzione di scene estese nel tempo e visualizzazione multi-frame.
    """

    def build_from_sequence(self, fused_frames: list[FusedFrame]) -> Scene3D:
        """Aggrega le point cloud di tutti i frame in un'unica nuvola di punti 3D."""
        all_points:      list[np.ndarray] = []
        all_reflectance: list[np.ndarray] = []
        all_frame_ids:   list[np.ndarray] = []

        for frame in fused_frames:
            pc = frame.point_cloud
            all_points.append(pc.points)
            all_reflectance.append(pc.reflectance)
            all_frame_ids.append(
                np.full(len(pc.points), frame.frame_id, dtype=np.int32)
            )

        points      = np.vstack(all_points)
        reflectance = np.concatenate(all_reflectance)
        frame_ids   = np.concatenate(all_frame_ids)

        return Scene3D(
            points=points,
            reflectance=reflectance,
            frame_ids=frame_ids,
            n_frames=len(fused_frames),
            bounding_box=(points.min(axis=0), points.max(axis=0)),
        )

    def to_open3d(self, scene: Scene3D) -> o3d.geometry.PointCloud:
        """Converte la scena aggregata in PointCloud Open3D per visualizzazione o export."""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(scene.points.astype(np.float64))

        colors = np.stack(
            [scene.reflectance, scene.reflectance, scene.reflectance], axis=1
        )
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

        return pcd

    def save_ply(self, scene: Scene3D, output_path: Path) -> None:
        """Salva la scena in formato .ply per utilizzo con visualizzatori 3D esterni."""
        pcd = self.to_open3d(scene)
        o3d.io.write_point_cloud(str(output_path), pcd)
        print(f"  Scena salvata in: {output_path}")
