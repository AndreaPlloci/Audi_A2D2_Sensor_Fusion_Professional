"""
Elaborazione delle immagini camera A2D2.
Gestisce la creazione di overlay LiDAR-camera e il parsing delle annotazioni semantiche.
"""

import json
import numpy as np
import cv2
from pathlib import Path
from dataclasses import dataclass

from ingestion.a2d2_loader import CameraFrame
from lidar.projection import ProjectedPoints


# Palette colori semantici A2D2 (classe → colore BGR)
# Fonte: documentazione ufficiale Audi A2D2
A2D2_CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "Car 1":         (0, 0, 255),
    "Car 2":         (0, 0, 200),
    "Pedestrian 1": (255, 0, 0),
    "Pedestrian 2": (200, 0, 0),
    "Truck 1":       (0, 255, 0),
    "Road":          (128, 64, 128),
    "Sidewalk":      (232, 35, 244),
    "Building":      (70, 70, 70),
    "Nature object": (107, 142, 35),
    "Sky":           (70, 130, 180),
}


@dataclass
class AnnotatedFrame:
    """Frame camera con overlay LiDAR e depth map."""
    camera_id: str
    image_rgb: np.ndarray           # Immagine originale RGB
    depth_map: np.ndarray           # Depth map da LiDAR proiettato
    overlay: np.ndarray             # Immagine con punti LiDAR sovrapposti
    n_projected_points: int


class ImageProcessor:
    """Elabora frame camera A2D2 e genera visualizzazioni della fusione LiDAR-camera."""

    def create_lidar_overlay(
        self,
        camera_frame: CameraFrame,
        projected: ProjectedPoints,
        colormap: int = cv2.COLORMAP_JET,
        point_radius: int = 2,
    ) -> AnnotatedFrame:
        """
        Genera un overlay visivo dei punti LiDAR proiettati sull'immagine camera.
        I punti sono colorati in base alla profondità (rosso=vicino, blu=lontano).
        """
        image_bgr = cv2.cvtColor(camera_frame.image.copy(), cv2.COLOR_RGB2BGR)

        h, w = camera_frame.image.shape[:2]
        depth_map = np.zeros((h, w), dtype=np.float32)

        if len(projected.pixels_uv) > 0:
            depths = projected.depths
            d_min, d_max = depths.min(), depths.max()
            depths_norm = ((depths - d_min) / (d_max - d_min + 1e-6) * 255).astype(np.uint8)
            colored = cv2.applyColorMap(depths_norm.reshape(-1, 1), colormap)  # (M, 1, 3)

            u_arr = projected.pixels_uv[:, 0].astype(np.int32)
            v_arr = projected.pixels_uv[:, 1].astype(np.int32)

            for i in range(len(u_arr)):
                u, v = u_arr[i], v_arr[i]
                color = tuple(int(c) for c in colored[i, 0])
                cv2.circle(image_bgr, (u, v), point_radius, color, -1)

                # Aggiorna depth map (mantieni punto più vicino)
                d = projected.depths[i]
                if depth_map[v, u] == 0.0 or d < depth_map[v, u]:
                    depth_map[v, u] = d

        return AnnotatedFrame(
            camera_id=camera_frame.camera_id,
            image_rgb=camera_frame.image,
            depth_map=depth_map,
            overlay=cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
            n_projected_points=len(projected.pixels_uv),
        )

    def load_semantic_labels(self, label_path: Path) -> dict:
        """
        Carica le annotazioni semantiche A2D2 dal file JSON corrispondente all'immagine.
        Restituisce dizionario con bounding boxes e classi per ogni oggetto annotato.
        """
        if not label_path.exists():
            return {}

        with open(label_path, "r") as f:
            return json.load(f)
