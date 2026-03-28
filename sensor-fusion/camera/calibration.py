"""
Gestione dei parametri di calibrazione camera del dataset A2D2.
Fornisce utilità per la rettifica immagini e la trasformazione di coordinate.
"""

import numpy as np
import cv2

from ingestion.a2d2_loader import CalibrationData, A2D2Loader


class CameraCalibrationManager:
    """
    Gestisce la calibrazione per tutte le camere di una sequenza A2D2.
    Fornisce rettifica immagine e conversione coordinate pixel ↔ raggio 3D.
    """

    def __init__(self, loader: A2D2Loader, camera_ids: list[str]):
        self._calibrations: dict[str, CalibrationData] = {}
        for cam_id in camera_ids:
            self._calibrations[cam_id] = loader.load_calibration(cam_id)

    def get_calibration(self, camera_id: str) -> CalibrationData:
        if camera_id not in self._calibrations:
            raise KeyError(f"Calibrazione non disponibile per camera: {camera_id}")
        return self._calibrations[camera_id]

    def undistort_image(self, image: np.ndarray, camera_id: str) -> np.ndarray:
        """Rimuove la distorsione radiale e tangenziale dall'immagine."""
        calib = self.get_calibration(camera_id)
        return cv2.undistort(image, calib.intrinsic, calib.distortion)

    def pixel_to_ray(self, pixel_uv: np.ndarray, camera_id: str) -> np.ndarray:
        """
        Converte coordinate pixel (u, v) in raggi 3D normalizzati
        nel riferimento camera (direzione senza profondità).
        """
        calib = self.get_calibration(camera_id)
        K_inv = np.linalg.inv(calib.intrinsic)

        pixels_h = np.hstack([pixel_uv, np.ones((len(pixel_uv), 1))])  # (N, 3)
        rays = (K_inv @ pixels_h.T).T                                    # (N, 3)

        norms = np.linalg.norm(rays, axis=1, keepdims=True)
        return rays / norms
