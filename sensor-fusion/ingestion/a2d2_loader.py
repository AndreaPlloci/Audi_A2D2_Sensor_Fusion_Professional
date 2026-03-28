"""
Modulo per il caricamento e la strutturazione dei dati grezzi del dataset Audi A2D2.
Gestisce file LiDAR (.npz), immagini camera (.png) e calibrazione (JSON).
"""

import json
import numpy as np
import cv2
from pathlib import Path
from dataclasses import dataclass


@dataclass
class LidarFrame:
    """Struttura dati per un singolo frame LiDAR."""
    timestamp: int
    points: np.ndarray          # (N, 3) - coordinate XYZ in metri
    reflectance: np.ndarray     # (N,)   - intensità di riflessione [0, 1]
    rows: np.ndarray            # (N,)   - riga pixel camera associata
    cols: np.ndarray            # (N,)   - colonna pixel camera associata
    distance: np.ndarray        # (N,)   - distanza euclidea dal sensore
    valid: np.ndarray           # (N,)   - flag di validità del punto


@dataclass
class CameraFrame:
    """Struttura dati per un singolo frame camera."""
    timestamp: int
    camera_id: str
    image: np.ndarray           # (H, W, 3) - immagine RGB
    image_path: Path


@dataclass
class CalibrationData:
    """Dati di calibrazione intrinseci ed estrinseci camera-LiDAR."""
    camera_id: str
    intrinsic: np.ndarray       # (3, 3) - matrice intrinseca
    extrinsic: np.ndarray       # (4, 4) - trasformazione cam2lidar
    distortion: np.ndarray      # (5,)   - coefficienti distorsione


class A2D2Loader:
    """
    Carica e struttura i dati del dataset Audi A2D2 per la pipeline di fusione.

    Struttura attesa del dataset:
        a2d2/
        └── camera_lidar/
            └── <sequenza>/
                ├── camera/
                │   └── cam_front_center/  (e altre 5 camere)
                │       └── <timestamp>.png
                ├── lidar/
                │   └── cam_front_center/
                │       └── <timestamp>.npz
                └── cams_lidars.json
    """

    def __init__(self, sequence_path: Path):
        self.sequence_path = sequence_path
        self.calibration_path = sequence_path / "cams_lidars.json"

        if not self.sequence_path.exists():
            raise FileNotFoundError(f"Sequenza non trovata: {sequence_path}")
        if not self.calibration_path.exists():
            raise FileNotFoundError(f"File di calibrazione non trovato: {self.calibration_path}")

        self._calibration_cache: dict[str, CalibrationData] = {}

    def load_lidar_frame(self, npz_path: Path) -> LidarFrame:
        """Carica un frame LiDAR da file .npz e restituisce una struttura tipizzata."""
        data = np.load(str(npz_path), allow_pickle=True)
        timestamp = int(npz_path.stem.split("_")[0])

        return LidarFrame(
            timestamp=timestamp,
            points=data["points"].astype(np.float32),
            reflectance=data["reflectance"].astype(np.float32),
            rows=data["row"].astype(np.int32),
            cols=data["col"].astype(np.int32),
            distance=data["distance"].astype(np.float32),
            valid=data["valid"].astype(bool),
        )

    def load_camera_frame(self, image_path: Path, camera_id: str) -> CameraFrame:
        """Carica un frame camera da file .png come array NumPy in formato RGB."""
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise IOError(f"Impossibile leggere immagine: {image_path}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        timestamp = int(image_path.stem.split("_")[0])

        return CameraFrame(
            timestamp=timestamp,
            camera_id=camera_id,
            image=image_rgb,
            image_path=image_path,
        )

    def load_calibration(self, camera_id: str) -> CalibrationData:
        """Carica e restituisce i parametri di calibrazione per una camera specifica."""
        if camera_id in self._calibration_cache:
            return self._calibration_cache[camera_id]

        with open(self.calibration_path, "r") as f:
            calib_raw = json.load(f)

        cam_data = calib_raw["cameras"][camera_id]

        calibration = CalibrationData(
            camera_id=camera_id,
            intrinsic=np.array(cam_data["CamMatrix"], dtype=np.float64),
            extrinsic=np.array(cam_data["cam2lidar_transform"], dtype=np.float64),
            distortion=np.array(cam_data["Distortion"], dtype=np.float64),
        )

        self._calibration_cache[camera_id] = calibration
        return calibration

    def list_lidar_files(self, camera_id: str = "cam_front_center") -> list[Path]:
        """Restituisce la lista ordinata dei file LiDAR per una data camera."""
        lidar_dir = self.sequence_path / "lidar" / camera_id
        return sorted(lidar_dir.glob("*.npz"))

    def list_camera_files(self, camera_id: str = "cam_front_center") -> list[Path]:
        """Restituisce la lista ordinata delle immagini per una data camera."""
        camera_dir = self.sequence_path / "camera" / camera_id
        return sorted(camera_dir.glob("*.png"))
