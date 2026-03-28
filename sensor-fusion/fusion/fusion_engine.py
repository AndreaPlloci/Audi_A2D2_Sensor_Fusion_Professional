"""
Motore principale di sensor fusion: allinea temporalmente LiDAR e camera,
proietta i punti 3D e costruisce frame fusi multi-modali.
"""

from dataclasses import dataclass
from pathlib import Path

from ingestion.a2d2_loader import A2D2Loader, LidarFrame, CameraFrame
from ingestion.bus_parser import BusSignalParser, BusSignalFrame
from lidar.point_cloud import PointCloudProcessor, ProcessedPointCloud
from lidar.projection import LidarCameraProjector, ProjectedPoints
from camera.image_processor import ImageProcessor, AnnotatedFrame
from config import SYNC_TOLERANCE_MS


@dataclass
class FusedFrame:
    """Frame completamente fuso: LiDAR + Camera + Bus signals sincronizzati."""
    frame_id: int
    timestamp_us: int
    camera_id: str
    camera_frame: CameraFrame
    lidar_frame: LidarFrame
    point_cloud: ProcessedPointCloud
    projected_points: ProjectedPoints
    annotated: AnnotatedFrame
    bus_signal: BusSignalFrame | None


class SensorFusionEngine:
    """
    Orchestratore della pipeline di sensor fusion per il dataset A2D2.

    Flusso di elaborazione per ogni frame:
    1. Carica frame LiDAR e camera con timestamp allineati
    2. Preprocessing point cloud (filtraggio + downsampling)
    3. Proiezione LiDAR sul piano camera
    4. Generazione overlay e depth map
    5. Associazione con segnali bus veicolo
    """

    def __init__(
        self,
        loader: A2D2Loader,
        bus_parser: BusSignalParser | None,
        camera_id: str = "cam_front_center",
    ):
        self.loader = loader
        self.bus_parser = bus_parser
        self.camera_id = camera_id

        self.pc_processor = PointCloudProcessor()
        self.img_processor = ImageProcessor()
        self.calibration = loader.load_calibration(camera_id)

    def process_frame(
        self,
        lidar_path: Path,
        camera_path: Path,
        frame_id: int,
    ) -> FusedFrame:
        """Esegue la pipeline completa di fusione per una coppia LiDAR-camera."""
        lidar_frame  = self.loader.load_lidar_frame(lidar_path)
        camera_frame = self.loader.load_camera_frame(camera_path, self.camera_id)

        h, w = camera_frame.image.shape[:2]

        point_cloud = self.pc_processor.process(lidar_frame)

        projector = LidarCameraProjector(self.calibration, h, w)
        projected = projector.project(point_cloud)

        annotated = self.img_processor.create_lidar_overlay(camera_frame, projected)

        # Sincronizzazione bus signals con tolleranza configurabile
        bus_signal = None
        if self.bus_parser:
            tolerance_us = SYNC_TOLERANCE_MS * 1_000
            bus_signal = self.bus_parser.get_nearest(lidar_frame.timestamp, tolerance_us)

        return FusedFrame(
            frame_id=frame_id,
            timestamp_us=lidar_frame.timestamp,
            camera_id=self.camera_id,
            camera_frame=camera_frame,
            lidar_frame=lidar_frame,
            point_cloud=point_cloud,
            projected_points=projected,
            annotated=annotated,
            bus_signal=bus_signal,
        )

    def process_sequence(self, max_frames: int | None = None) -> list[FusedFrame]:
        """
        Elabora una sequenza completa di frame LiDAR-camera in ordine cronologico.
        I frame vengono abbinati per stem del filename (stesso timestamp).
        """
        lidar_files  = self.loader.list_lidar_files(self.camera_id)
        camera_files = self.loader.list_camera_files(self.camera_id)

        lidar_map  = {f.stem: f for f in lidar_files}
        camera_map = {f.stem: f for f in camera_files}
        common_stems = sorted(set(lidar_map.keys()) & set(camera_map.keys()))

        if max_frames is not None:
            common_stems = common_stems[:max_frames]

        fused_frames = []
        for i, stem in enumerate(common_stems):
            print(f"  Frame {i + 1}/{len(common_stems)}: {stem}")
            fused = self.process_frame(lidar_map[stem], camera_map[stem], frame_id=i)
            fused_frames.append(fused)

        return fused_frames
