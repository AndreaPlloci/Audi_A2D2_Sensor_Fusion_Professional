"""
Configurazione centralizzata per la pipeline di sensor fusion A2D2.
Percorsi locali mascherati per protezione privacy.
"""

from pathlib import Path

# -----------------------------------------------------------------------
# Percorso radice del dataset A2D2 - mascherato per protezione privacy
# Sostituire con il percorso reale sul sistema locale prima dell'esecuzione
# -----------------------------------------------------------------------
A2D2_DATASET_ROOT = Path("YOUR_DATASET_PATH/a2d2")

# Struttura cartelle del dataset A2D2
LIDAR_DIR = A2D2_DATASET_ROOT / "camera_lidar"
BUS_SIGNAL_DIR = A2D2_DATASET_ROOT / "bus_signal"

# Directory di output per i risultati della fusione - mascherata per protezione privacy
OUTPUT_DIR = Path("YOUR_OUTPUT_PATH/sensor_fusion_output")

# Camera IDs disponibili nel dataset A2D2
CAMERA_IDS = [
    "cam_front_center",
    "cam_front_left",
    "cam_front_right",
    "cam_rear_center",
    "cam_side_left",
    "cam_side_right",
]

# Parametri LiDAR
LIDAR_VOXEL_SIZE = 0.1         # Risoluzione voxel grid in metri
LIDAR_MAX_RANGE  = 80.0        # Distanza massima punti LiDAR in metri
LIDAR_MIN_RANGE  = 1.0         # Distanza minima punti LiDAR in metri

# Parametri sincronizzazione temporale
SYNC_TOLERANCE_MS = 50         # Tolleranza temporale sincronizzazione in millisecondi
