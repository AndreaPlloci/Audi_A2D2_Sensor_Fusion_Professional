# Audi A2D2: Professional Sensor Fusion Pipeline 🚗📡

This repository implements a **professional-grade sensor fusion pipeline** for the [Audi A2D2 Autonomous Driving Dataset](https://www.a2d2.audi). It integrates multi-modal perception data — **LiDAR point clouds**, **front/side cameras**, and **vehicle bus signals** — into a unified Python processing environment for autonomous driving research.

The system processes, aligns, and semantically annotates raw A2D2 sensor frames for downstream tasks such as 3D object detection, lane segmentation, and scene understanding.

---

## 🏗️ System Architecture

### 1. Data Ingestion & Parsing (`sensor-fusion/ingestion/`)
Handles raw A2D2 data loading and decoding across all sensor modalities.
* **LiDAR:** Parses `.npz` point cloud files and extracts 3D coordinates, reflectance, and row/column indices.
* **Camera:** Loads undistorted RGB frames from all 6 cameras (front, front-left, front-right, rear, rear-left, rear-right) with calibration matrices.
* **Vehicle Bus Signals:** Decodes `bus_signals` JSON fields (speed, acceleration, steering angle) to contextualize sensor readings with vehicle dynamics.

### 2. LiDAR Processing (`sensor-fusion/lidar/`)
Dedicated pipeline for 3D point cloud processing and cross-modal projection.
* **Ego-motion Correction:** Compensates for vehicle movement between LiDAR sweeps using IMU data.
* **Camera Projection:** Projects 3D LiDAR points onto 2D camera image planes using intrinsic/extrinsic calibration matrices.
* **Filtering & Downsampling:** Voxel grid downsampling and range-based filtering to reduce noise and computational load.

### 3. Camera Processing (`sensor-fusion/camera/`)
Image-based perception pipeline for visual feature extraction.
* **Calibration Management:** Loads and applies camera intrinsic/extrinsic parameters for all 6 camera views.
* **LiDAR Overlay:** Renders projected LiDAR depth maps onto camera images with colormap-based depth encoding.
* **Semantic Label Parsing:** Reads A2D2 semantic bounding box annotations from JSON label files.

### 4. Sensor Fusion (`sensor-fusion/fusion/`)
Core multi-modal alignment and fusion logic.
* **Temporal Synchronization:** Aligns LiDAR, camera, and bus signal timestamps within a configurable tolerance window.
* **LiDAR-Camera Association:** Associates projected 3D points with image pixels for dense depth-annotated frames.
* **3D Scene Reconstruction:** Aggregates multi-frame point clouds into a unified 3D scene exportable as `.ply`.

---

## 🛠️ Tech Stack

* **Core:** Python 3.10+, NumPy, Pandas.
* **LiDAR:** Open3D (point cloud processing, voxel downsampling, visualization, `.ply` export).
* **Camera:** OpenCV (image loading, undistortion, LiDAR overlay rendering).
* **Data:** Audi A2D2 Dataset — Camera + LiDAR + Bus Signals splits.

---

## 📂 Project Structure

```text
├── sensor-fusion/
│   ├── config.py                  # Percorsi dataset e parametri pipeline
│   ├── main.py                    # Entry point della pipeline
│   ├── ingestion/
│   │   ├── a2d2_loader.py         # Caricamento LiDAR (.npz), camera (.png) e calibrazione
│   │   └── bus_parser.py          # Parser segnali CAN bus veicolo
│   ├── lidar/
│   │   ├── point_cloud.py         # Filtraggio, downsampling e preprocessing point cloud
│   │   └── projection.py          # Proiezione LiDAR → piano immagine camera
│   ├── camera/
│   │   ├── calibration.py         # Gestione calibrazione intrinseca/estrinseca
│   │   └── image_processor.py     # Overlay LiDAR-camera e parsing annotazioni semantiche
│   └── fusion/
│       ├── fusion_engine.py       # Orchestratore principale della pipeline di fusione
│       └── scene_builder.py       # Aggregazione multi-frame e ricostruzione scena 3D
└── requirements.txt
```

---
*Engineered for professional autonomous driving research on the Audi A2D2 dataset.*
