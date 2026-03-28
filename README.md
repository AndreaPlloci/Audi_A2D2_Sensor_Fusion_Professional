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
* **Semantic Segmentation:** Per-pixel class labeling using the A2D2 semantic label taxonomy (38 classes).
* **Feature Extraction:** Extracts visual embeddings from RGB frames using pretrained CNN backbones for cross-modal association.
* **Bounding Box Parsing:** Reads 2D ground truth annotations from A2D2 label files.

### 4. Sensor Fusion (`sensor-fusion/fusion/`)
Core multi-modal alignment and fusion logic.
* **Temporal Synchronization:** Aligns LiDAR, camera, and bus signal timestamps to build coherent multi-sensor frames.
* **LiDAR-Camera Association:** Associates projected LiDAR points with semantic image labels for dense 3D scene annotation.
* **3D Scene Reconstruction:** Builds fused 3D scene representations combining depth (LiDAR) and semantics (camera).

---

## 🛠️ Tech Stack

* **Core:** Python 3.10+, NumPy, Pandas.
* **LiDAR:** Open3D (point cloud processing, visualization, voxel ops).
* **Camera:** OpenCV (image processing, projection, annotation rendering).
* **Visualization:** Matplotlib, Open3D visualizer.
* **Data:** Audi A2D2 Dataset (LiDAR + Camera + Bus Signals splits).

---

## 📂 Project Structure

```text
├── sensor-fusion/           # Core Python Processing Pipeline
│   ├── ingestion/           # A2D2 data loading and parsing utilities
│   ├── lidar/               # Point cloud processing and camera projection
│   ├── camera/              # Image processing, segmentation, feature extraction
│   └── fusion/              # Temporal sync, LiDAR-camera association, 3D fusion
└── README.md
```

---
*Engineered for professional autonomous driving research on the Audi A2D2 dataset.*
