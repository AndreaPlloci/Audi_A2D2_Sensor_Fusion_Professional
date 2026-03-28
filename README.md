# Audi A2D2: Sensor Fusion Pipeline 🚗📡

Complete sensor fusion pipeline for autonomous driving using the **Audi A2D2 Dataset**. Compares three perception approaches — pure LiDAR, pure AI vision, and true multi-modal fusion — generating annotated output videos with Bird’s Eye View (BEV) mapping and performance analytics.

Designed to run on both **Google Colab** (GPU, auto dataset mount) and **local machine** (NAS path via `config.txt`).

---

## 🏗️ Pipeline Architecture

### Approach 1 — LiDAR Sensors Only (DBSCAN + Tracking + BEV)
Pure LiDAR-based detection without any AI model.
* **Clustering:** DBSCAN algorithm segments the point cloud into discrete objects.
* **Tracking:** Hungarian algorithm maintains persistent object IDs across frames.
* **BEV Map:** Bird’s Eye View overlay reconstructed from LiDAR projections.
* **Output:** `Video1_PRO_Solo_Sensori.mp4`

### Approach 2 — YOLO Only (AI Detection + ByteTrack + BEV)
Pure camera-based AI detection without LiDAR depth data.
* **Detection:** YOLOv8 (medium) per-frame object detection.
* **Tracking:** ByteTrack for persistent multi-object temporal tracking.
* **BEV Map:** Estimated spatial positions derived from 2D bounding boxes.
* **Output:** `Video2_PRO_YOLO_ByteTrack.mp4`

### Approach 3 — True Sensor Fusion (YOLO + LiDAR)
Multi-modal perception combining AI semantics with LiDAR spatial accuracy.
* **Detection:** YOLO provides stable 2D bounding boxes and class labels.
* **Depth:** LiDAR point cloud projected onto the camera frame for precise distance estimation.
* **Validation:** LiDAR clusters validate and refine YOLO detections in 3D space.
* **Output:** `Video3_PRO_Sensor_Fusion.mp4`

### Comparative Analysis
* Temporal evolution graphs for all three approaches.
* Tracking stability and object count comparison.
* LiDAR validation confidence metrics.
* Processing time benchmarks per frame.

---

## 🛠️ Tech Stack

* **Core:** Python 3.10+, Jupyter Notebook, NumPy, OpenCV.
* **AI & Detection:** YOLOv8 (Ultralytics), ByteTrack, PyTorch.
* **LiDAR Processing:** DBSCAN (scikit-learn), Hungarian algorithm (scipy).
* **Visualization:** Matplotlib, tqdm.
* **Infrastructure:** Google Colab (GPU), NAS local storage, Google Drive sync.

---

## 📂 Project Structure

```text
├── Audi_A2D2_Sensor_Fusion_Professional.ipynb   # Complete pipeline
└── README.md
```

---

## ⚙️ Setup

**Local:**
1. Ensure dataset at `YOUR_NAS_PATH\Progetto Audi A2D2` (or create `config.txt` with custom path)
2. Run notebook cells sequentially

**Google Colab:**
1. Open notebook in Colab
2. Mount Google Drive when prompted
3. Add dataset folder shortcut to `MyDrive/Audi_A2D2_Data/`
4. Run all cells

---
*Part of the Jarvis AI ecosystem for autonomous driving research.*
