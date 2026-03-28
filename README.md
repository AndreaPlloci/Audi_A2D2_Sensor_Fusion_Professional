# Audi A2D2: Professional Sensor Fusion Pipeline 🚗📡

This repository implements a **professional-grade sensor fusion pipeline** for the [Audi A2D2 Autonomous Driving Dataset](https://www.a2d2.audi). It integrates multi-modal perception data — **LiDAR point clouds**, **front/side cameras**, and **vehicle bus signals** — into a unified, AI-enriched analysis environment, orchestrated via n8n automation.

The system is designed to process, fuse, and semantically annotate raw A2D2 sensor frames for downstream tasks such as 3D object detection, lane segmentation, and predictive driving analytics.

---

## 🏗️ System Architecture

### 1. Data Ingestion & Preprocessing (`sensor-fusion/`)
The ingestion layer handles raw A2D2 data extraction and alignment across all sensor modalities.
* **LiDAR Processing:** Parses `.npz` point cloud files, applies ego-motion correction, and projects 3D points onto camera planes for cross-modal alignment.
* **Camera Pipeline:** Reads undistorted RGB frames from all 6 cameras (front, front-left, front-right, rear, rear-left, rear-right) and extracts semantic features using pretrained backbones.
* **Vehicle Bus Signals:** Decodes `bus_signals` JSON (speed, acceleration, steering angle) to contextualize sensor readings with vehicle dynamics.
* **Temporal Synchronization:** Aligns sensor timestamps across modalities to construct coherent multi-sensor frames.

### 2. AI Fusion & Annotation Orchestration (`n8n-workflows/`)
An n8n pipeline orchestrates the AI analysis layer, enabling automated semantic enrichment of fused sensor data.
* **Scene Understanding:** Triggers AI inference on fused frames for 3D bounding box estimation and semantic segmentation.
* **Anomaly Detection:** Flags sensor inconsistencies (e.g., LiDAR/camera depth mismatches) and logs them to a MariaDB instance for audit trails.
* **Report Generation:** Produces structured JSON reports per scene, including detected objects, confidence scores, and driving context annotations.

### 3. Infrastructure & Data Sovereignty
* **Jarvis Server:** Processing runs on the private headless Windows Server node within the Jarvis ecosystem.
* **Security:** All data flows are encapsulated within the **Tailscale VPN Mesh** (Zero-Trust).
* **Persistence:** Fused scene metadata and inference results are stored on a local **MariaDB/MySQL** instance.

---

## 🛠️ Tech Stack

* **Sensor Processing:** Python, NumPy, Open3D (LiDAR), OpenCV (Camera), Pandas (Bus Signals).
* **AI & Inference:** n8n (Orchestration), OpenAI GPT-4o-mini, Pinecone (Vector DB for scene embeddings).
* **Infrastructure:** Windows Server, MariaDB/MySQL, Tailscale VPN, AdGuard Home.

---

## 📂 Project Structure

```text
├── sensor-fusion/           # Core Python Processing Pipeline
│   ├── ingestion/           # A2D2 data loading and parsing utilities
│   ├── lidar/               # Point cloud processing and projection
│   ├── camera/              # Image processing and feature extraction
│   └── fusion/              # Multi-modal alignment and fusion logic
├── n8n-workflows/           # AI Orchestration & Automation
│   ├── scene_analysis.json  # Sensor fusion trigger and AI annotation pipeline
│   └── anomaly_monitor.json # Sensor inconsistency detection and alerting
└── README.md
```

---
*Engineered for industrial-grade autonomous driving research within the Jarvis AI ecosystem.*
