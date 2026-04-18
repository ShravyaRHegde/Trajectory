# KASMU Trajectory Prediction & Simulation System

A high-performance system for vehicle trajectory prediction using the **KASMU v4** (Kinematic-Aware Safety Monitoring Unit) architecture. This repository provides a complete pipeline from raw validation data processing to interactive visualization.

## 🚀 System Architecture

- **KASMU v4 Brain**: LSTM-GRU architecture optimized for C2 continuity and 0.14 m/s³ jerk limits.
- **REST API**: FastAPI server with Intel XPU (IPEX) hardware acceleration.
- **Simulation Engine**: Automated data loader supporting scenario normalization and feature extraction.
- **Dashboard**: Web-based interactive visualizer for map geometry and trajectory overlays.

## 📦 Installation

1. Create a virtual environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

## 🛠 Usage

### 1. Start the API & Dashboard
```powershell
python Simulation/api_server.py
```
Visit [http://localhost:8000](http://localhost:8000) for the interactive visualizer.

### 2. Run Static Visualization Script
Generates high-quality PNGs of model predictions:
```powershell
python Simulation/run_visualizer.py
```

## 🧪 API Endpoints

- `GET /health`: System status and hardware acceleration info.
- `GET /verify`: Runs a kinematic integrity check on dummy data.
- `POST /predict`: Real-time inference on provided history tensors.
- `GET /scenarios`: Lists available validation scenarios.
- `GET /evaluate/{id}`: Detailed evaluation including map geometry and ground truth comparison.

## 📝 Preprocessing Logic
All data is normalized using the **index 19 (t=2.0s) origin shift** to ensure the model focuses on local displacement relative to the decision point. Features include `[X, Y, VX, VY, Lane_Count]`.