# KASMU Trajectory Prediction & Simulation System

A high-performance trajectory prediction workbench powered by **KASMU v4** (Kinematic-Aware Safety Monitoring Unit). This system provides an end-to-end laboratory environment for evaluating vehicle motion models with real-time XPU acceleration.

## ✨ Simulation Workbench (Pro Edition)

The system includes a premium, web-based Simulation Workbench ([http://localhost:8000](http://localhost:8000)) featuring:

- **Dynamic Scene Synthesis**: Selective object filtering where checking/unchecking entities triggers a high-speed re-render on the Intel XPU.
- **Interactive Inspection**: Full Pan & Zoom capabilities for detailed intersection analysis.
- **Smart Quality Guard**: Automated rejection sampling favoring high-confidence, professional scenarios.
- **Live Metadata Sidebar**: Deep-dive into surrounding traffic actor types and unique track IDs.
- **High-Res Export**: Native PNG downloads with scenario watermarking.

## 🚀 Hardware Acceleration

Optimized for **Intel XPU (Native)** using the Intel Extension for PyTorch (IPEX). 
- **Auto-Detection**: The server automatically benchmarks and selects the best available backend (XPU -> CUDA -> CPU).
- **Inference Latency**: Sub-10ms prediction heads for real-time safety envelope calculation.

## 📦 Installation

1. **Environment Setup**:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. **Run Workbench**:
   ```powershell
   python Simulation/api_server.py
   ```

## 🛠 Project Structure

- `Simulation/api_server.py`: FastAPI backend & Workbench host.
- `Simulation/run_visualizer.py`: Headless PNG generation & scene synthesis engine.
- `Simulation/data_loader.py`: Spatial normalization (2.0s anchor) and map extraction.
- `Simulation/model.py`: KASMU v4 LSTM-GRU architecture.
- `Visualisations/`: Managed synthesis cache (Auto-rolling cleanup).

## 📊 Visual Semantics

- **Cyan/Blue**: 2.0s Historical context.
- **Purple**: KASMU Predicted trajectory.
- **Green**: Ground Truth (Validation anchor).
- **Tags**: Automated labels for Cars, Pedestrians, and Cyclists.

---
*Powered by KASMU v4 Safety Controller.*