import os
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
try:
    from .model import KASMUModel_v4
except (ImportError, ValueError):
    from model import KASMUModel_v4

# --- Intel & Hardware Acceleration ---
try:
    import intel_extension_for_pytorch as ipex
    HAS_IPEX = True
except ImportError:
    HAS_IPEX = False

# Device Detection: XPU -> CUDA -> CPU
if torch.xpu.is_available():
    device = torch.device("xpu")
    ACCELERATION = "Intel XPU (Native)"
    if HAS_IPEX:
        ACCELERATION = "Intel XPU (IPEX Optimized)"
elif torch.cuda.is_available():
    device = torch.device("cuda")
    ACCELERATION = "NVIDIA CUDA"
else:
    device = torch.device("cpu")
    ACCELERATION = "CPU"

# --- API Configuration ---
app = FastAPI(
    title="KASMU v4 Trajectory Prediction API",
    description=f"REST API serving the KASMU v4 Brain with {ACCELERATION} Acceleration.",
    version="1.1.0"
)

# --- Path Resolution ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "Model", "kasmu_v4_weights.pth")
SAFETY_KEY_PATH = os.path.join(BASE_DIR, "Model", "q_horizon_v4.npy")

# --- Model Lifecycle ---
model = None
q_horizon = None

@app.on_event("startup")
async def load_resources():
    global model, q_horizon
    try:
        # Load Model
        model = KASMUModel_v4().to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        
        # Apply Intel Optimizations if ipex is available
        if HAS_IPEX:
            model = ipex.optimize(model, dtype=torch.float32)
            print(f"IPEX Optimization Applied (Device: {device})")
        else:
            # For native XPU, we can still use torch.compile for optimization
            try:
                # model = torch.compile(model) # Optional: torch.compile works well on XPU
                # print("🔥 Torch Compile applied for XPU acceleration.")
                pass
            except:
                pass
        
        # Load Safety Key
        q_horizon = np.load(SAFETY_KEY_PATH)
        
        print(f"KASMU v4 Safety Controller is LIVE on {device} ({ACCELERATION})")
    except Exception as e:
        print(f"Failed to load model resources: {e}")
        raise RuntimeError(f"Model initialization failed: {e}")

# --- Data Models ---
class HistoryRequest(BaseModel):
    # Expecting shape (Batch, 20, 5) flattened or as nested list
    # For simplicity, we'll accept a nested list: [[[x, y, vx, vy, lane], ...], ...]
    data: List[List[List[float]]]

class PredictionResponse(BaseModel):
    # Expecting shape (Batch, Horizon, 2, 3) -> 4D list
    median_trajectory: List[List[List[List[float]]]]
    safety_envelope: List[float]
    status: str

# --- Endpoints ---
@app.get("/health")
async def health_check():
    return {"status": "online", "device": str(device), "model": "KASMU_v4"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: HistoryRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input to tensor
        history_tensor = torch.tensor(request.data, dtype=torch.float32).to(device)
        
        # Validate shape (B, 20, 5)
        if len(history_tensor.shape) != 3 or history_tensor.shape[1] != 20 or history_tensor.shape[2] != 5:
            raise HTTPException(status_code=400, detail=f"Invalid input shape {history_tensor.shape}. Expected (B, 20, 5)")

        with torch.no_grad():
            preds = model(history_tensor)
            
        # Extract median (P50) across the 3 heads for the first batch item (or all)
        # Assuming we return the full prediction tensor and the conformal key
        return {
            "median_trajectory": preds.cpu().numpy().tolist(),
            "safety_envelope": q_horizon.tolist(),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.get("/verify")
async def verify_integrity():
    """
    Runs the 'State Check' on a dummy scenario to confirm physics synchronization.
    """
    try:
        # Mock history: Car cruising at 20 m/s in a 10-lane scenario
        dummy_history = torch.zeros((1, 20, 5)).to(device)
        dummy_history[0, :, 2] = 20.0 # vx
        dummy_history[0, :, 4] = 10.0 # lane count
        
        with torch.no_grad():
            preds = model(dummy_history)[0].cpu().numpy()
            
        # Extract Median Path (P50) - using head index 1 as per original logic
        median = preds[:, :, 1]
        
        # Calculate Instantaneous Jerk (3rd Derivative)
        vel = np.diff(median, axis=0) / 0.1
        acc = np.diff(vel, axis=0) / 0.1
        jerk = np.linalg.norm(np.diff(acc, axis=0) / 0.1, axis=1)
        
        avg_jerk = np.mean(jerk)
        sync_status = "Physics Engine Synchronized (Limousine Mode)" if avg_jerk < 0.5 else "Kinematic Drift Detected"
        
        return {
            "integrity_check": "passed" if avg_jerk < 0.5 else "warning",
            "average_jerk": f"{avg_jerk:.4f} m/s³",
            "ribbon_confidence_3s": f"±{q_horizon[-1]:.3f}m",
            "status_message": sync_status
        }
    except Exception as e:
        return {"integrity_check": "failed", "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
