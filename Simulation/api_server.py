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

from fastapi.staticfiles import StaticFiles
try:
    from .data_loader import get_scenario_data, get_normalized_map
except (ImportError, ValueError):
    from data_loader import get_scenario_data, get_normalized_map

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

# --- Visualisation & Scenario Endpoints ---

VAL_DIR = os.path.join(BASE_DIR, "val")

@app.get("/scenarios")
async def list_scenarios():
    """Returns a list of scenario IDs from the validation directory."""
    if not os.path.exists(VAL_DIR):
        return []
    return [d for d in os.listdir(VAL_DIR) if os.path.isdir(os.path.join(VAL_DIR, d))]

@app.get("/evaluate/{scenario_id}")
async def evaluate_scenario(scenario_id: str):
    """
    Combines data loading, map processing, and model prediction.
    Used by the frontend dashboard.
    """
    scenario_path = os.path.join(VAL_DIR, scenario_id)
    if not os.path.exists(scenario_path):
        raise HTTPException(status_code=404, detail="Scenario not found")
    
    # Load and Preprocess Data
    data = get_scenario_data(scenario_path)
    if data is None:
        raise HTTPException(status_code=500, detail="Error processing scenario data")
    
    # Run Prediction
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    with torch.no_grad():
        # Input shape (20, 5) -> (1, 20, 5)
        history_tensor = torch.tensor(data["history"], dtype=torch.float32).unsqueeze(0).to(device)
        preds = model(history_tensor)
        
    # Get Map geometry
    map_geom = get_normalized_map(scenario_path, data["origin"])
    
    return {
        "scenario_id": scenario_id,
        "history": data["history"].tolist(),
        "future_gt": data["future_gt"].tolist(),
        "prediction": preds.cpu().numpy().tolist(), # (1, 30, 2, 3)
        "safety_envelope": q_horizon.tolist(),
        "map": map_geom,
        "origin": data["origin"].tolist()
    }

# Mount Static Files for Dashboard
STATIC_PATH = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(STATIC_PATH):
    app.mount("/", StaticFiles(directory=STATIC_PATH, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
