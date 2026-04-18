import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import random
import argparse

try:
    from model import KASMUModel_v4
    from data_loader import get_scenario_data, get_normalized_map, get_surrounding_objects
except ImportError:
    from .model import KASMUModel_v4
    from .data_loader import get_scenario_data, get_normalized_map, get_surrounding_objects

"""
Standalone Trajectory Visualizer
Generates static PNG plots of KASMU v4 trajectory predictions against ground truth.
Includes map geometry, historical paths, and shaded safety envelopes.
Supports batch generation and seeded dataset splitting.
"""

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VAL_DIR = os.path.join(BASE_DIR, "val")
MODEL_PATH = os.path.join(BASE_DIR, "Model", "kasmu_v4_weights.pth")
SAFETY_KEY_PATH = os.path.join(BASE_DIR, "Model", "q_horizon_v4.npy")

def run_visualizer(scenario_id=None, output_name=None):
    # 1. Setup Device
    device = torch.device("xpu" if torch.xpu.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    # 2. Load Model & Weights
    model = KASMUModel_v4().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    q_horizon = np.load(SAFETY_KEY_PATH)

    # 3. Select Scenario if not provided
    if not scenario_id:
        scenarios = sorted([d for d in os.listdir(VAL_DIR) if os.path.isdir(os.path.join(VAL_DIR, d))])
        scenario_id = random.choice(scenarios)
    
    scenario_path = os.path.join(VAL_DIR, scenario_id)

    # 4. Load Data
    data = get_scenario_data(scenario_path)
    if not data:
        return None

    # 5. Predict
    with torch.no_grad():
        history_tensor = torch.tensor(data["history"], dtype=torch.float32).unsqueeze(0).to(device)
        preds = model(history_tensor)
        p50 = preds[0, :, :, 1].cpu().numpy() 

    # 6. Load Map & Surrounding Objects
    map_geom = get_normalized_map(scenario_path, data["origin"])
    surroundings = get_surrounding_objects(scenario_path, data["origin"])

    # 7. Plotting
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor('#0d0d0f')
    fig.patch.set_facecolor('#0d0d0f')

    # Draw Map
    if map_geom:
        for poly in map_geom["drivable_areas"]:
            ax.add_patch(Polygon(poly, closed=True, color='#1c1c1f', alpha=0.9, zorder=1))
        for lane in map_geom["lanes"]:
            pts = np.array(lane["centerline"])
            ax.plot(pts[:, 0], pts[:, 1], color='#333336', linestyle='--', linewidth=0.5, zorder=2)

    # NEW: Draw Surrounding Objects with Annotations
    for obj in surroundings:
        color = '#4a4a4d' # default (other vehicles)
        label_text = "CAR"
        if obj['type'] == 'pedestrian': 
            color = '#38bdf8'
            label_text = "PED"
        elif obj['type'] in ['motorcyclist', 'cyclist']: 
            color = '#f59e0b'
            label_text = "CYC"
        
        ax.scatter(obj['pos'][0], obj['pos'][1], color=color, s=45, marker='s', alpha=0.7, zorder=4)
        
        # Add Tag
        ax.text(obj['pos'][0] + 0.8, obj['pos'][1] + 0.8, label_text, 
                color=color, fontsize=7, fontweight='bold', alpha=0.9,
                bbox=dict(facecolor='black', alpha=0.4, edgecolor='none', pad=1),
                zorder=10)

    # Draw Focal Vehicle States
    hist = data["history"]
    ax.plot(hist[:, 0], hist[:, 1], color='#38bdf8', linewidth=2.8, alpha=0.9, label='History', zorder=5)
    
    gt = data["future_gt"]
    ax.plot(gt[:, 0], gt[:, 1], color='#10b981', linestyle=':', linewidth=2, label='Ground Truth', zorder=5)

    # Draw Safety Envelope & Prediction
    horizon = p50.shape[0]
    envelope_pts = []
    for i in range(horizon):
        q = q_horizon[i]
        envelope_pts.append([p50[i, 0] + q, p50[i, 1]])
    for i in range(horizon - 1, -1, -1):
        q = q_horizon[i]
        envelope_pts.append([p50[i, 0] - q, p50[i, 1]])
    
    ax.add_patch(Polygon(envelope_pts, closed=True, color='#d946ef', alpha=0.12, zorder=3))
    ax.plot(p50[:, 0], p50[:, 1], color='#d946ef', linewidth=4.0, label='KASMU Prediction', zorder=6)

    # Focal Point
    ax.scatter(0, 0, color='white', s=100, edgecolors='black', linewidth=1.5, zorder=15)

    # Styling
    ax.set_aspect('equal')
    ax.axis('off')

    # Watermark / Metadata
    ax.text(0.02, 0.02, f"SCENARIO: {scenario_id}\nKASMU v4 SAFETY SYSTEM", 
            transform=ax.transAxes, color='#333336', fontsize=8, family='monospace',
            verticalalignment='bottom', alpha=0.8)
    
    # Save
    if not output_name:
        output_name = f"prediction_{scenario_id[:8]}.png"
    
    output_path = os.path.join(BASE_DIR, "Visualisations", output_name)
    plt.savefig(output_path, dpi=120, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    return output_name, surroundings

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate KASMU v4 Simulation Plots")
    parser.add_argument("-n", "--num", type=int, default=1, help="Number of simulations to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data partitioning")
    parser.add_argument("--skip", type=int, default=10000, help="Number of samples to discard (training set)")
    args = parser.parse_args()

    # Get Scenarios
    all_scenarios = sorted([d for d in os.listdir(VAL_DIR) if os.path.isdir(os.path.join(VAL_DIR, d))])
    
    # Apply Seeded Partitioning
    random.seed(args.seed)
    random.shuffle(all_scenarios)
    
    # Selection Pool
    test_scenarios = all_scenarios[args.skip:]
    
    samples = []
    pool_indices = list(range(len(test_scenarios)))
    random.shuffle(pool_indices)
    
    idx = 0
    while len(samples) < args.num and idx < len(pool_indices):
        sid = test_scenarios[pool_indices[idx]]
        idx += 1
        
        try:
            p = os.path.join(VAL_DIR, sid)
            data = get_scenario_data(p)
            gt = data["future_gt"]
            dist = np.linalg.norm(gt[-1] - gt[0])
            if dist > 35 and random.random() < 0.7:  # Skip erratic/extreme outliers 70% of time
                continue
        except: pass
        
        samples.append(sid)
    
    if not samples: samples = random.sample(test_scenarios, args.num)
    
    for i, sid in enumerate(samples):
        print(f"[{i+1}/{args.num}] Processing {sid}...")
        fname, _ = run_visualizer(scenario_id=sid)
        print(f"Saved: {fname}")
