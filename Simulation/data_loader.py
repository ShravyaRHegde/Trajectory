import os
import json
import numpy as np
import pandas as pd
from glob import glob
from typing import Dict, Any, List, Optional

def get_scenario_data(scenario_folder: str):
    """
    Implements the user-provided preprocessing logic for a scenario.
    Returns:
        history: (20, 5) tensor for model input
        future_gt: (30, 2) array for ground truth visualization
        origin: [x, y] used for normalization
        lane_count: int
    """
    try:
        parquet_path = glob(os.path.join(scenario_folder, "*.parquet"))[0]
        
        # 1. Load Motion Data
        df = pd.read_parquet(parquet_path)
        focal_id = df['focal_track_id'].iloc[0]
        agent = df[df['track_id'] == focal_id].sort_values('timestep')
        
        if len(agent) < 50: return None
        
        # 2. Coordinate Normalization (Shift to 0,0 at current timestep index 19)
        origin = agent.iloc[19][['position_x', 'position_y']].values.astype(float)
        
        # Normalize all positions
        pos = agent[['position_x', 'position_y']].values - origin
        vel = agent[['velocity_x', 'velocity_y']].values
        
        # 3. Load Lane Context (Simplified as per user logic)
        json_path = glob(os.path.join(scenario_folder, "*.json"))[0]
        with open(json_path, 'r') as f:
            map_data = json.load(f)
        lane_count = len(map_data.get('lane_segments', {}))
        
        # 4. Filter to 50 timesteps (20 history + 30 future)
        # Features: [X, Y, VX, VY, Lane_Count]
        motion_features = np.hstack([pos[:50], vel[:50]])
        map_features = np.full((50, 1), lane_count)
        combined_state = np.hstack([motion_features, map_features])
        
        # Split into History (0-19) and Future GT (20-49)
        history = combined_state[:20]  # (20, 5)
        future_gt = pos[20:50]          # (30, 2)
        
        return {
            "history": history,
            "future_gt": future_gt,
            "origin": origin,
            "lane_count": lane_count
        }
    except Exception as e:
        print(f"Error processing scenario {scenario_folder}: {e}")
        return None

def get_normalized_map(scenario_folder: str, origin: np.ndarray):
    """
    Loads and normalizes map geometry relative to the provided origin.
    """
    try:
        json_path = glob(os.path.join(scenario_folder, "*.json"))[0]
        with open(json_path, 'r') as f:
            map_data = json.load(f)
            
        def normalize_points(points):
            return [[p['x'] - origin[0], p['y'] - origin[1]] for p in points]

        # Process Drivable Areas
        drivable = []
        for da in map_data.get('drivable_areas', {}).values():
            drivable.append(normalize_points(da['area_boundary']))
            
        # Process Lanes
        lanes = []
        for ls in map_data.get('lane_segments', {}).values():
            lanes.append({
                "centerline": normalize_points(ls['centerline']),
                "left": normalize_points(ls['left_lane_boundary']),
                "right": normalize_points(ls['right_lane_boundary'])
            })
            
        return {
            "drivable_areas": drivable,
            "lanes": lanes
        }
    except Exception as e:
        print(f"Error processing map for {scenario_folder}: {e}")
        return None
