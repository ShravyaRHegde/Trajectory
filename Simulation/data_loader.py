import os
import json
import numpy as np
import pandas as pd
from glob import glob
from typing import Dict, Any, List, Optional

def get_scenario_data(scenario_folder: str):
    """
    Implements normalization and feature extraction for a single scenario.
    
    This matches the specific ZF-KASMU hackathon preprocessing requirements:
    1. Origin Shift: Coordinates are normalized relative to the current position 
       at index 19 (t=2.0s).
    2. Vector Layout: [x, y, vx, vy, lane_count].
    3. Windowing: 50 steps total (20 history, 30 future).
    
    Args:
        scenario_folder (str): Absolute path to the folder containing parquet and json files.
        
    Returns:
        dict: Processed tensors and metadata:
            - "history": np.ndarray (20, 5) - Encoder input
            - "future_gt": np.ndarray (30, 2) - Evaluation ground truth
            - "origin": np.ndarray [x, y] - Translation vector used
            - "lane_count": int - Environmental context feature
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

def get_surrounding_objects(scenario_folder: str, origin: np.ndarray):
    """
    Extracts all actors in the scene at t=2.0s (index 19).
    """
    try:
        parquet_path = glob(os.path.join(scenario_folder, "*.parquet"))[0]
        df = pd.read_parquet(parquet_path)
        focal_id = df['focal_track_id'].iloc[0]
        
        # Get all objects at index 19
        snapshot = df[df['timestep'] == 19]
        
        objects = []
        for _, row in snapshot.iterrows():
            if row['track_id'] == focal_id:
                continue
                
            objects.append({
                "id": row['track_id'],
                "type": row['object_type'],
                "pos": [row['position_x'] - origin[0], row['position_y'] - origin[1]],
                "heading": row['heading']
            })
        return objects
    except Exception as e:
        print(f"Error extracting surrounding objects: {e}")
        return []
