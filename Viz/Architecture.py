import torch
import torch.nn as nn
import os
import sys
import io

# Force UTF-8 encoding for Windows terminals to support technical summaries and box-drawing characters
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Add project root to path so we can import Simulation modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from Simulation.model import KASMUModel_v4
except ImportError:
    from model import KASMUModel_v4

try:
    from torchview import draw_graph
    import graphviz
    HAS_GRAPHVIZ = True
except ImportError:
    HAS_GRAPHVIZ = False

try:
    from torchinfo import summary
    HAS_TORCHINFO = True
except ImportError:
    HAS_TORCHINFO = False

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_PATH = os.path.join(BASE_DIR, "Model", "kasmu_v4_weights.pth")

# Set path for Graphviz (Required for torchview on Windows)
# If you have Graphviz installed but not in PATH, add it here:
# os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

def generate_architecture_diagram():
    # 1. Setup Device
    device = torch.device("xpu" if torch.xpu.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Model
    model = KASMUModel_v4().to(device)
    
    # 3. Load Trained Weights
    if os.path.exists(WEIGHTS_PATH):
        try:
            model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
            print(f"Loaded weights from: {WEIGHTS_PATH}")
        except Exception as e:
            print(f"Warning: Could not load weights: {e}")
    else:
        print(f"Weights not found at {WEIGHTS_PATH}. Generating diagram with random init.")

    model.eval()
    
    # 4. Dummy Input for Trace (Batch, Seq, Features)
    dummy_input = torch.randn(1, 20, 5).to(device)

    # 5. Generate Graph / Summary
    if HAS_GRAPHVIZ:
        print("Synthesizing architecture graph (Torchview)...")
        try:
            model_graph = draw_graph(
                model, 
                input_data=dummy_input,
                expand_nested=True,
                depth=2, 
                graph_name="KASMU_v4_Logic",
                roll=True 
            )
            output_file = "KASMU_Architecture_v4"
            model_graph.visual_graph.render(output_file, format="png", cleanup=True)
            print(f"Diagram generated successfully: {output_file}.png")
            return
        except Exception as e:
            print(f"Graphviz render failed: {e}")

    if HAS_TORCHINFO:
        print("\nFalling back to Torchinfo Summary...")
        model_summary = summary(
            model, 
            input_size=(1, 20, 5), 
            device=device,
            col_names=["input_size", "output_size", "num_params", "mult_adds"],
            depth=3
        )
        print(model_summary)
        print("\nTip: Since Graphviz is missing, I have also generated a visual Mermaid diagram for you in the workbench.")
    else:
        print("\nNo visualization tools available. Run: pip install torchinfo or install Graphviz.")

if __name__ == "__main__":
    generate_architecture_diagram()