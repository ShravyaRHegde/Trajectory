import torch
import torch.nn as nn
from torchview import draw_graph
import os

# Set path for Graphviz (Required for torchview)
# If you are on Windows and haven't added Graphviz to Path, uncomment below:
# os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

class KASMUModel_v4(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=256):
        super().__init__()
        # Hyperparameters baked in for the trace
        self.HORIZON = 30
        self.DT = 0.1
        
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.context_net = nn.Sequential(
            nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, hidden_dim * 2)
        )
        
        self.decoder_cell = nn.GRUCell(2, hidden_dim) 
        self.jerk_head = nn.Linear(hidden_dim, 6) 

    def forward(self, x):
        batch_size = x.size(0)
        lane_context = x[:, -1, 4:5]
        _, (h_n, _) = self.encoder(x)
        
        # FiLM Context Modulation
        gamma, beta = self.context_net(lane_context).chunk(2, dim=-1)
        h_t = (gamma * h_n[-1]) + beta
        
        curr_pos, curr_vel = x[:, -1, :2], x[:, -1, 2:4]
        curr_acc = torch.zeros_like(curr_vel) 
        
        predictions = []
        for _ in range(self.HORIZON):
            h_t = self.decoder_cell(curr_vel, h_t)
            delta_a = self.jerk_head(h_t).view(batch_size, 2, 3) 
            
            # Integration Chain
            new_acc = curr_acc.unsqueeze(-1) + delta_a
            new_vel = curr_vel.unsqueeze(-1) + (new_acc * self.DT)
            new_pos = curr_pos.unsqueeze(-1) + (new_vel * self.DT)
            predictions.append(new_pos)
            
            # Recursive updates
            curr_pos, curr_vel, curr_acc = new_pos[:,:,1], new_vel[:,:,1], new_acc[:,:,1]
            
        return torch.stack(predictions, dim=1)

# --- Execution ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = KASMUModel_v4().to(device)
dummy_input = torch.randn(1, 20, 5).to(device)

# Using depth=1 is often cleaner for LSTMs to avoid showing every internal gate
model_graph = draw_graph(
    model, 
    input_data=dummy_input,
    expand_nested=True,
    depth=2, 
    graph_name="KASMU_v4_Logic",
    roll=True 
)

# Render
model_graph.visual_graph.render("KASMU_Architecture_v4", format="png")
print("✅ Diagram generated successfully!")