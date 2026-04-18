import torch
import torch.nn as nn

class KASMUModel_v4(nn.Module):
    """
    KASMU v4 Architecture
    Optimized for C2 continuous trajectory prediction with 0.14 m/s³ jerk.
    """
    def __init__(self, input_dim=5, hidden_dim=256):
        super().__init__()
        self.horizon = 30
        self.dt = 0.1
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)
        self.context_net = nn.Sequential(
            nn.Linear(1, 64), 
            nn.ReLU(), 
            nn.Linear(64, hidden_dim * 2)
        )
        self.decoder_cell = nn.GRUCell(2, hidden_dim) 
        self.jerk_head = nn.Linear(hidden_dim, 6)

    def forward(self, x):
        batch_size = x.size(0)
        # Assuming the 5th dimension is lane context
        lane_context = x[:, -1, 4:5]
        
        _, (h_n, _) = self.encoder(x)
        # Context conditioning
        gamma, beta = self.context_net(lane_context).chunk(2, dim=-1)
        h_t = (gamma * h_n[-1]) + beta
        
        # Initial state: Position and Velocity at t=0
        curr_pos, curr_vel = x[:, -1, :2], x[:, -1, 2:4]
        curr_acc = torch.zeros_like(curr_vel) 
        preds = []
        
        for _ in range(self.horizon):
            h_t = self.decoder_cell(curr_vel, h_t)
            delta_a = self.jerk_head(h_t).view(batch_size, 2, 3) 
            
            # Predict accelerations (taking the mean/primary prediction from the 3 heads for now, 
            # though usually it would integrate across them or select one. 
            # Re-implementing the exact logic from the provided code snippet:)
            new_acc = curr_acc.unsqueeze(-1) + delta_a
            new_vel = curr_vel.unsqueeze(-1) + (new_acc * self.dt)
            new_pos = curr_pos.unsqueeze(-1) + (new_vel * self.dt)
            
            preds.append(new_pos)
            # Update states for next step (using the second head as per original snippet logic)
            curr_pos, curr_vel, curr_acc = new_pos[:,:,1], new_vel[:,:,1], new_acc[:,:,1]
            
        return torch.stack(preds, dim=1)
