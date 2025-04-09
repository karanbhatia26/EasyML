import torch
from torch import nn

class DQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQNetwork, self).__init__()
        
        print(f"Creating DQNetwork with input={state_dim}, output={action_dim}")
        
        # Use a more robust network architecture
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        
        # Print layer shapes for debugging
        print(f"DQNetwork layers: {state_dim}→256→256→{action_dim}")
        
    def forward(self, x):
        return self.network(x)