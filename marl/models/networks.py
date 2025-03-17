import torch
from torch import nn
class DQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DQNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, x):
        return self.network(x)