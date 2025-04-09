from torch import nn
from collections import deque, namedtuple
import random


Experience = namedtuple('Experience', 
                        ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity, parent_model=None):
        self.memory = deque(maxlen=capacity)
        self.parent_model = parent_model  # Add reference to parent model
        
    def push(self, *args):
        """Store transition in replay buffer with dimension change detection"""
        # Extract the state and next_state from args (follows Experience namedtuple order)
        state, action, reward, next_state, done = args
        
        # Check if state dimensions have changed
        if state is not None and next_state is not None and len(self.memory) > 0:
            prev_state = self.memory[0].state
            
            # Handle numpy arrays or torch tensors
            if hasattr(prev_state, 'shape') and hasattr(state, 'shape'):
                if prev_state.shape != state.shape:
                    print(f"State dimension changed from {prev_state.shape} to {state.shape}")
                    self.memory.clear()
                    # Notify parent model if available
                    if self.parent_model and hasattr(self.parent_model, 'rebuild_networks'):
                        new_dim = state.shape[0] if len(state.shape) > 0 else 1
                        self.parent_model.rebuild_networks(new_dim)
            
            # Handle lists or other sequence types
            elif hasattr(prev_state, '__len__') and hasattr(state, '__len__'):
                if len(prev_state) != len(state):
                    print(f"State dimension changed from {len(prev_state)} to {len(state)}")
                    self.memory.clear()
                    # Notify parent model if available
                    if self.parent_model and hasattr(self.parent_model, 'rebuild_networks'):
                        self.parent_model.rebuild_networks(len(state))
        
        # Store the experience
        self.memory.append(Experience(*args))
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

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