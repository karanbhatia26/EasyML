from torch import nn
from collections import deque, namedtuple
import random


Experience = namedtuple('Experience', 
                        ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity, parent_model=None):
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)
        self.parent_model = parent_model
        # Check if parent_model exists and has state_processor attribute
        self.state_processor = parent_model.state_processor if (parent_model and hasattr(parent_model, 'state_processor')) else None
        
    def push(self, state, action, reward, next_state, done):
        """Store transition in replay buffer with dimension change detection"""
        # Process states if processor exists
        if self.state_processor:
            state = self.state_processor.preprocess(state)
            next_state = self.state_processor.preprocess(next_state)
        
        # Store the experience (without dimension checks since preprocessor handles it)
        experience = Experience(state, action, reward, next_state, done)
        self.memory.append(experience)
        
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