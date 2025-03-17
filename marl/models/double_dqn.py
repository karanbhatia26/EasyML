import torch
from torch import nn, optim
from ..utils.buffer import ReplayBuffer, Experience
from .networks import DQNetwork
import random
import numpy as np

class DoubleDQN:
    def __init__(self, state_dim, action_dim, lr=1e-4, gamma=0.99, 
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, 
                 buffer_size=10000, batch_size=64, update_freq=10):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Set device for GPU acceleration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Double DQN networks - move to GPU immediately
        self.policy_net = DQNetwork(state_dim, action_dim).to(self.device)
        self.target_net = DQNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network doesn't need gradients
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.update_counter = 0
        self.update_freq = update_freq
        
    def select_action(self, state, valid_actions=None):
        if valid_actions is None or len(valid_actions) == 0:
            return -1
            
        if random.random() > self.epsilon:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                
                if valid_actions:
                    valid_q = q_values[0][valid_actions]
                    best_valid_idx = valid_q.argmax().item()
                    return valid_actions[best_valid_idx]
                return q_values.argmax(1).item()
        else:
            return random.choice(valid_actions)
    
    def store_experience(self, state, action, reward, next_state, done):
        if action == -1:
            return
        self.buffer.push(state, action, reward, next_state, done)

    def update_model(self):
        if len(self.buffer) < self.batch_size:
            return
        
        experiences = self.buffer.sample(self.batch_size)
        states = torch.FloatTensor(np.array([exp.state for exp in experiences])).to(self.device)
        actions = torch.LongTensor(np.array([exp.action for exp in experiences])).to(self.device)
        rewards = torch.FloatTensor(np.array([exp.reward for exp in experiences])).to(self.device)
        next_states = torch.FloatTensor(np.array([exp.next_state for exp in experiences])).to(self.device)
        dones = torch.FloatTensor(np.array([exp.done for exp in experiences])).to(self.device)
        
        if (actions < 0).any():
            valid_mask = actions >= 0
            if not valid_mask.any():
                return  # No valid actions in this batch

            states = states[valid_mask]
            actions = actions[valid_mask]
            rewards = rewards[valid_mask]
            next_states = next_states[valid_mask]
            dones = dones[valid_mask]
        
        next_q_values = self.policy_net(next_states)
        next_actions = next_q_values.max(1)[1].unsqueeze(1)  # Shape: [batch_size, 1]
        target_q_values = self.target_net(next_states).gather(1, next_actions)
        rewards = rewards.unsqueeze(1)  # Shape: [batch_size, 1]
        dones = dones.unsqueeze(1)      # Shape: [batch_size, 1]
        target_q = rewards + (1 - dones) * self.gamma * target_q_values
        actions = actions.unsqueeze(1)  # Shape: [batch_size, 1]
        current_q = self.policy_net(states).gather(1, actions)
        loss = self.loss_fn(current_q, target_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1)
        self.optimizer.step()
        self.update_counter += 1
        if self.update_counter % self.update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        
    def save(self, path):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
        
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
