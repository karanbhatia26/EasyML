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
        self.lr = lr
        
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
        self.buffer = ReplayBuffer(buffer_size, parent_model=self)
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
        
        batch = self.buffer.sample(self.batch_size)
        
        # First convert to numpy arrays (more efficient)
        states_array = np.array([exp.state for exp in batch])
        actions_array = np.array([exp.action for exp in batch])
        rewards_array = np.array([exp.reward for exp in batch])
        next_states_array = np.array([exp.next_state for exp in batch])
        dones_array = np.array([exp.done for exp in batch])
        
        # Convert to tensors
        states = torch.FloatTensor(states_array).to(self.device)
        actions = torch.LongTensor(actions_array).to(self.device)
        rewards = torch.FloatTensor(rewards_array).to(self.device)
        next_states = torch.FloatTensor(next_states_array).to(self.device)
        dones = torch.BoolTensor(dones_array).to(self.device)
        
        # Check if networks need rebuilding
        if self.rebuild_networks_if_needed(states):
            return  # Skip this update cycle
        
        # Continue with normal DQN update
        curr_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)
            next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1).float()) * self.gamma * next_q
        
        loss = self.loss_fn(curr_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        # Update target network periodically
        self.update_counter += 1
        if self.update_counter % self.update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
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
        
    def rebuild_networks_if_needed(self, state_tensor):
        """Rebuild networks if input dimension doesn't match state dimension"""
        input_size = state_tensor.shape[1]
        current_size = None
        
        # Check current network input size directly from policy_net
        if hasattr(self.policy_net, 'input_size'):
            current_size = self.policy_net.input_size
        elif hasattr(self.policy_net, 'network'):
            for layer in self.policy_net.network:
                if hasattr(layer, 'in_features'):
                    current_size = layer.in_features
                    break
        
        # If sizes don't match or no network exists yet, rebuild
        if current_size is None or current_size != input_size:
            self.rebuild_networks(input_size)
            return True
        return False

    def rebuild_networks(self, new_state_dim):
        """Properly rebuild networks with new input dimensions"""
        print(f"DoubleDQN rebuilding networks: {self.state_dim} -> {new_state_dim}")
        
        # Update the state dimension
        old_state_dim = self.state_dim
        self.state_dim = new_state_dim
        
        # Get the current device being used
        device = next(self.policy_net.parameters()).device
        
        # Create completely new network instances with the new dimensions
        from marl.models.networks import DQNetwork
        self.policy_net = DQNetwork(new_state_dim, self.action_dim).to(device)
        self.target_net = DQNetwork(new_state_dim, self.action_dim).to(device)
        
        # Create a new optimizer for the new policy network
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)
        
        # Reset the update counter
        self.update_counter = 0
        
        print(f"Created new networks with input size {new_state_dim} on {device}")
        
        # Clear the replay buffer since old experiences have different dimension
        self.buffer.memory.clear()
        print("Cleared replay buffer due to dimension change")
