from marl.agents.base import BaseAgent
from marl.models.double_dqn import DoubleDQN
import random
import numpy as np
import torch

class StudentAgent(BaseAgent):
    """Agent responsible for generating ML pipelines."""
    
    def __init__(self, state_dim, action_dim, config=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Default config values
        self.config = {
            'learning_rate': 1e-4,
            'gamma': 0.99,
            'epsilon': 1.0,
            'epsilon_min': 0.1,
            'epsilon_decay': 0.995,
            'buffer_size': 10000,
            'batch_size': 256,
            'update_freq': 10
        }
        
        # Override defaults with provided config
        if config:
            self.config.update(config)
        
        # Initialize the Double DQN model
        self.model = DoubleDQN(
            state_dim=state_dim,
            action_dim=action_dim,
            lr=self.config['learning_rate'],
            gamma=self.config['gamma'],
            epsilon=self.config['epsilon'],
            epsilon_min=self.config['epsilon_min'],
            epsilon_decay=self.config['epsilon_decay'],
            buffer_size=self.config['buffer_size'],
            batch_size=self.config['batch_size'],
            update_freq=self.config['update_freq']
        )
        
    def act(self, state, valid_actions=None, teacher_feedback=None, pipeline_memory=None, env=None):
        """Unified action selection with memory and teacher guidance"""
        # Check state dimension and rebuild if necessary
        if isinstance(state, np.ndarray) and state.shape[0] != self.state_dim:
            print(f"Student state dimension mismatch: got {state.shape[0]}, expected {self.state_dim}")
            self.rebuild_networks(state.shape[0])
        
        # Handle case with no valid actions
        if valid_actions is None or len(valid_actions) == 0:
            return -1
        
        # MEMORY GUIDANCE: Use pipeline memory if available
        if env and hasattr(env, 'pipeline_memory') and env.pipeline_memory and random.random() < 0.2:
            current_len = len(env.current_pipeline) if hasattr(env, 'current_pipeline') else 0
            
            for mem_pipeline in env.pipeline_memory[:2]:  # Look at top 2 pipelines
                if current_len < len(mem_pipeline['components']):
                    next_comp = mem_pipeline['components'][current_len]
                    
                    for action in valid_actions:
                        if str(env.available_components[action]) == str(next_comp):
                            print(f"Student using pipeline memory guidance: {next_comp}")
                            return action
        
        # TEACHER GUIDANCE: If teacher feedback is provided
        if teacher_feedback is not None and random.random() < 0.7:  # 70% chance to use feedback
            if teacher_feedback in valid_actions:
                return teacher_feedback
        
        # DEFAULT: Use model for action selection
        return self.model.select_action(state, valid_actions)
    
    def select_action(self, state, valid_actions):
        """Fallback action selection if model doesn't have an action method"""
        # Epsilon-greedy action selection
        if np.random.random() < self.config.get('epsilon', 0.1):
            return np.random.choice(valid_actions)
            
        # Otherwise, use policy network
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_values = self.model.policy_net(state_tensor)
            
        # Filter to only valid actions
        valid_action_values = action_values[0][valid_actions]
        return valid_actions[torch.argmax(valid_action_values).item()]
    
    def learn(self, state, action, reward, next_state, done, teacher_intervened=False):
        """Learn from experience with awareness of teacher interventions"""
        if isinstance(state, np.ndarray) and state.shape[0] != self.state_dim:
            print(f"Student dimension mismatch in learn method: got {state.shape[0]}, expected {self.state_dim}")
            self.rebuild_networks(state.shape[0])
            # Skip this experience as the buffer was just cleared
            return
        # Option to adjust reward when learning from teacher's suggestions
        if teacher_intervened:
            # You can adjust how the student learns from teacher interventions
            # For example, slightly boost reward to encourage learning from teacher
            adjusted_reward = reward * 1.05
        else:
            adjusted_reward = reward
        
        self.model.buffer.push(state, action, adjusted_reward, next_state, done)
        
        if not hasattr(self, 'recent_experiences'):
            self.recent_experiences = []
        self.recent_experiences.append((state, action, reward, next_state, done))
        
        self.model.update_model()
    
    def save(self, path):
        """Save the model"""
        self.model.save(path)
        
    def load(self, path):
        """Load the model"""
        self.model.load(path)
    
    def apply_final_reward(self, final_reward, decay=0.95):
        """Apply final episode reward to recent experiences"""
        if not hasattr(self, 'recent_experiences'):
            return
        
        reward = final_reward
        for exp in reversed(self.recent_experiences):
            exp_state, exp_action, _, exp_next_state, exp_done = exp
            self.model.buffer.push(exp_state, exp_action, reward, exp_next_state, exp_done)
            reward *= decay

    def rebuild_networks(self, new_state_dim):
        """Rebuild networks and clear buffer when dimensions change"""
        print(f"Student rebuilding networks: state size changing from {self.model.state_dim} to {new_state_dim}")
        
        # IMPORTANT: Clear the replay buffer to prevent dimension mismatches
        if hasattr(self.model, 'buffer'):
            print(f"Clearing student replay buffer with {len(self.model.buffer)} experiences due to dimension change")
            self.model.buffer.memory.clear()
        
        # Forward the call to the model's rebuild_networks method
        if hasattr(self.model, 'rebuild_networks'):
            self.model.rebuild_networks(new_state_dim)
        else:
            # Manual rebuilding if method doesn't exist
            print("WARNING: Student DoubleDQN has no rebuild_networks method - creating new model")
            device = next(self.model.policy_net.parameters()).device
            
            # Save old configuration
            old_config = {
                'lr': self.model.lr,
                'gamma': self.model.gamma,
                'epsilon': self.model.epsilon,
                'epsilon_min': self.model.epsilon_min,
                'epsilon_decay': self.model.epsilon_decay,
                'buffer_size': len(self.model.buffer.memory),
                'batch_size': self.model.batch_size,
                'update_freq': self.model.update_freq
            }
            
            # Create a new model with the new state_dim
            self.model = DoubleDQN(
                state_dim=new_state_dim,
                action_dim=self.model.action_dim,
                **old_config
            )
            
            # Ensure it's on the same device
            self.model.policy_net = self.model.policy_net.to(device)
            self.model.target_net = self.model.target_net.to(device)
        
        # Update local state dimension
        self.state_dim = new_state_dim
        print("Student network rebuilt to match new state dimensions")