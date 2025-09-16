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
        
        if config:
            self.config.update(config)
        
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
        if isinstance(state, np.ndarray) and state.shape[0] != self.state_dim:
            print(f"Student state dimension mismatch: got {state.shape[0]}, expected {self.state_dim}")
            self.rebuild_networks(state.shape[0])
        
        # Handle case with no valid actions
        if valid_actions is None or len(valid_actions) == 0:
            return -1

        # Optional memory/teacher hints (kept minimal)
        if env and hasattr(env, 'pipeline_memory') and env.pipeline_memory and random.random() < 0.2:
            current_len = len(env.current_pipeline) if hasattr(env, 'current_pipeline') else 0
            for mem_pipeline in env.pipeline_memory[:2]:
                if current_len < len(mem_pipeline['components']):
                    next_comp = mem_pipeline['components'][current_len]
                    for action in valid_actions:
                        if str(env.available_components[action]) == str(next_comp):
                            print(f"Student using pipeline memory guidance: {next_comp}")
                            return action
        
        if teacher_feedback is not None and random.random() < 0.7 and teacher_feedback in valid_actions:
            return int(teacher_feedback)
        
        return self.select_action(state, valid_actions)
    
    def select_action(self, state, valid_actions):
        if valid_actions is None or len(valid_actions) == 0:
            return -1

        eps = self.config.get('epsilon', 0.1)
        if np.random.random() < eps:
            return int(np.random.choice(valid_actions))

        device = next(self.model.policy_net.parameters()).device
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.model.policy_net(state_tensor).squeeze(0)

        mask = torch.full((self.action_dim,), float('-inf'), device=q_values.device)
        mask[valid_actions] = 0.0
        masked_q = q_values + mask

        return int(torch.argmax(masked_q).item())
    
    def learn(self, state, action, reward, next_state, done, teacher_intervened=False):
        if isinstance(state, np.ndarray) and state.shape[0] != self.state_dim:
            print(f"Student dimension mismatch in learn method: got {state.shape[0]}, expected {self.state_dim}")
            self.rebuild_networks(state.shape[0])
            return
        if teacher_intervened:
            adjusted_reward = reward * 1.05
        else:
            adjusted_reward = reward
        
        self.model.buffer.push(state, action, adjusted_reward, next_state, done)
        
        if not hasattr(self, 'recent_experiences'):
            self.recent_experiences = []
        self.recent_experiences.append((state, action, reward, next_state, done))
        
        self.model.update_model()
    
    def save(self, path):
        self.model.save(path)
        
    def load(self, path):
        self.model.load(path)
    
    def apply_final_reward(self, final_reward, decay=0.95):
        if not hasattr(self, 'recent_experiences'):
            return
        
        reward = final_reward
        for exp in reversed(self.recent_experiences):
            exp_state, exp_action, _, exp_next_state, exp_done = exp
            self.model.buffer.push(exp_state, exp_action, reward, exp_next_state, exp_done)
            reward *= decay

    def rebuild_networks(self, new_state_dim):
        print(f"Student rebuilding networks: state size changing from {self.model.state_dim} to {new_state_dim}")
        
        if hasattr(self.model, 'buffer'):
            print(f"Clearing student replay buffer with {len(self.model.buffer)} experiences due to dimension change")
            self.model.buffer.memory.clear()
        
        if hasattr(self.model, 'rebuild_networks'):
            self.model.rebuild_networks(new_state_dim)
        else:
            print("WARNING: Student DoubleDQN has no rebuild_networks method - creating new model")
            device = next(self.model.policy_net.parameters()).device
            
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
            
            self.model = DoubleDQN(
                state_dim=new_state_dim,
                action_dim=self.model.action_dim,
                **old_config
            )
            
            self.model.policy_net = self.model.policy_net.to(device)
            self.model.target_net = self.model.target_net.to(device)
        
        self.state_dim = new_state_dim
        print("Student network rebuilt to match new state dimensions")
    def act_greedy(self, state, valid_actions=None, env=None):
        if isinstance(state, np.ndarray) and state.shape[0] != self.state_dim:
            print(f"Student state dimension mismatch in act_greedy: got {state.shape[0]}, expected {self.state_dim}")
            self.rebuild_networks(state.shape[0])

        if valid_actions is None or len(valid_actions) == 0:
            return -1

        device = next(self.model.policy_net.parameters()).device
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.model.policy_net(state_tensor).squeeze(0)

        mask = torch.full((self.action_dim,), float('-inf'), device=q_values.device)
        mask[valid_actions] = 0.0
        masked_q = q_values + mask
        return int(torch.argmax(masked_q).item())