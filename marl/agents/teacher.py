from marl.agents.base import BaseAgent
from marl.models.double_dqn import DoubleDQN
import numpy as np
import os
import torch
import random

class TeacherAgent(BaseAgent):
    
    def __init__(self, state_dim, action_dim, config=None):
        self.state_dim = state_dim
        self.action_dim = action_dim + 1
        
        self.config = {
            'learning_rate': 5e-5,
            'gamma': 0.99,
            'epsilon': 0.7,
            'epsilon_min': 0.1,
            'epsilon_decay': 0.99,
            'buffer_size': 20000,
            'batch_size': 256,
            'update_freq': 10
        }
        if config:
            self.config.update(config)
        
        self.model = DoubleDQN(
            state_dim=state_dim,
            action_dim=self.action_dim,
            lr=self.config['learning_rate'],
            gamma=self.config['gamma'],
            epsilon=self.config['epsilon'],
            epsilon_min=self.config['epsilon_min'],
            epsilon_decay=self.config['epsilon_decay'],
            buffer_size=self.config['buffer_size'],
            batch_size=self.config['batch_size'],
            update_freq=self.config['update_freq']
        )
        
        self.component_knowledge = {}

        self.intervention_history = []
        self.intervention_success = []
        self.intervention_threshold = 0.1
        self.intervention_history = []
        self.episode_counter = 0
        self.intervention_decay = config.get('intervention_decay', True)
        
    def act(self, state, valid_actions=None, student_action=None, env=None):
        state_tensor_dim = len(state)
        network_input_dim = self.model.policy_net.network[0].in_features
        
        if state_tensor_dim != network_input_dim:
            print(f"Rebuilding teacher network: input dim {network_input_dim} → {state_tensor_dim}")
            self.rebuild_networks(state_tensor_dim)
        
        if valid_actions is None or len(valid_actions) == 0:
            return False, 0
        base_threshold = 0.4
        min_threshold = 0.05
        decay_episodes = 800
        
        if self.intervention_decay:
            intervention_threshold = max(min_threshold, 
                                       base_threshold * (1.0 - self.episode_counter / decay_episodes))
        else:
            intervention_threshold = base_threshold
        
        if np.random.random() < self.config['epsilon']:
            should_intervene = np.random.random() < intervention_threshold
            if should_intervene and len(valid_actions) > 0:
                action = np.random.choice(valid_actions)
                self.intervention_history.append((self.episode_counter, True))
                return True, action
            else:
                self.intervention_history.append((self.episode_counter, False))
                return False, 0
        
        device = next(self.model.policy_net.parameters()).device
        
        if isinstance(state, np.ndarray):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        with torch.no_grad():
            q_values = self.model.policy_net(state_tensor).squeeze()
        
        if valid_actions:
            valid_q = np.array([q_values[a].item() for a in valid_actions])
            best_action_idx = np.argmax(valid_q)
            best_action = valid_actions[best_action_idx]
            
            if student_action is not None:
                student_q = q_values[student_action].item() if student_action < len(q_values) else 0
                improvement = valid_q[best_action_idx] - student_q
                
                need_intervention = improvement > 0.15
                
                if need_intervention and np.random.random() < intervention_threshold:
                    self.intervention_history.append((self.episode_counter, True))
                    return True, best_action
        
        self.intervention_history.append((self.episode_counter, False))
        return False, 0
        
    def learn(self, state, action_tuple, reward, next_state, done):
        if isinstance(state, np.ndarray) and state.shape[0] != self.state_dim:
            print(f"Teacher dimension mismatch in learn method: got {state.shape[0]}, expected {self.state_dim}")
            self.rebuild_networks(state.shape[0])
            return
        
        should_intervene, teacher_action = action_tuple

        action_code = int(should_intervene) * self.action_dim + teacher_action
        
        self.model.buffer.push(state, action_code, reward, next_state, done)
        
        if len(self.model.buffer) > self.config['batch_size']:
            try:
                self.model.update_model()
            except RuntimeError as e:
                if "shapes cannot be multiplied" in str(e):
                    print("WARNING: Dimension mismatch in teacher buffer. Clearing replay buffer.")
                    self.model.buffer.memory.clear()
                    print(f"Rebuilding teacher networks to match state dimension: {state.shape[0]}")
                    self.rebuild_networks(state.shape[0])
                else:
                    raise e
        
        if hasattr(self, 'intervention_outcomes'):
            self.intervention_outcomes.append((should_intervene, reward > 0))

        if done:
            self.episode_counter += 1
    
    def _analyze_interventions(self):
        if len(self.intervention_outcomes) < 10:
            return
        
        recent = self.intervention_outcomes[-30:]
        
        rewards = [r for _, _, r, _ in recent]
        success_rate = sum(1 for r in rewards if r > 0) / max(1, len(rewards))
        
        if success_rate < 0.3:
            self.config['epsilon'] = min(0.8, self.config['epsilon'] + 0.05)
            print("Teacher increasing exploration: interventions not effective")
        elif success_rate > 0.7:
            self.config['epsilon'] = max(0.1, self.config['epsilon'] - 0.05)
            print("Teacher reducing exploration: interventions very effective")
    
    def save(self, path):
        self.model.save(path)
        
    def load(self, path):
        self.model.load(path)
        
    def update_component_knowledge(self, pipeline, performance):
        if len(pipeline) < 2:
            return
            
        for i in range(len(pipeline) - 1):
            from_comp = pipeline[i]
            to_comp = pipeline[i+1]
            
            if from_comp not in self.component_knowledge:
                self.component_knowledge[from_comp] = {}
                
            if to_comp not in self.component_knowledge[from_comp]:
                self.component_knowledge[from_comp][to_comp] = {
                    'count': 0,
                    'total_performance': 0.0,
                    'avg_performance': 0.0
                }
                
            info = self.component_knowledge[from_comp][to_comp]
            info['count'] += 1
            info['total_performance'] += performance
            info['avg_performance'] = info['total_performance'] / info['count']
    
    def calculate_teacher_reward(self, student_action, should_intervene, teacher_action, performance):
        if not should_intervene:
            return 0.02
        if performance > 0:
            return performance * 0.3
        elif student_action == teacher_action:
            return -0.03
        else:
            return -0.005
    
    def rebuild_networks(self, new_state_dim):
        print(f"Teacher rebuilding networks: state size changing from {self.model.state_dim} to {new_state_dim}")
        
        if hasattr(self.model, 'buffer'):
            print(f"Clearing replay buffer with {len(self.model.buffer)} experiences due to dimension change")
            self.model.buffer.memory.clear()
        
        if hasattr(self.model, 'rebuild_networks'):
            self.model.rebuild_networks(new_state_dim)
        else:
            print("WARNING: DoubleDQN has no rebuild_networks method - creating new model")
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
        print("Teacher network rebuilt to match new state dimensions")
    
    def apply_final_reward(self, final_reward, decay=0.95):
        if not hasattr(self, 'recent_experiences'):
            self.recent_experiences = []
            return
        reward = final_reward
        for exp in reversed(self.recent_experiences):
            state, action_tuple, _, next_state, done = exp
            self.model.buffer.push(state, action_tuple, reward, next_state, done)
            reward *= decay
        
        self.recent_experiences = []
    def act_greedy(self, state, valid_actions, student_action, env=None):
        """Act greedily with no exploration for evaluation purposes"""
        if isinstance(state, np.ndarray) and state.shape[0] != self.state_dim:
            print(f"Teacher state dimension mismatch in act_greedy: got {state.shape[0]}, expected {self.state_dim}")
            self.rebuild_networks(state.shape[0])
        
        # Handle case with no valid actions
        if valid_actions is None or len(valid_actions) == 0:
            return False, -1
        
        # Get Q-values for all actions
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(next(self.model.policy_net.parameters()).device)
        with torch.no_grad():
            action_values = self.model.policy_net(state_tensor)
        
        # Filter to only valid actions
        mask = torch.ones(self.action_dim) * float('-inf')
        mask[valid_actions] = 0
        masked_action_values = action_values + mask
        
        # Get teacher's best action
        teacher_action = torch.argmax(masked_action_values).item()
        
        # Compare teacher's best action with student's action
        should_intervene = False
        
        if teacher_action != student_action:
            # Get Q-values for both actions
            q_teacher = masked_action_values[0][teacher_action].item()
            q_student = masked_action_values[0][student_action].item() if student_action in valid_actions else float('-inf')
            
            # Intervene if teacher's action has higher value (beyond threshold)
            threshold = getattr(self, 'intervention_threshold', 0.1)
            should_intervene = q_teacher > q_student + threshold
        
        return should_intervene, teacher_action
