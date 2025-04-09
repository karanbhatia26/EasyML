from marl.agents.base import BaseAgent
from marl.models.double_dqn import DoubleDQN
import numpy as np
import os
import torch
import random

class TeacherAgent(BaseAgent):
    """Teacher agent that uses reinforcement learning to guide student."""
    
    def __init__(self, state_dim, action_dim, config=None):
        self.state_dim = state_dim
        self.action_dim = action_dim + 1  # +1 for "no intervention" action
        
        # Default config values
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
        
        # Override defaults with provided config
        if config:
            self.config.update(config)
        
        # Initialize the Double DQN model - same architecture as student
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
        
        # Component knowledge tracking - maintain this from original implementation
        self.component_knowledge = {}
        
        # Track intervention history
        self.intervention_history = []
        self.intervention_success = []
        self.intervention_threshold = 0.1  # Minimum expected improvement to intervene
        
    def act(self, state, valid_actions=None, student_action=None, env=None):
        """Take action with dimension checking"""
        # Skip if no valid actions
        if valid_actions is None or len(valid_actions) == 0:
            return False, 0
        
        # NEW: State dimension check
        if isinstance(state, np.ndarray) and state.shape[0] != self.model.state_dim:
            print(f"Teacher state dimension mismatch: got {state.shape[0]}, expected {self.model.state_dim}")
            # Force rebuild
            self.rebuild_networks(state.shape[0])  
        
        # Check if exploration
        if np.random.random() < self.config['epsilon']:
            # Random decision to intervene
            should_intervene = np.random.random() < 0.4
            if should_intervene and len(valid_actions) > 0:
                action = np.random.choice(valid_actions)
                return True, action
            else:
                return False, 0
        
        # Get device of policy network
        device = next(self.model.policy_net.parameters()).device
        
        # Convert state to tensor on the correct device
        if isinstance(state, np.ndarray):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        # Use model to get Q-values (all on same device)
        with torch.no_grad():
            q_values = self.model.policy_net(state_tensor).squeeze()
        
        # Filter to valid actions
        if valid_actions:
            valid_q = np.array([q_values[a].item() for a in valid_actions])
            best_action_idx = np.argmax(valid_q)
            best_action = valid_actions[best_action_idx]
            
            # Decide whether to intervene
            if student_action is None or best_action != student_action:
                # Intervention threshold - only intervene if significantly better
                if len(valid_q) > 1:
                    # Find improvement margin
                    if student_action is not None:
                        student_q = q_values[student_action].item()
                        improvement = valid_q[best_action_idx] - student_q
                        
                        # Only intervene if significant improvement
                        if improvement > 0.1:
                            return True, best_action
                    else:
                        return True, best_action
            
            # Default - don't intervene
            return False, 0
        
    def learn(self, state, action_tuple, reward, next_state, done):
        """Learn from experience with awareness of teacher interventions"""
        # First check state dimension and rebuild if necessary
        if isinstance(state, np.ndarray) and state.shape[0] != self.state_dim:
            print(f"Teacher dimension mismatch in learn method: got {state.shape[0]}, expected {self.state_dim}")
            self.rebuild_networks(state.shape[0])
            # Skip this experience as the buffer was just cleared
            return
        
        # Convert the tuple action to a single integer representation
        should_intervene, teacher_action = action_tuple
        
        # Convert the tuple action to a flat tensor representation
        action_code = int(should_intervene) * self.action_dim + teacher_action
        
        # Store in replay buffer
        self.model.buffer.push(state, action_code, reward, next_state, done)
        
        # Update the network with safe error handling
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
                    # Re-raise if it's not a dimension error
                    raise e
        
        # Record intervention outcome for adaptive intervention strategy
        if hasattr(self, 'intervention_outcomes'):
            self.intervention_outcomes.append((should_intervene, reward > 0))
    
    def _analyze_interventions(self):
        if len(self.intervention_outcomes) < 10:
            return
            
        # Get recent interventions
        recent = self.intervention_outcomes[-30:]
        
        # Calculate success rate
        rewards = [r for _, _, r, _ in recent]
        success_rate = sum(1 for r in rewards if r > 0) / max(1, len(rewards))
        
        # Adjust epsilon based on success
        if success_rate < 0.3:  # Interventions not helping much
            self.config['epsilon'] = min(0.8, self.config['epsilon'] + 0.05)
            print("Teacher increasing exploration: interventions not effective")
        elif success_rate > 0.7:  # Interventions helping a lot
            self.config['epsilon'] = max(0.1, self.config['epsilon'] - 0.05)
            print("Teacher reducing exploration: interventions very effective")
    
    def save(self, path):
        """Save the model and performance data."""
        self.model.save(path)
        
    def load(self, path):
        """Load the model."""
        self.model.load(path)
        
    def update_component_knowledge(self, pipeline, performance):
        """Update knowledge about component effectiveness"""
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
    
    def calculate_teacher_reward(self, env, student_action, teacher_action, performance):
        """Calculate improved reward based on counterfactual analysis"""
        if student_action == teacher_action:
            # Agreed with student - no extra credit
            return performance * 0.5
            
        # Simulate what would happen with student's choice
        test_pipeline = env.current_pipeline + [str(env.available_components[student_action])]
        student_performance = env.evaluate_pipeline(test_pipeline)
        
        # Compare with actual performance
        improvement = performance - student_performance
        
        if improvement > 0.1:
            return performance * 0.8  # Significant improvement
        elif improvement > 0:
            return performance * 0.6  # Modest improvement
        else:
            return performance * 0.2  # Made things worse
    
    def rebuild_networks(self, new_state_dim):
        """Rebuild networks and clear buffer when dimensions change"""
        print(f"Teacher rebuilding networks: state size changing from {self.model.state_dim} to {new_state_dim}")
        
        # IMPORTANT: Clear the replay buffer to prevent dimension mismatches
        if hasattr(self.model, 'buffer'):
            print(f"Clearing replay buffer with {len(self.model.buffer)} experiences due to dimension change")
            self.model.buffer.memory.clear()
        
        # Forward the call to the model's rebuild_networks method
        if hasattr(self.model, 'rebuild_networks'):
            self.model.rebuild_networks(new_state_dim)
        else:
            # Manual rebuilding if method doesn't exist
            print("WARNING: DoubleDQN has no rebuild_networks method - creating new model")
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
        print("Teacher network rebuilt to match new state dimensions")
    
    def apply_final_reward(self, final_reward, decay=0.95):
        """Apply final episode reward to recent experiences"""
        if not hasattr(self, 'recent_experiences'):
            self.recent_experiences = []
            return
        
        # Apply decayed final reward to recent experiences
        reward = final_reward
        for exp in reversed(self.recent_experiences):
            state, action_tuple, _, next_state, done = exp
            # For teacher, action is a tuple (should_intervene, action)
            # Update with the new reward
            self.model.buffer.push(state, action_tuple, reward, next_state, done)
            # Decay reward for earlier steps
            reward *= decay
        
        # Clear for next episode
        self.recent_experiences = []
