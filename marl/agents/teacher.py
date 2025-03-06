from marl.agents.base import BaseAgent
from marl.models.double_dqn import DoubleDQN
import numpy as np
import os
import random

class TeacherAgent(BaseAgent):
    """Agent responsible for evaluating and providing feedback on ML pipelines."""
    
    def __init__(self, state_dim, action_dim, config=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Default config values
        self.config = {
            'learning_rate': 5e-5,
            'gamma': 0.99,
            'epsilon': 0.5,
            'epsilon_min': 0.05,
            'epsilon_decay': 0.99,
            'buffer_size': 10000,
            'batch_size': 64,
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
        
        # Component performance tracking
        self.component_performance = {}
        
        # Initialize rules
        self.rules = self._init_rules()
        
    def _init_rules(self):
        """Initialize rule-based knowledge of pipeline construction."""
        # Define rule checker functions (instead of lambdas)
        def check_needs_imputer(state, env):
            return 'SimpleImputer' not in ' '.join(env.current_pipeline)
            
        def check_needs_scaler(state, env):
            return 'StandardScaler' not in ' '.join(env.current_pipeline) and 'MinMaxScaler' not in ' '.join(env.current_pipeline)
            
        def check_needs_classifier(state, env):
            return len(env.current_pipeline) == env.max_pipeline_length - 1
        
        # Return a dictionary of rules instead of a list
        return {
            "imputer_rule": {
                "condition": check_needs_imputer,
                "action": 0,  # This should be the action index for SimpleImputer
                "confidence": 0.8
            },
            "scaler_rule": {
                "condition": check_needs_scaler,
                "action": 3,  # This should be the action index for StandardScaler
                "confidence": 0.7
            },
            "classifier_rule": {
                "condition": check_needs_classifier,
                "action": 10,  # This should be the action index for RandomForest
                "confidence": 0.9
            }
        }
        
    def evaluate_pipeline(self, pipeline, data):
        """Evaluate a pipeline's performance on the data."""
        # This would be implemented to actually run the pipeline and compute metrics
        # Simplified version:
        try:
            # train_score = pipeline.fit(data['X_train'], data['y_train']).score(data['X_train'], data['y_train'])
            # val_score = pipeline.score(data['X_val'], data['y_val'])
            # Return various metrics
            pass
        except Exception as e:
            # Handle failures gracefully
            return {"success": False, "error": str(e)}
    
    def act(self, state, valid_actions=None, env=None):
        # Use rules if environment is provided
        if env is not None and hasattr(self, 'rules'):
            # Try to apply rules
            for rule_name, rule_data in self.rules.items():
                condition = rule_data["condition"]
                if condition(state, env):
                    action = rule_data["action"]
                    # Check if action is valid
                    if valid_actions is None or action in valid_actions:
                        return action
        
        # Fallback to epsilon-greedy
        # Simple epsilon-greedy strategy
        if valid_actions and len(valid_actions) > 0:
            if np.random.random() < self.config['epsilon']:
                # Exploration: random valid action
                return np.random.choice(valid_actions)
            else:
                # Exploitation: use model
                return self.model.select_action(state, valid_actions)
        else:
            # No valid actions or invalid input
            return 0  # Default action
    
    def learn(self, state, action, reward, next_state, done):
        """Learn from interaction with the environment."""
        # Store experience
        self.model.buffer.push(state, action, reward, next_state, done)
        
        # Only update model periodically to avoid computational bottlenecks
        if np.random.random() < 0.25:  # Update only 25% of the time
            self.model.update_model()
    
    def save(self, path):
        """Save the model and performance data."""
        self.model.save(path)
        
        # Save component performance data
        if hasattr(self, 'component_performance') and self.component_performance:
            np.save(path + "_component_perf", self.component_performance)
    
    def load(self, path):
        """Load the model and performance data."""
        self.model.load(path)
        
        # Load component performance data if it exists
        if os.path.exists(path + "_component_perf.npy"):
            self.component_performance = np.load(path + "_component_perf.npy", 
                                               allow_pickle=True).item()
