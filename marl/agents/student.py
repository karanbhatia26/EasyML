from marl.agents.base import BaseAgent
from marl.models.double_dqn import DoubleDQN

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
        
    def act(self, state, valid_actions=None):
        """Select a component to add to the ML pipeline."""
        return self.model.select_action(state, valid_actions)
    
    def learn(self, state, action, reward, next_state, done):
        """Learn from the teacher's feedback."""
        self.model.buffer.push(state, action, reward, next_state, done)
        self.model.update_model()
    
    def save(self, path):
        """Save the student's model."""
        self.model.save(path)
    
    def load(self, path):
        """Load the student's model."""
        self.model.load(path)
