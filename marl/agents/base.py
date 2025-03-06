from abc import ABC, abstractmethod
class BaseAgent(ABC):
    """Abstract base class for all agents."""
    
    @abstractmethod
    def act(self, state, valid_actions=None):
        """Select an action based on the current state."""
        pass
    
    @abstractmethod
    def learn(self, state, action, reward, next_state, done):
        """Learn from experience."""
        pass
    
    @abstractmethod
    def save(self, path):
        """Save the agent's model."""
        pass
    
    @abstractmethod
    def load(self, path):
        """Load the agent's model."""
        pass
