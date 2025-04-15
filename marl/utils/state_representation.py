from typing import Dict, Any, Set, Tuple
import numpy as np
from collections import namedtuple

class PipelineStateRepresentation:
    def __init__(self):
        self.state_space: Dict[str, Dict[str, Any]] = {}
        
    def add_component_data(self, current_component: str, next_component: str, 
                          performance_delta: float, context: Dict[str, Any]):
        if current_component not in self.state_space:
            self.state_space[current_component] = {
                "next_components": set(),
                "performance": {},
                "contexts": {}
            }
        
        self.state_space[current_component]["next_components"].add(next_component)
        
        if next_component not in self.state_space[current_component]["performance"]:
            self.state_space[current_component]["performance"][next_component] = {
                "count": 0,
                "sum_delta": 0.0,
                "mean": 0.0,
                "variance": 0.0
            }
        
        perf = self.state_space[current_component]["performance"][next_component]
        perf["count"] += 1
        perf["sum_delta"] += performance_delta
        
        old_mean = perf["mean"]
        perf["mean"] = old_mean + (performance_delta - old_mean) / perf["count"]
        perf["variance"] += (performance_delta - old_mean) * (performance_delta - perf["mean"])
        if next_component not in self.state_space[current_component]["contexts"]:
            self.state_space[current_component]["contexts"][next_component] = []
        self.state_space[current_component]["contexts"][next_component].append(context)
    
    def get_valid_next_components(self, current_component: str) -> Set[str]:
        if current_component in self.state_space:
            return self.state_space[current_component]["next_components"]
        return set()
    
    def get_component_performance(self, current_component: str, next_component: str) -> Tuple[float, float]:
        if (current_component in self.state_space and 
            next_component in self.state_space[current_component]["performance"]):
            perf = self.state_space[current_component]["performance"][next_component]
            mean = perf["mean"]
            std = np.sqrt(perf["variance"] / perf["count"]) if perf["count"] > 1 else float('inf')
            return mean, std
        return 0.0, float('inf')
class StatePreprocessor:
    def __init__(self, fixed_dim=None, max_dim=100):
        self.fixed_dim = fixed_dim
        self.max_dim = max_dim
    
    def preprocess(self, state, preserve_batch=False):
        """Process state to fixed dimensions regardless of source"""
        if state is None:
            return None
            
        if preserve_batch and len(state.shape) > 1:
            # Handle batch of states
            batch_size = state.shape[0]
            processed_batch = []
            for i in range(batch_size):
                processed_batch.append(self._process_single_state(state[i]))
            return np.array(processed_batch)
        else:
            return self._process_single_state(state)
    
    def _process_single_state(self, state):
        """Process a single state to fixed dimensions"""
        if not isinstance(state, np.ndarray):
            state = np.array(state)
            
        state_dim = state.shape[0] if len(state.shape) > 0 else 1
        target_dim = self.fixed_dim or min(state_dim, self.max_dim)
        
        if state_dim == target_dim:
            return state
        elif state_dim < target_dim:
            # Pad with zeros
            return np.pad(state, (0, target_dim - state_dim))
        else:
            # Truncate
            return state[:target_dim]
