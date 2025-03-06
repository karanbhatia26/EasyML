import numpy as np
from marl.environments.pipeline_env import PipelineEnvironment
from sklearn.datasets import load_iris

def test_environment():
    # Load data
    data = load_iris()
    X_train, X_test = data.data[:100], data.data[100:]
    y_train, y_test = data.target[:100], data.target[100:]
    
    dataset = {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test
    }
    
    # Create environment
    env = PipelineEnvironment(dataset)
    
    # Test reset
    state = env.reset()
    print("Initial state shape:", state.shape)
    
    # Test step with a few actions
    for action in [0, 3, 5]:  # Example actions
        print(f"\nTaking action {action} ({env.available_components[action]})")
        next_state, reward, done, info = env.step(action)
        print(f"Reward: {reward}")
        print(f"Done: {done}")
        print(f"Performance: {info['performance']}")
        print(f"Pipeline so far: {env.current_pipeline}")
    
    # Test pipeline evaluation
    performance = env._evaluate_pipeline()
    print(f"\nFinal pipeline performance: {performance}")

if __name__ == "__main__":
    test_environment()