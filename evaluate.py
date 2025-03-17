import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from marl.agents.student import StudentAgent
from marl.agents.teacher import TeacherAgent
from marl.environments.pipeline_env import PipelineEnvironment
from sklearn.datasets import load_iris, load_wine, load_diabetes, load_breast_cancer
import json
import os

def evaluate_pipeline(pipeline, dataset):
    """Evaluate a pipeline on a dataset."""
    # This would create and evaluate the actual pipeline
    # For now, we'll return a simple performance metric
    env = PipelineEnvironment(dataset)
    env.current_pipeline = pipeline
    return env._evaluate_pipeline()

def main():
    # Load the trained models
    state_dim = 100  # Use the same dimensions used during training
    action_dim = 32  # Use the same dimensions used during training
    
    student = StudentAgent(state_dim, action_dim)
    student.load("models/student_model_test.pt")
    
    # Set to evaluation mode (no exploration)
    student.config['epsilon'] = 0
    
    # Load test datasets
    datasets = {
        "iris": load_iris(),
        "wine": load_wine(),
        "breast_cancer": load_breast_cancer()
    }
    
    results = {}
    validation_results = {}
    
    for name, data in datasets.items():
        print(f"\n\n{'='*40}")
        print(f"Evaluating on {name} dataset...")
        print(f"{'='*40}")
        
        # Prepare dataset
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            data.data, data.target, test_size=0.2, random_state=42)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42)
            
        dataset = {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
            "feature_names": data.feature_names,
            "n_classes": len(np.unique(data.target))
        }
        
        # Generate pipeline using student
        env = PipelineEnvironment(dataset)
        state = env.reset()
        done = False
        pipeline_components = []
        
        while not done:
            valid_actions = env.get_valid_actions()
            action = student.act(state, valid_actions)
            next_state, reward, done, info = env.step(action)
            component = env.available_components[action]
            pipeline_components.append(component)
            state = next_state
        
        # Evaluate generated pipeline
        print(f"Generated pipeline: {pipeline_components}")
        
        # Run comprehensive pipeline validation
        validation = env.validate_pipeline_results(pipeline_components)
        validation_results[name] = validation
        
        # Store basic results
        results[name] = {
            "pipeline": pipeline_components,
            "test_score": validation["test_performance"]["test_score"],
            "improvement": validation["baseline_comparison"]["improvement"],
            "stability": validation["stability"]["stability"],
            "quality": validation["overall_quality"]
        }
    
    # Save results to JSON
    os.makedirs("results", exist_ok=True)
    with open("results/evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.bar(results.keys(), [r["test_score"] for r in results.values()])
    plt.title("Pipeline Performance Across Datasets")
    plt.xlabel("Dataset")
    plt.ylabel("Test Accuracy")
    plt.ylim(0, 1)
    plt.savefig("evaluation_results.png")
    plt.show()
    
    # Plot improvement over baselines
    plt.figure(figsize=(10, 6))
    plt.bar(results.keys(), [r["improvement"] for r in results.values()])
    plt.title("Improvement Over Best Baseline (%)")
    plt.xlabel("Dataset")
    plt.ylabel("Improvement (%)")
    plt.savefig("baseline_improvement.png")
    plt.show()

if __name__ == "__main__":
    main()