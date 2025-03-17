import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from marl.environments.pipeline_env import PipelineEnvironment
import json
import os

def main():
    # Define baseline pipelines
    baselines = {
        "Simple LR": [
            "StandardScaler()",
            "LogisticRegression(max_iter=1000)"
        ],
        "PCA + LR": [
            "StandardScaler()",
            "PCA(n_components=2)",
            "LogisticRegression(max_iter=1000)"
        ],
        "RF": [
            "StandardScaler()",
            "RandomForestClassifier(n_estimators=100)"
        ],
        "SVM": [
            "StandardScaler()",
            "SVC(probability=True)"
        ]
    }
    
    # Load datasets
    datasets = {
        "iris": load_iris(),
        "wine": load_wine(),
        "breast_cancer": load_breast_cancer()
    }
    
    # Your trained model's pipeline (load from results if available)
    try:
        with open("results/evaluation_results.json", "r") as f:
            eval_results = json.load(f)
            marl_pipelines = {dataset: result["pipeline"] for dataset, result in eval_results.items()}
    except (FileNotFoundError, json.JSONDecodeError):
        # Default pipeline if results not available
        marl_pipelines = {
            "iris": ["StandardScaler()", "RandomForestClassifier(n_estimators=100)"],
            "wine": ["StandardScaler()", "RandomForestClassifier(n_estimators=100)"],
            "breast_cancer": ["StandardScaler()", "RandomForestClassifier(n_estimators=100)"]
        }
    
    results = {}
    detailed_results = {}
    
    # Evaluate all pipelines on all datasets
    for dataset_name, data in datasets.items():
        print(f"\n\n{'='*40}")
        print(f"Evaluating on {dataset_name} dataset...")
        print(f"{'='*40}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            data.data, data.target, test_size=0.2, random_state=42)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42)
            
        dataset_dict = {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
            "feature_names": data.feature_names,
            "n_classes": len(np.unique(data.target))
        }
        
        # Create environment for evaluation
        env = PipelineEnvironment(dataset_dict)
        
        results[dataset_name] = {}
        detailed_results[dataset_name] = {}
        
        # Evaluate baselines
        for name, pipeline in baselines.items():
            print(f"\nEvaluating {name}...")
            test_results = env.evaluate_pipeline_on_test(pipeline)
            cv_results = env.cross_validate_pipeline(pipeline)
            
            results[dataset_name][name] = test_results["test_score"]
            detailed_results[dataset_name][name] = {
                "test_score": test_results["test_score"],
                "val_score": test_results["val_score"],
                "cv_score": cv_results["mean_score"],
                "stability": cv_results["stability"]
            }
            
        # Evaluate your MARL model
        marl_pipeline = marl_pipelines.get(dataset_name, ["StandardScaler()", "RandomForestClassifier(n_estimators=100)"])
        print(f"\nEvaluating MARL-AutoML pipeline: {marl_pipeline}")
        
        # Run full validation
        validation = env.validate_pipeline_results(marl_pipeline)
        
        # Store results
        results[dataset_name]["MARL-AutoML"] = validation["test_performance"]["test_score"]
        detailed_results[dataset_name]["MARL-AutoML"] = {
            "test_score": validation["test_performance"]["test_score"],
            "val_score": validation["test_performance"]["val_score"],
            "cv_score": validation["stability"]["mean_score"],
            "stability": validation["stability"]["stability"],
            "improvement": validation["baseline_comparison"]["improvement"],
            "quality": validation["overall_quality"]
        }
    
    # Save detailed results
    os.makedirs("results", exist_ok=True)
    with open("results/baseline_comparison.json", "w") as f:
        json.dump(detailed_results, f, indent=4)
    
    # Plot results
    fig, axes = plt.subplots(1, len(datasets), figsize=(15, 5))
    
    for i, (dataset_name, dataset_results) in enumerate(results.items()):
        names = list(dataset_results.keys())
        values = list(dataset_results.values())
        
        # Highlight MARL-AutoML in a different color
        colors = ['blue' if name != 'MARL-AutoML' else 'red' for name in names]
        
        axes[i].bar(names, values, color=colors)
        axes[i].set_title(dataset_name)
        axes[i].set_ylim([0, 1])
        axes[i].set_ylabel("Test Accuracy")
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig("baseline_comparison.png")
    plt.show()
    
    # Plot stability comparison
    plt.figure(figsize=(12, 6))
    
    for dataset_name in datasets.keys():
        stability_values = [detailed_results[dataset_name][model]["stability"] 
                          for model in detailed_results[dataset_name]]
        plt.plot(list(detailed_results[dataset_name].keys()), stability_values, 
                marker='o', label=dataset_name)
    
    plt.title("Model Stability Comparison (Cross-validation)")
    plt.xlabel("Model")
    plt.ylabel("Stability Score (higher is better)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("stability_comparison.png")
    plt.show()

if __name__ == "__main__":
    main()