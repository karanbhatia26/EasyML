import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Import your evaluation function
from evaluate import evaluate_pipeline

def main():
    # Define baseline pipelines
    baselines = {
        "Simple LR": [
            "StandardScaler()",
            "LogisticRegression()"
        ],
        "PCA + LR": [
            "StandardScaler()",
            "PCA(n_components=2)",
            "LogisticRegression()"
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
    
    # Your trained model's pipeline (loaded from saved best pipeline)
    your_pipeline = [
        # Load this from your saved best pipeline
        "StandardScaler()",
        "PCA(n_components=5)",
        "RandomForestClassifier(n_estimators=100)"
    ]
    
    results = {}
    
    # Evaluate all pipelines on all datasets
    for dataset_name, data in datasets.items():
        X_train, X_test, y_train, y_test = train_test_split(
            data.data, data.target, test_size=0.2, random_state=42)
            
        dataset = {
            "X_train": X_train,
            "y_train": y_train, 
            "X_test": X_test,
            "y_test": y_test
        }
        
        results[dataset_name] = {}
        
        # Evaluate baselines
        for name, pipeline in baselines.items():
            performance = evaluate_pipeline(pipeline, dataset)
            results[dataset_name][name] = performance
            
        # Evaluate your model
        performance = evaluate_pipeline(your_pipeline, dataset)
        results[dataset_name]["MARL-AutoML"] = performance
    
    # Plot results
    fig, axes = plt.subplots(1, len(datasets), figsize=(15, 5))
    
    for i, (dataset_name, dataset_results) in enumerate(results.items()):
        names = list(dataset_results.keys())
        values = list(dataset_results.values())
        
        axes[i].bar(names, values)
        axes[i].set_title(dataset_name)
        axes[i].set_ylim([0, 1])
        axes[i].set_ylabel("Accuracy")
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig("baseline_comparison.png")
    plt.show()

if __name__ == "__main__":
    main()