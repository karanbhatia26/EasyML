from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import pandas as pd
import multiprocessing
import time
import warnings
from typing import Dict, List, Tuple, Union, Optional
import os
import sys
import random

try:
    from tpot import TPOTClassifier
    TPOT_AVAILABLE = True
except ImportError:
    TPOT_AVAILABLE = False
    print("TPOT not available. Install with: pip install tpot")

try:
    import autosklearn.classification
    AUTOSKLEARN_AVAILABLE = True
except ImportError:
    AUTOSKLEARN_AVAILABLE = False
    print("Auto-sklearn not available. Install with: pip install auto-sklearn")

# Define MARL components for your AutoML approach
class PipelineEnvironment:
    """Environment for pipeline construction"""
    
    def __init__(self, X_train, y_train, X_val, y_val, cv=3):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.cv = cv
        self.preprocessors = [
            ("imputer", SimpleImputer(strategy="mean")),
            ("imputer", SimpleImputer(strategy="median")),
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("scaler", StandardScaler()),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
        self.classifiers = [
            LogisticRegression(max_iter=1000),
            RandomForestClassifier(n_estimators=100, random_state=42)
        ]
        
    def evaluate_pipeline(self, pipeline: Pipeline) -> float:
        """Evaluate a pipeline and return accuracy score"""
        try:
            pipeline.fit(self.X_train, self.y_train)
            y_pred = pipeline.predict(self.X_val)
            return accuracy_score(self.y_val, y_pred)
        except Exception as e:
            print(f"Pipeline evaluation failed: {str(e)}")
            return 0.0


class StudentAgent:
    """Student agent that learns to construct ML pipelines"""
    
    def __init__(self, env: PipelineEnvironment, learning_rate=0.1, epsilon=0.3):
        self.env = env
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.q_values = {}  # State-action values
        self.experiences = []  # (state, action, reward, next_state) tuples
        
    def select_action(self, state: Tuple, explore=True) -> int:
        """Select preprocessing or model component based on current state"""
        if state not in self.q_values:
            self.q_values[state] = [0.1] * (len(self.env.preprocessors) + len(self.env.classifiers))
            
        if explore and random.random() < self.epsilon:
            return random.randint(0, len(self.q_values[state]) - 1)
        else:
            return np.argmax(self.q_values[state])
            
    def update_q_values(self, state, action, reward, next_state):
        """Update Q-values based on experience"""
        if state not in self.q_values:
            self.q_values[state] = [0.1] * (len(self.env.preprocessors) + len(self.env.classifiers))
            
        if next_state not in self.q_values:
            self.q_values[next_state] = [0.1] * (len(self.env.preprocessors) + len(self.env.classifiers))
            
        max_next_q = max(self.q_values[next_state])
        self.q_values[state][action] += self.learning_rate * (reward + 0.9 * max_next_q - self.q_values[state][action])


class TeacherAgent:
    """Teacher agent that guides the student's learning process"""
    
    def __init__(self, env: PipelineEnvironment, intervention_threshold=0.3):
        self.env = env
        self.intervention_threshold = intervention_threshold
        self.knowledge_base = {}  # Prior knowledge about what works well
        self.intervention_rate = 0.5  # Starts with 50% intervention
        
    def should_intervene(self, state, student_choice) -> bool:
        """Decide whether to intervene in student's decision"""
        # Initially intervene more, then gradually reduce
        if random.random() > self.intervention_rate:
            return False
            
        if state in self.knowledge_base and self.knowledge_base[state] != student_choice:
            # Intervene if we know a better choice
            return True
        return False
        
    def select_action(self, state) -> int:
        """Select a better action based on teacher's knowledge"""
        if state in self.knowledge_base:
            return self.knowledge_base[state]
        else:
            # Fall back to a safe default
            return len(self.env.preprocessors) - 1  # Last preprocessor
            
    def update_knowledge(self, state, action, reward):
        """Update teacher's knowledge based on outcomes"""
        if state not in self.knowledge_base or reward > self.knowledge_base.get(state, {}).get("reward", 0):
            self.knowledge_base[state] = {"action": action, "reward": reward}
            
    def adapt_intervention_rate(self, student_performance):
        """Reduce intervention as student improves"""
        self.intervention_rate = max(0.1, self.intervention_rate - 0.01 * student_performance)


class CreditAssignment:
    """Distributes credit for pipeline performance to components"""
    
    def __init__(self):
        self.component_values = {}
        
    def assign_credit(self, pipeline_components, performance):
        """Distribute performance credit to pipeline components"""
        # Simple credit assignment - equal distribution
        credit_per_component = performance / len(pipeline_components)
        
        for component in pipeline_components:
            if component not in self.component_values:
                self.component_values[component] = []
            self.component_values[component].append(credit_per_component)
            
    def get_component_value(self, component):
        """Get average performance contribution of a component"""
        if component in self.component_values and len(self.component_values[component]) > 0:
            return sum(self.component_values[component]) / len(self.component_values[component])
        return 0.1  # Default value for new components


class MARLAutoML:
    """Main class implementing MARL for AutoML pipeline construction"""
    
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test, 
                 n_iterations=50, cv=3):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.n_iterations = n_iterations
        self.cv = cv
        
        # Initialize environment and agents
        self.env = PipelineEnvironment(X_train, y_train, X_val, y_val, cv)
        self.student = StudentAgent(self.env)
        self.teacher = TeacherAgent(self.env)
        self.credit_assignment = CreditAssignment()
        
        self.best_pipeline = None
        self.best_score = 0.0
        
    def construct_pipeline(self, components):
        """Construct a scikit-learn pipeline from components"""
        # Separate preprocessors and classifier
        preprocessors = [c for c in components if c[0] in ('imputer', 'scaler', 'encoder')]
        classifiers = [c for c in components if c not in preprocessors]
        
        # For simplicity, we'll use the first classifier found
        classifier = classifiers[0] if classifiers else self.env.classifiers[0]
        
        # Create preprocessing steps for numerical and categorical features
        numeric_cols = self.X_train.select_dtypes(include=['number']).columns
        categorical_cols = self.X_train.select_dtypes(include=['category', 'object']).columns
        
        numeric_steps = [p for p in preprocessors if p[0] in ('imputer', 'scaler')]
        categorical_steps = [p for p in preprocessors if p[0] in ('imputer', 'encoder')]
        
        numeric_transformer = Pipeline(steps=numeric_steps if numeric_steps else [("imputer", SimpleImputer(strategy="mean"))])
        categorical_transformer = Pipeline(steps=categorical_steps if categorical_steps else [("encoder", OneHotEncoder(handle_unknown="ignore"))])
        
        preprocessor = ColumnTransformer(transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ])
        
        # Create full pipeline
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ])
        
        return pipeline
        
    def run(self):
        """Run the MARL AutoML algorithm"""
        start_time = time.time()
        
        for iteration in range(self.n_iterations):
            # Current state is represented as a tuple of current pipeline components
            state = tuple()
            components = []
            
            # Build pipeline step by step
            for step in range(5):  # Limit to 5 steps for simplicity
                # Student selects action
                student_action = self.student.select_action(state)
                
                # Teacher may intervene
                if self.teacher.should_intervene(state, student_action):
                    action = self.teacher.select_action(state)
                    intervention = True
                else:
                    action = student_action
                    intervention = False
                
                # Convert action to actual component
                if action < len(self.env.preprocessors):
                    component = self.env.preprocessors[action]
                else:
                    component = self.env.classifiers[action - len(self.env.preprocessors)]
                
                components.append(component)
                
                # Move to next state
                next_state = state + (action,)
                
                # If we added a classifier, we're done building this pipeline
                if action >= len(self.env.preprocessors):
                    break
                
                state = next_state
            
            # Evaluate the constructed pipeline
            pipeline = self.construct_pipeline(components)
            score = self.env.evaluate_pipeline(pipeline)
            
            # Update best pipeline if better
            if score > self.best_score:
                self.best_pipeline = pipeline
                self.best_score = score
                print(f"Iteration {iteration}: New best score {score:.4f}")
            
            # Assign credit to components
            self.credit_assignment.assign_credit(components, score)
            
            # Teacher updates knowledge
            self.teacher.update_knowledge(state, action, score)
            
            # Student learns from experience
            self.student.update_q_values(state, action, score, next_state)
            
            # Adapt teacher intervention rate
            self.teacher.adapt_intervention_rate(score)
            
            # Reduce student's exploration rate over time
            self.student.epsilon = max(0.1, self.student.epsilon * 0.99)
            
        # Evaluate best pipeline on test data
        if self.best_pipeline:
            self.best_pipeline.fit(pd.concat([self.X_train, self.X_val]), 
                                pd.concat([self.y_train, self.y_val]))
            y_pred = self.best_pipeline.predict(self.X_test)
            test_acc = accuracy_score(self.y_test, y_pred)
            print(f"MARL AutoML completed in {time.time() - start_time:.2f} seconds")
            print(f"Best pipeline test accuracy: {test_acc:.4f}")
            return test_acc, self.best_pipeline
        else:
            print("No valid pipeline found")
            return 0.0, None


def main():
    """Main function to run all benchmarks"""
    print("Loading Adult Income dataset...")
    data = fetch_openml(name='adult', version=2, as_frame=True)
    X = data.data
    y = data.target
    y = y.astype(str)
    
    # Basic dataset information
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {X.columns.tolist()}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Train-validation-test split
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.25, random_state=42
    )  # 0.25 * 0.8 = 0.2 of original data
    
    print(f"Train set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")
    
    # Identify categorical and numerical features
    categorical_cols = X.select_dtypes(include=['category', 'object']).columns
    numeric_cols = X.select_dtypes(include=['number']).columns
    
    print(f"Categorical features: {len(categorical_cols)}, Numerical features: {len(numeric_cols)}")
    
    # Create preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ])
    
    # Preprocess data for methods that expect preprocessed data
    print("Preprocessing data...")
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_val_preprocessed = preprocessor.transform(X_val)
    X_test_preprocessed = preprocessor.transform(X_test)
    
    # Store results
    results = {}
    
    # =================== 1. GRID SEARCH ===================
    print("\n======= Running Grid Search =======")
    start_time = time.time()
    
    logreg_grid = {
        "classifier__C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "classifier__solver": ["liblinear", "saga"]
    }
    
    grid_search = GridSearchCV(
        Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000))
        ]), 
        logreg_grid, 
        cv=3, 
        n_jobs=1,
        scoring="accuracy"
    )
    
    grid_search.fit(X_train, y_train)
    grid_time = time.time() - start_time
    
    # Evaluate on test set
    grid_preds = grid_search.predict(X_test)
    grid_acc = accuracy_score(y_test, grid_preds)
    grid_f1 = f1_score(y_test, grid_preds, average='weighted')
    
    results["Grid Search"] = {
        "accuracy": grid_acc,
        "f1_score": grid_f1,
        "training_time": grid_time,
        "best_params": grid_search.best_params_
    }
    
    print(f"Grid Search completed in {grid_time:.2f} seconds")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Test accuracy: {grid_acc:.4f}")
    print(f"Test F1 score: {grid_f1:.4f}")
    
    # =================== 2. RANDOM SEARCH ===================
    print("\n======= Running Random Search =======")
    start_time = time.time()
    
    random_search = RandomizedSearchCV(
        Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000))
        ]), 
        {
            "classifier__C": np.logspace(-3, 3, 20),
            "classifier__solver": ["liblinear", "saga", "lbfgs"]
        }, 
        n_iter=10, 
        cv=3, 
        n_jobs=1, 
        random_state=42,
        scoring="accuracy"
    )
    
    random_search.fit(X_train, y_train)
    random_time = time.time() - start_time
    
    # Evaluate on test set
    random_preds = random_search.predict(X_test)
    random_acc = accuracy_score(y_test, random_preds)
    random_f1 = f1_score(y_test, random_preds, average='weighted')
    
    results["Random Search"] = {
        "accuracy": random_acc,
        "f1_score": random_f1,
        "training_time": random_time,
        "best_params": random_search.best_params_
    }
    
    print(f"Random Search completed in {random_time:.2f} seconds")
    print(f"Best parameters: {random_search.best_params_}")
    print(f"Test accuracy: {random_acc:.4f}")
    print(f"Test F1 score: {random_f1:.4f}")
    
    # =================== 3. TPOT ===================
    if TPOT_AVAILABLE:
        print("\n======= Running TPOT =======")
        start_time = time.time()
        
        try:
            # Use a smaller subset for faster training
            sample_size = min(5000, len(X_train))
            indices = np.random.choice(len(X_train), size=sample_size, replace=False)
            X_train_sample = X_train.iloc[indices]
            y_train_sample = y_train.iloc[indices]
            
            tpot = TPOTClassifier(
                generations=5,
                population_size=20,
                verbose=2,
                max_time_mins=15,
                cv=3,
                random_state=42,
                n_jobs=1
            )
            
            # NOTE: Pass RAW data to TPOT, not preprocessed
            tpot.fit(X_train_sample, y_train_sample)
            tpot_time = time.time() - start_time
            
            # Evaluate on test set
            tpot_preds = tpot.predict(X_test)
            tpot_acc = accuracy_score(y_test, tpot_preds)
            tpot_f1 = f1_score(y_test, tpot_preds, average='weighted')
            
            results["TPOT"] = {
                "accuracy": tpot_acc,
                "f1_score": tpot_f1,
                "training_time": tpot_time,
                "best_pipeline": str(tpot.fitted_pipeline_)
            }
            
            print(f"TPOT completed in {tpot_time:.2f} seconds")
            print(f"Test accuracy: {tpot_acc:.4f}")
            print(f"Test F1 score: {tpot_f1:.4f}")
            print(f"Best pipeline: {tpot.fitted_pipeline_}")
            
        except Exception as e:
            print(f"TPOT training failed: {str(e)}")
            print("Falling back to basic RandomForest classifier")
            
            # Fallback to simple RandomForest
            start_time = time.time()
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_train_preprocessed, y_train)
            rf_time = time.time() - start_time
            
            rf_preds = rf.predict(X_test_preprocessed)
            rf_acc = accuracy_score(y_test, rf_preds)
            rf_f1 = f1_score(y_test, rf_preds, average='weighted')
            
            results["TPOT (Fallback RF)"] = {
                "accuracy": rf_acc,
                "f1_score": rf_f1,
                "training_time": rf_time
            }
            
            print(f"Fallback RF completed in {rf_time:.2f} seconds")
            print(f"Test accuracy: {rf_acc:.4f}")
            print(f"Test F1 score: {rf_f1:.4f}")
    else:
        print("TPOT is not available. Skipping TPOT benchmark.")
    
    # =================== 4. Auto-sklearn ===================
    if AUTOSKLEARN_AVAILABLE:
        print("\n======= Running Auto-sklearn =======")
        start_time = time.time()
        
        try:
            # Use a smaller subset for faster training
            sample_size = min(5000, len(X_train))
            indices = np.random.choice(len(X_train), size=sample_size, replace=False)
            X_train_sample = X_train.iloc[indices]
            y_train_sample = y_train.iloc[indices]
            
            # Create Auto-sklearn classifier
            automl = autosklearn.classification.AutoSklearnClassifier(
                time_left_for_this_task=900,  # 15 minutes
                per_run_time_limit=60,      # 1 minute per run
                memory_limit=4096,          # 4GB
                n_jobs=1,
                ensemble_size=1,            # Smaller ensemble for speed
                initial_configurations_via_metalearning=0
            )
            
            # NOTE: Auto-sklearn needs raw data, not preprocessed
            automl.fit(X_train_sample, y_train_sample)
            autoskl_time = time.time() - start_time
            
            # Evaluate on test set
            autoskl_preds = automl.predict(X_test)
            autoskl_acc = accuracy_score(y_test, autoskl_preds)
            autoskl_f1 = f1_score(y_test, autoskl_preds, average='weighted')
            
            results["Auto-sklearn"] = {
                "accuracy": autoskl_acc,
                "f1_score": autoskl_f1,
                "training_time": autoskl_time,
                "models": str(automl.show_models())
            }
            
            print(f"Auto-sklearn completed in {autoskl_time:.2f} seconds")
            print(f"Test accuracy: {autoskl_acc:.4f}")
            print(f"Test F1 score: {autoskl_f1:.4f}")
            
        except Exception as e:
            print(f"Auto-sklearn training failed: {str(e)}")
            print("Skipping Auto-sklearn benchmark")
    else:
        print("Auto-sklearn is not available. Skipping Auto-sklearn benchmark.")
    
    # =================== 5. Your MARL AutoML ===================
    print("\n======= Running MARL AutoML =======")
    
    # Since this is computationally intensive, we'll use a sample
    sample_size = min(5000, len(X_train))
    indices = np.random.choice(len(X_train), size=sample_size, replace=False)
    X_train_sample = X_train.iloc[indices]
    y_train_sample = y_train.iloc[indices]
    
    indices = np.random.choice(len(X_val), size=min(1000, len(X_val)), replace=False)
    X_val_sample = X_val.iloc[indices]
    y_val_sample = y_val.iloc[indices]
    
    try:
        # Try to import your custom MARL implementation
        sys.path.append(os.path.join(os.getcwd(), 'marl'))
        from marl.agents.student import StudentAgent
        from marl.agents.teacher import TeacherAgent
        from marl.environments.pipeline_env import PipelineEnvironment
        from marl.utils.credit_assignment import CreditAssignment
        
        # Create dataset dictionary for MARL
        dataset = {
            'X_train': X_train_sample,
            'X_test': X_test,
            'X_val': X_val_sample,
            'y_train': y_train_sample, 
            'y_test': y_test,
            'y_val': y_val_sample,
            'feature_names': X_train.columns,
            'n_classes': len(np.unique(y_train))
        }
        
        # Use your trained models if available
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        student_path = os.path.join(model_dir, f"student_model_marl_adult.pt")
        teacher_path = os.path.join(model_dir, f"teacher_model_marl_adult.pt")
        
        print("Using pre-trained MARL models for AutoML")
        start_time = time.time()
        
        # Run a short training session with the existing models
        from marl.train import marl_training
        env = marl_training(dataset_name="adult", episodes=30)
        
        # Get best pipeline and performance
        marl_acc = env.best_performance if hasattr(env, 'best_performance') else env.last_performance
        marl_pipeline = env.best_pipeline if hasattr(env, 'best_pipeline') else env.current_pipeline
        marl_time = time.time() - start_time
        
        # Evaluate on full test data
        if hasattr(env, 'evaluate_pipeline'):
            marl_f1 = env.evaluate_pipeline(marl_pipeline, metric='f1')
        else:
            # Fallback evaluation
            marl_preds = env.predict(X_test)
            marl_f1 = f1_score(y_test, marl_preds, average='weighted')
        
    except (ImportError, Exception) as e:
        print(f"Could not use custom MARL implementation: {str(e)}")
        print("Falling back to simplified MARL implementation")
        
        # Use the simplified implementation
        start_time = time.time()
        marl_automl = MARLAutoML(
            X_train=X_train_sample,
            y_train=y_train_sample,
            X_val=X_val_sample,
            y_val=y_val_sample,
            X_test=X_test,
            y_test=y_test,
            n_iterations=30,
            cv=3
        )
        
        marl_acc, marl_pipeline = marl_automl.run()
        marl_time = time.time() - start_time
        
        if marl_pipeline:
            marl_preds = marl_pipeline.predict(X_test)
            marl_f1 = f1_score(y_test, marl_preds, average='weighted')
        else:
            marl_f1 = 0.0
    
    results["MARL AutoML"] = {
        "accuracy": marl_acc,
        "f1_score": marl_f1,
        "training_time": marl_time
    }
    
    print("\n======= BENCHMARK SUMMARY =======")
    print(f"{'Method':<20} {'Accuracy':<10} {'F1 Score':<10} {'Time (s)':<10}")
    print("-" * 50)
    
    for method, metrics in results.items():
        print(f"{method:<20} {metrics['accuracy']:.4f}     {metrics['f1_score']:.4f}     {metrics['training_time']:.1f}")
    
    return results

if __name__ == "__main__":
    multiprocessing.freeze_support()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = main()