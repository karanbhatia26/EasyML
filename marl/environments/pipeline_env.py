import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import gym
from gym import spaces
from marl.utils.component_guide import ComponentTransitionRules
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from marl.environments.ml_components import COMPONENT_MAP

class PipelineEnvironment(gym.Env):
    """Environment for building ML pipelines."""
    
    def __init__(self, dataset, available_components=None, max_pipeline_length=10):
        super().__init__()
        
        # Make sure dataset is a dictionary with all required keys
        required_keys = ['X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test']
        for key in required_keys:
            if key not in dataset:
                raise KeyError(f"Dataset missing required key: {key}")
        
        self.X_train = dataset['X_train'] 
        self.y_train = dataset['y_train']
        self.X_val = dataset['X_val']
        self.y_val = dataset['y_val']
        self.X_test = dataset['X_test']
        self.y_test = dataset['y_test']
        
        # Available ML components
        self.available_components = available_components or self._default_components()
        self.component_ids = {comp: i for i, comp in enumerate(self.available_components)}
        
        # Environment parameters
        self.max_pipeline_length = max_pipeline_length
        
        # Action and observation spaces
        self.action_space = spaces.Discrete(len(self.available_components))
        
        # Observation space: one-hot encoding of current pipeline components
        # plus dataset characteristics
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(len(self.available_components) * max_pipeline_length + 10,), 
            dtype=np.float32
        )
        
        # Current state
        self.current_pipeline = []
        self.last_performance = 0.0
        
        self.transition_rules = ComponentTransitionRules()
        
    def _default_components(self):
        """Define default available components if none provided."""
        components = [
            
            # Imputers for missing values
            "SimpleImputer(strategy='mean')",
            "SimpleImputer(strategy='median')",
            "SimpleImputer(strategy='most_frequent')",
            "SimpleImputer(strategy='constant', fill_value=0)",

            # Scalers
            "StandardScaler()",
            "MinMaxScaler()",
            "RobustScaler()",
            "MaxAbsScaler()",
            "Normalizer()",

            # Encoders
            "OneHotEncoder(drop='first')",
            "OrdinalEncoder()",

            # Feature selection
            "SelectKBest(k=5)",
            "SelectKBest(k=10)",
            "SelectPercentile(percentile=50)",
            "VarianceThreshold(threshold=0.1)",

            # Feature transformation
            "PCA(n_components=2)",
            "PCA(n_components=5)",
            "PCA(n_components=10)",
            "TruncatedSVD(n_components=5)",
            "PolynomialFeatures(degree=2)",

            # Classifiers
            "LogisticRegression(max_iter=1000)",
            "LogisticRegression(max_iter=1000, C=0.1)",
            "LogisticRegression(max_iter=1000, C=10)",
            "DecisionTreeClassifier(max_depth=5)",
            "DecisionTreeClassifier(max_depth=10)",
            "DecisionTreeClassifier(max_depth=None)",
            "RandomForestClassifier(n_estimators=50)",
            "RandomForestClassifier(n_estimators=100)",
            "RandomForestClassifier(n_estimators=200)",
            "GradientBoostingClassifier(n_estimators=100)",
            "KNeighborsClassifier(n_neighbors=5)",

            # End pipeline marker
            "END_PIPELINE"
        ]
        return components
        
    def _get_state_representation(self):
        """Convert current pipeline to state representation."""
        # One-hot encoding of pipeline components
        pipeline_state = np.zeros(len(self.available_components) * self.max_pipeline_length)
        
        for i, component in enumerate(self.current_pipeline):
            if i < self.max_pipeline_length:
                component_id = self.component_ids[component]
                pipeline_state[i * len(self.available_components) + component_id] = 1
                
        # Dataset characteristics - handle numeric vs categorical columns properly
        data_chars = []
        
        # 1. Basic dataset properties
        data_chars.append(self.X_train.shape[0] / 10000)  # Normalized sample count
        data_chars.append(self.X_train.shape[1] / 100)    # Normalized feature count
        data_chars.append(len(np.unique(self.y_train)) / 10)  # Normalized class count
        
        # 2. Identify numeric columns and calculate statistics only on those
        numeric_cols = self.X_train.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            numeric_data = self.X_train[numeric_cols]
            # Calculate variance on numeric columns only
            try:
                data_chars.append(np.mean(numeric_data.var(numeric_only=True)))
            except:
                data_chars.append(0.0)  # Default if calculation fails
        else:
            data_chars.append(0.0)  # No numeric columns
        
        # 3. Count of categorical vs numeric columns
        total_cols = self.X_train.shape[1]
        categorical_cols = total_cols - len(numeric_cols)
        data_chars.append(categorical_cols / total_cols)  # Ratio of categorical features
        
        # 4. Missing value statistics - handle all column types
        missing_counts = self.X_train.isna().sum()
        if len(missing_counts) > 0:
            missing_ratio = missing_counts.sum() / (self.X_train.shape[0] * total_cols)
            data_chars.append(missing_ratio)
        else:
            data_chars.append(0.0)  # No missing values
        
        # Add some placeholder values to ensure consistent shape
        while len(data_chars) < 10:
            data_chars.append(0.0)
        
        # Ensure we don't exceed 10 features
        data_chars = data_chars[:10]
        
        return np.concatenate([pipeline_state, np.array(data_chars)])
        
    def _evaluate_pipeline(self):
        """Evaluate the current pipeline with better error handling."""
        if not self.current_pipeline:
            return 0.0
            
        # Remove END_PIPELINE token if present
        pipeline_components = [comp for comp in self.current_pipeline if comp != "END_PIPELINE"]
        
        # Check if the pipeline ends with a classifier
        if not pipeline_components or not any(model_name in pipeline_components[-1] 
                for model_name in ['LogisticRegression', 'DecisionTreeClassifier', 'RandomForestClassifier', 
                                  'GradientBoostingClassifier', 'KNeighborsClassifier']):
            print("Pipeline evaluation skipped: Pipeline must end with a classifier")
            return 0.0
        
        try:
            # Create a scikit-learn Pipeline object
            steps = []
            
            # First, detect if we need special column handling
            categorical_cols = self.X_train.select_dtypes(include=['category', 'object']).columns
            numeric_cols = self.X_train.select_dtypes(include=['number']).columns
            
            # Only add column handling if we have mixed column types
            if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                from sklearn.compose import ColumnTransformer
                from sklearn.preprocessing import OneHotEncoder, StandardScaler
                from sklearn.impute import SimpleImputer
                
                # Create preprocessing for each column type
                numeric_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ])
                
                categorical_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ])
                
                # Create column transformer
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', numeric_transformer, numeric_cols),
                        ('cat', categorical_transformer, categorical_cols)
                    ])
                
                # Add preprocessor as first step
                steps.append(('preprocessor', preprocessor))
            
            # Add user-selected components
            for i, component_str in enumerate(pipeline_components):
                if component_str == "END_PIPELINE":
                    continue
                    
                if component_str not in COMPONENT_MAP:
                    print(f"Pipeline evaluation failed: Component not found: {component_str}")
                    return 0.0
                    
                component = COMPONENT_MAP[component_str]
                steps.append((f'step_{i}', component))
            
            # Create and evaluate the pipeline using a timeout mechanism that works on Windows
            try:
                # Use a context manager with timeout
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self._fit_and_score_pipeline, steps)
                    try:
                        # Set a 30-second timeout
                        score = future.result(timeout=300)
                        return score
                    except concurrent.futures.TimeoutError:
                        print("Pipeline evaluation timed out after 300 seconds")
                        return 0.0
            except Exception as inner_e:
                print(f"Pipeline evaluation failed during fitting/scoring: \n{str(inner_e)}")
                return 0.0
                
        except Exception as e:
            print(f"Pipeline evaluation failed: \n{str(e)}")
            return 0.0

    def _fit_and_score_pipeline(self, steps):
        """Helper method to fit and score pipeline with proper error handling."""
        try:
            pipeline = Pipeline(steps)
            pipeline.fit(self.X_train, self.y_train)
            score = pipeline.score(self.X_val, self.y_val)
            return score
        except Exception as e:
            print(f"Pipeline fitting/scoring error: {str(e)}")
            return 0.0
    
    def evaluate_pipeline_on_test(self, pipeline_components):
        """Evaluate a pipeline on the true test set to get unbiased performance."""
        steps = []
        
        # Remove END_PIPELINE token if present
        pipeline_components = [comp for comp in pipeline_components if comp != "END_PIPELINE"]
        
        # Check if the pipeline ends with a classifier
        if not pipeline_components or not any(model_name in pipeline_components[-1] 
                for model_name in ['LogisticRegression', 'DecisionTreeClassifier', 'RandomForestClassifier', 
                                 'GradientBoostingClassifier', 'KNeighborsClassifier']):
            print("Pipeline evaluation skipped: Pipeline must end with a classifier")
            return {'val_score': 0.0, 'test_score': 0.0, 'gap': 0.0}
        
        try:
            # Add user-selected components
            for i, component_str in enumerate(pipeline_components):
                if component_str == "END_PIPELINE":
                    continue
                    
                if component_str not in COMPONENT_MAP:
                    print(f"Pipeline evaluation failed: Component not found: {component_str}")
                    return {'val_score': 0.0, 'test_score': 0.0, 'gap': 0.0}
                    
                component = COMPONENT_MAP[component_str]
                steps.append((f'step_{i}', component))
            
            pipeline = Pipeline(steps)
            pipeline.fit(self.X_train, self.y_train)  # Train on training data
            val_score = pipeline.score(self.X_val, self.y_val)  # Validation score
            test_score = pipeline.score(self.X_test, self.y_test)  # True test score
            
            return {
                'val_score': val_score,
                'test_score': test_score,
                'gap': val_score - test_score  # Overfitting indicator
            }
        except Exception as e:
            print(f"Test evaluation error: {str(e)}")
            return {'val_score': 0.0, 'test_score': 0.0, 'gap': 0.0}

    def compare_with_baselines(self, pipeline_components):
        """Compare your pipeline with baseline models."""
        from sklearn.dummy import DummyClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        
        # Your best pipeline
        pipeline_result = self.evaluate_pipeline_on_test(pipeline_components)
        
        # Train baseline models
        baselines = {}
        
        # 1. Majority class baseline
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(self.X_train, self.y_train)
        baselines["Majority Class"] = dummy.score(self.X_test, self.y_test)
        
        # 2. Simple Logistic Regression
        lr = LogisticRegression(max_iter=1000)
        lr.fit(self.X_train, self.y_train)
        baselines["Logistic Regression"] = lr.score(self.X_test, self.y_test)
        
        # 3. RandomForest with default params
        rf = RandomForestClassifier(n_estimators=100)
        rf.fit(self.X_train, self.y_train)
        baselines["Random Forest"] = rf.score(self.X_test, self.y_test)
        
        # Compare results
        results = {
            "Your Pipeline": pipeline_result['test_score'],
            **baselines
        }
        
        # Print comparison
        print("\nModel Performance Comparison:")
        print("="*40)
        for model, score in results.items():
            print(f"{model:20s}: {score:.4f}")
        
        # Return improvement percentage over best baseline
        best_baseline = max(baselines.values())
        improvement = (pipeline_result['test_score'] - best_baseline) / best_baseline * 100 if best_baseline > 0 else 0
        
        return {
            'pipeline_score': pipeline_result['test_score'],
            'best_baseline': best_baseline,
            'improvement': improvement,
            'all_results': results
        }

    def cross_validate_pipeline(self, pipeline_components, cv=5):
        """Test pipeline stability through cross-validation."""
        from sklearn.model_selection import cross_val_score
        import numpy as np
        
        # Remove END_PIPELINE token if present
        pipeline_components = [comp for comp in pipeline_components if comp != "END_PIPELINE"]
        
        # Create steps
        steps = []
        for i, component_str in enumerate(pipeline_components):
            if component_str == "END_PIPELINE":
                continue
                
            if component_str not in COMPONENT_MAP:
                print(f"Pipeline evaluation failed: Component not found: {component_str}")
                return {'mean_score': 0.0, 'std_score': 0.0, 'stability': 0.0}
                
            component = COMPONENT_MAP[component_str]
            steps.append((f'step_{i}', component))
        
        try:
            pipeline = Pipeline(steps)
            
            # Combine train and val sets for cross-validation
            import pandas as pd
            X_combined = pd.concat([self.X_train, self.X_val])
            y_combined = np.concatenate([self.y_train, self.y_val])
            
            # Perform cross-validation
            cv_scores = cross_val_score(pipeline, X_combined, y_combined, cv=cv)
            
            return {
                'mean_score': cv_scores.mean(),
                'std_score': cv_scores.std(),
                'all_scores': cv_scores,
                'stability': 1 - (cv_scores.std() / cv_scores.mean()) if cv_scores.mean() > 0 else 0  # Higher is more stable
            }
        except Exception as e:
            print(f"Cross-validation error: {str(e)}")
            return {'mean_score': 0.0, 'std_score': 0.0, 'stability': 0.0}

    def validate_pipeline_results(self, best_pipeline):
        """Comprehensive validation of pipeline results."""
        print(f"\n{'='*20} PIPELINE VALIDATION {'='*20}\n")
        
        # 1. Test set performance
        print("\n[1] Test Set Performance:")
        test_results = self.evaluate_pipeline_on_test(best_pipeline)
        print(f"Validation score: {test_results['val_score']:.4f}")
        print(f"Test score: {test_results['test_score']:.4f}")
        print(f"Gap (overfitting indicator): {test_results['gap']:.4f}")
        
        # 2. Baseline comparison
        print("\n[2] Baseline Comparison:")
        baseline_results = self.compare_with_baselines(best_pipeline)
        print(f"Improvement over best baseline: {baseline_results['improvement']:.2f}%")
        
        # 3. Stability check
        print("\n[3] Stability Analysis (Cross-validation):")
        cv_results = self.cross_validate_pipeline(best_pipeline)
        print(f"Mean CV score: {cv_results['mean_score']:.4f} ± {cv_results['std_score']:.4f}")
        print(f"Stability score: {cv_results['stability']:.4f} (higher is better)")
        
        # 4. Summary
        print("\n[4] Final Assessment:")
        overall_quality = "Poor"
        if test_results['test_score'] > 0.7 and baseline_results['improvement'] > 5 and cv_results['stability'] > 0.9:
            overall_quality = "Excellent"
        elif test_results['test_score'] > 0.6 and baseline_results['improvement'] > 2 and cv_results['stability'] > 0.8:
            overall_quality = "Good"
        elif test_results['test_score'] > 0.5 and baseline_results['improvement'] > 0:
            overall_quality = "Fair"
        
        print(f"Overall pipeline quality: {overall_quality}")
        
        return {
            'test_performance': test_results,
            'baseline_comparison': baseline_results,
            'stability': cv_results,
            'overall_quality': overall_quality
        }
    
    def reset(self):
        """Reset the environment for a new episode."""
        self.current_pipeline = []
        self.last_performance = 0.0
        return self._get_state_representation()
    
    def step(self, action):
        """Add a component to the pipeline and evaluate it."""
        component = self.available_components[action]
        
        # Check if pipeline already contains END_PIPELINE or a classifier (terminal state)
        if any(comp == "END_PIPELINE" for comp in self.current_pipeline):
            print("Pipeline already ended - ignoring additional components")
            return self._get_state_representation(), -0.5, True, {"performance": 0.0}
            
        # Check if we already have a classifier
        if any(model_name in ' '.join(self.current_pipeline) 
               for model_name in ['LogisticRegression', 'DecisionTreeClassifier', 'RandomForestClassifier',
                                  'GradientBoostingClassifier', 'KNeighborsClassifier']):
            # Only allow END_PIPELINE after a classifier
            if component != "END_PIPELINE":
                print("Pipeline already has classifier - only END_PIPELINE allowed")
                return self._get_state_representation(), -0.5, False, {"performance": 0.0}
        
        # Check if this is the END_PIPELINE action
        if component == "END_PIPELINE":
            # Only end if there's at least one component and the last one is a classifier
            if len(self.current_pipeline) > 0 and any(model_name in self.current_pipeline[-1] 
                                                   for model_name in ['LogisticRegression', 'DecisionTreeClassifier', 'RandomForestClassifier',
                                                                    'GradientBoostingClassifier', 'KNeighborsClassifier']):
                self.current_pipeline.append(component)  # Add END_PIPELINE as marker
                done = True
                next_state = self._get_state_representation()
                current_performance = self._evaluate_pipeline()
                return next_state, 0.0, done, {"performance": current_performance}
            else:
                # Trying to end without a classifier, penalize
                print("Cannot end pipeline - last component is not a classifier")
                return self._get_state_representation(), -0.5, False, {"performance": 0.0}
        
        # Calculate repetition penalty - ADD THIS SECTION
        repetition_penalty = 0.0
        if component in self.current_pipeline:
            # Count how many times this component is already in pipeline
            component_count = self.current_pipeline.count(component)
            # Increase penalty for each repetition
            repetition_penalty = -0.05 * component_count
        
        # Original code for regular components
        self.current_pipeline.append(component)
        done = len(self.current_pipeline) >= self.max_pipeline_length
        
        # Evaluate the pipeline
        next_state = self._get_state_representation()
        
        if done:
            # If pipeline is complete, evaluate it
            current_performance = self._evaluate_pipeline()
            # Final reward = performance - penalties
            reward = current_performance + repetition_penalty
            return next_state, reward, done, {"performance": current_performance}
        else:
            # Intermediate reward with penalty
            reward = -0.01 + repetition_penalty  # Small negative reward for each step
            return next_state, reward, done, {"performance": 0.0}
    
    def get_valid_actions(self):
        """Return indices of valid next components based on compatibility."""
        # If pipeline at max length, no more actions
        if len(self.current_pipeline) >= self.max_pipeline_length:
            return []
            
        # Get base valid actions from previous logic
        base_valid_actions = []
        
        # If we already have a classifier, only allow END_PIPELINE
        has_classifier = any(model_name in ' '.join(self.current_pipeline) 
                          for model_name in ['LogisticRegression', 'DecisionTreeClassifier', 
                                             'RandomForestClassifier', 'GradientBoostingClassifier', 'KNeighborsClassifier'])
        
        if has_classifier:
            # Find END_PIPELINE action index
            for i, comp in enumerate(self.available_components):
                if comp == "END_PIPELINE":
                    return [i]
            return []
        
        # At the last position, must add a classifier
        if len(self.current_pipeline) == self.max_pipeline_length - 1:
            return [i for i, comp in enumerate(self.available_components) 
                   if any(model_name in comp for model_name in 
                         ['LogisticRegression', 'DecisionTreeClassifier', 'RandomForestClassifier', 
                          'GradientBoostingClassifier', 'KNeighborsClassifier'])]
        
        # Otherwise, filter valid actions based on compatibility
        valid_actions = []
        for i, comp_name in enumerate(self.available_components):
            if self._is_compatible_with_pipeline(comp_name):
                valid_actions.append(i)
        
        return valid_actions

    def _is_compatible_with_pipeline(self, component_name):
        """Check if a component is compatible with the current pipeline."""
        # Nothing is incompatible with an empty pipeline
        if not self.current_pipeline:
            return True

        # Get component categories
        feature_extractors = ['PCA', 'TruncatedSVD', 'SelectKBest']
        encoders = ['OneHotEncoder', 'OrdinalEncoder']
        scalers = ['StandardScaler', 'MinMaxScaler', 'MaxAbsScaler', 'RobustScaler', 'Normalizer']
        imputers = ['SimpleImputer']
        classifiers = ['LogisticRegression', 'DecisionTreeClassifier', 'RandomForestClassifier', 'KNeighborsClassifier', 'GradientBoostingClassifier']
        
        # Check if we already have dimension reduction
        has_dim_reduction = any(extractor in ' '.join(self.current_pipeline) for extractor in feature_extractors)
        
        # Check if previous component is a classifier
        prev_is_classifier = any(clf in self.current_pipeline[-1] for clf in classifiers)
        
        # Check for incompatibilities
        
        # 1. Don't add anything after a classifier except END_PIPELINE
        if prev_is_classifier and component_name != 'END_PIPELINE':
            print(f"Incompatible: Cannot add {component_name} after a classifier")
            return False
        
        # 2. Don't add the same imputer/scaler type if already present
        component_type = None
        for cat in [imputers, scalers, encoders, feature_extractors]:
            if any(c in component_name for c in cat):
                component_type = cat
                break
                
        if component_type:
            existing_same_type = [c for c in self.current_pipeline 
                                 if any(t in c for t in component_type)]
            if len(existing_same_type) >= 1:
                print(f"Redundant: Pipeline already contains {existing_same_type}")
                return False
                
        # 3. Don't apply encoders after numerical transformations
        if any(encoder in component_name for encoder in encoders) and has_dim_reduction:
            print(f"Incompatible: Cannot apply {component_name} after dimension reduction")
            return False
            
        # 4. Check for specific component sequence issues
        if 'PCA' in component_name:
            # Make sure PCA n_components is not too large 
            n_components = int(component_name.split('(n_components=')[1].split(')')[0])
            max_features = min(self.X_train.shape[0], self.X_train.shape[1])
            if n_components > max_features:
                print(f"Invalid: PCA n_components={n_components} exceeds max of {max_features}")
                return False
                
        return True

def is_transformer(component_name):
    """Check if a component is a transformer rather than classifier."""
    classifier_names = [
        'LogisticRegression', 'DecisionTreeClassifier', 'RandomForestClassifier',
        'GradientBoostingClassifier', 'KNeighborsClassifier'
    ]
    return not any(clf in component_name for clf in classifier_names)

def is_classifier(component_name):
    """Check if a component is a classifier."""
    classifier_names = [
        'LogisticRegression', 'DecisionTreeClassifier', 'RandomForestClassifier',
        'GradientBoostingClassifier', 'KNeighborsClassifier'
    ]
    return any(clf in component_name for clf in classifier_names)